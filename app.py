"""app.py — the single entrypoint: CLI and web UI around the same engine.

    python app.py run --case case-011      # native tool_use agent, unlabelled evidence
    python app.py serve                    # web UI at http://127.0.0.1:8000
    python app.py cases                    # list cases

The web layer is a thin FastAPI wrapper around the engine (src/). Sessions use memory or Redis; the hidden
truth and evaluation are available only AFTER the investigation finishes (409 before that).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src import access, session_codec, session_store
from src.agent import AgentInvestigator, iter_agent
from src.cases import Case, list_cases, load_case
from src.deciders import ProviderConfigError
from src.evaluator import evaluate
from src.generator import build_and_verify, builder_options, validate_recipe
from src.human_responder import CLIResponder
from src.investigator import LLMInvestigator, RuleBasedInvestigator
from src.loop import GameSession, InvestigationComplete
from src import history
from src.judge import make_llm_conclusion_judge

app = FastAPI(title="LLM Detective")

WEB_DIR = Path(__file__).resolve().parent / "web"


@app.middleware("http")
async def access_gate(request: Request, call_next):
    """Require the shared password when one is configured.

    HTTP Basic, so the browser prompts once and then sends the header on every fetch
    the UI makes — no login page and no session of our own to get wrong. Applied as
    middleware rather than per-route so a route added later is protected by default;
    forgetting to decorate a new endpoint is exactly how these gates leak.

    Unset password -> no gate at all, so local runs and the offline suite are
    untouched.
    """
    if access.gate_enabled() and not access.is_public(request.url.path):
        if not access.credentials_ok(request.headers.get("authorization")):
            return JSONResponse(
                {"detail": "Authentication required."},
                status_code=401,
                # Without this header the browser never prompts and the UI just looks
                # broken. `charset` keeps non-ASCII passwords from being mangled.
                headers={"WWW-Authenticate": 'Basic realm="LLM Detective", '
                                             'charset="UTF-8"'},
            )
    return await call_next(request)

# Sessions are serialized between requests instead of held as live objects, so the
# app works on a serverless platform where the next request may hit a different
# instance. The backend is chosen by environment: a dict locally, Redis on Vercel.
# See src/session_store.py for why, and src/session_codec.py for what survives.
STORE = session_store.build_store()

SESSION_PREFIX = "trace:session:"
CASE_PREFIX = "trace:case:"
CUSTOM_CASE_INDEX = "trace:custom_cases"


def _session_key(sid: str) -> str:
    return f"{SESSION_PREFIX}{sid}"


def _case_key(case_id: str) -> str:
    return f"{CASE_PREFIX}{case_id}"


def _load_case_any(case_id: str) -> Case:
    """A built-in case from disk, or a generated one from the store.

    A generated case body contains the hidden truth, so it lives in the session store
    (engine side) and never in anything sent to the browser.
    """
    stored = STORE.get(_case_key(case_id))
    if stored is not None:
        return Case(stored)
    return load_case(case_id)


def _custom_case_briefings() -> list[dict[str, Any]]:
    """Public listings for generated cases that are still alive.

    Set members carry no TTL of their own, so an id can outlive its case body. Ids
    that no longer resolve are skipped rather than reported as broken cases.
    """
    out: list[dict[str, Any]] = []
    for case_id in STORE.members(CUSTOM_CASE_INDEX):
        stored = STORE.get(_case_key(case_id))
        if stored is None:
            continue
        b = Case(stored).public_briefing()
        out.append({"case_id": case_id, "title": b["title"],
                    "difficulty": b["difficulty"],
                    "short_description": b["short_description"],
                    "budget": b["budget"], "inference": b.get("inference")})
    return out


def _save(sid: str, session: GameSession, mode: str, policy: str,
          case_is_custom: bool) -> None:
    STORE.put(_session_key(sid),
              session_codec.dump_session(session, mode=mode, policy=policy,
                                         case_is_custom=case_is_custom))


class StartRequest(BaseModel):
    case_id: str
    # Empty mode means LLM, including when credentials are missing: no silent fallback.
    mode: str = ""                  # "" (llm default) | rule_based | llm (native) | llm_schema
    policy: str = "demo"            # llm_schema only: demo | evaluation


def _resolve_mode(mode: str) -> str:
    if mode:
        return mode
    return "llm"


def _get(session_id: str) -> dict[str, Any]:
    """Rehydrate a session, or 404.

    A miss and an expiry are indistinguishable here on purpose: both mean "this id
    no longer names an investigation", and the client's response to either is to
    start a new one.
    """
    payload = STORE.get(_session_key(session_id))
    if payload is None:
        raise HTTPException(404, f"No session '{session_id}'.")
    try:
        session, case = session_codec.load_session(payload, _build_session)
    except ValueError as exc:            # unsupported schema after a deploy
        raise HTTPException(409, str(exc))
    except KeyError as exc:              # the case id no longer exists on disk
        raise HTTPException(404, f"Session '{session_id}' references a missing case: {exc}")
    return {"session": session, "case": case,
            "mode": payload["mode"], "policy": payload["policy"],
            "case_is_custom": bool(payload.get("case_data")),
            "sid": session_id}


def _commit(entry: dict[str, Any]) -> None:
    """Persist a session after a turn mutated it."""
    _save(entry["sid"], entry["session"], entry["mode"], entry["policy"],
          entry["case_is_custom"])


def _charge_llm_budget(mode: str) -> None:
    """Cap how many LLM investigations may be started per window.

    A single investigation already has a ceiling: the case budget bounds world actions
    and AGENT_MAX_MODEL_CALLS bounds model calls per turn. What has no ceiling is how
    many investigations a visitor may start, so that is what is counted.

    Only `llm` runs are charged — the rule_based baseline calls no model and costs
    nothing, and rate-limiting the free path would just make the offline demo worse.
    The counter lives in the session store so the limit holds across function
    instances; an in-process counter on a serverless platform would let each instance
    spend the full allowance.
    """
    limit = access.llm_session_limit()
    if mode != "llm" or limit <= 0:
        return
    window = access.llm_window_seconds()
    try:
        used = STORE.incr(access.llm_budget_key(window), window)
    except Exception as exc:
        # Refuse rather than fail open: an unreachable counter must not become an
        # unlimited spend allowance.
        raise HTTPException(503, "Spend limit unavailable; refusing to start an LLM "
                                 "run. Retry shortly.") from exc
    if used > limit:
        raise HTTPException(
            429,
            f"This deployment allows {limit} LLM investigations per "
            f"{window // 60} minutes and the limit is reached. The rule_based "
            f"baseline still works: start a session with mode 'rule_based'.",
        )


def _build_session(case: Case, mode: str, policy: str) -> GameSession:
    if mode == "llm":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(400, "ANTHROPIC_API_KEY is not set. Export it, or use "
                                     "rule_based mode.")
        inv = AgentInvestigator(case)
        inv._conclusion_judge = make_llm_conclusion_judge(inv._client, inv._model)
        session = GameSession(case, investigator=inv, requested_mode="llm")
        inv.session = session
        return session
    if mode == "llm_schema":
        session = GameSession(case, requested_mode="llm")
        try:
            session.investigator = LLMInvestigator(
                session.toolbox, briefing=case.public_briefing(), mode=policy)
        except ProviderConfigError as exc:
            raise HTTPException(400, str(exc))
        return session
    session = GameSession(case, requested_mode="rule_based")
    session.investigator = RuleBasedInvestigator(session.toolbox)
    return session


def _view(entry: dict[str, Any], latest=None, trace=None) -> dict[str, Any]:
    session: GameSession = entry["session"]
    out = {
        "events": list(session.events),
        "trace": [a.model_dump() for a in session.audit],
        "agent_proposal": getattr(session.investigator, "final_proposal", None),
        "state": session.state.model_dump(),
        "run_metadata": session.run_metadata(),
        # Native agent: get_skill + judge/revise events (happen inside decide(),
        # so not in the audit) — the UI shows them so you can see WHY a conclusion was rejected.
        "agent_events": [e for e in session.events
                         if e.get("type") == "gate" or e.get("name") == "get_skill"][-30:],
        # Set when the run is parked on a question only a person can answer. The UI
        # renders the role card and posts the reply to /answer.
        "awaiting_human": session.awaiting_human,
        "pending_question": session.pending_question(),
    }
    if latest is not None:
        out["latest"] = latest.model_dump()
    if trace is not None:
        out["trace"] = [a.model_dump() for a in trace]
    return out


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/board")
def board():
    return FileResponse(WEB_DIR / "board.html")


@app.get("/api/config")
def config():
    """Lets the UI preselect the right investigator and label the baseline honestly.

    No secret is exposed — only whether a key is present in the server environment,
    and which session backend is live. The latter is the one-call check that a
    serverless deployment is wired correctly: `session_store: "memory"` together with
    `serverless: true` means investigations will be dropped between turns.
    """
    return {
        "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "session_store": STORE.name,
        "serverless": bool(os.environ.get("VERCEL")),
        # Reachable without credentials on purpose: these say whether the protections
        # are ON, never what they are. A deployment that is publicly spendable should
        # be visible in one curl instead of discovered on the invoice.
        "access_gate": access.gate_enabled(),
        "llm_sessions_per_window": access.llm_session_limit(),
    }


@app.get("/api/cases")
def cases():
    return {"cases": list_cases() + _custom_case_briefings()}


@app.get("/api/builder_options")
def get_builder_options():
    return builder_options()


@app.post("/api/custom_case")
def custom_case(recipe: dict):
    """Recipe -> a generated, verified case. Returns ONLY the public briefing."""
    try:
        case = build_and_verify(recipe)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    # The body carries the hidden truth, so it goes to the session store (engine
    # side) and never to the browser, which receives only the public briefing.
    STORE.put(_case_key(case.case_id), case._data)
    STORE.add_to_index(CUSTOM_CASE_INDEX, case.case_id)
    return {"case_id": case.case_id, "briefing": case.public_briefing()}


@app.post("/api/describe_case")
def describe_case(body: dict):
    """A free-form scenario description -> a recipe (one LLM call), then the same generator.
    The LLM only FILLS IN the recipe fields — the world and truth are still composed by the server,
    so bad LLM output cannot create an inconsistent case."""
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Give a short description of the scenario.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(400, "ANTHROPIC_API_KEY is not set — use the builder "
                                 "dropdowns instead.")
    import re as _re
    from anthropic import Anthropic
    opts = builder_options()
    prompt = (
        "Map this mystery-scenario description to a JSON recipe. Return ONLY JSON:\n"
        '{"incident_type": one of ' + str(opts["incident_types"]) + ", "
        '"location": short_snake_case_id, "suspects": [3-5 first names], '
        '"culprit": one of the suspects or "random", '
        '"twist": one of ' + str(opts["twists"]) + ", "
        '"red_herring": true|false, "budget": 6-14, "title": "short title"}\n\n'
        "Description:\n" + text)
    resp = Anthropic().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=400, messages=[{"role": "user", "content": prompt}])
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if not m:
        raise HTTPException(400, "Could not derive a recipe from that description.")
    try:
        recipe = validate_recipe(json.loads(m.group(0)))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"Derived recipe was invalid: {exc}")
    return {"recipe": recipe}


@app.post("/api/sessions")
def start(req: StartRequest):
    try:
        case = _load_case_any(req.case_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    mode = _resolve_mode(req.mode)
    # Built first, charged second: _build_session makes no model call, but it does
    # reject an llm run when no key is configured. Charging before it would let a
    # request that never started an investigation consume an allowance slot.
    session = _build_session(case, mode, req.policy)
    _charge_llm_budget(mode)
    sid = uuid.uuid4().hex
    case_is_custom = STORE.get(_case_key(req.case_id)) is not None
    entry = {"session": session, "case": case, "mode": mode,
             "policy": req.policy, "case_is_custom": case_is_custom, "sid": sid}
    _commit(entry)
    out = _view(entry)
    out["session_id"] = sid
    out["briefing"] = case.public_briefing()
    return out


@app.post("/api/sessions/{sid}/step")
def step(sid: str):
    entry = _get(sid)
    try:
        latest = entry["session"].step()
    except InvestigationComplete as exc:
        raise HTTPException(409, str(exc))
    # One turn per request is the serverless-friendly shape the UI already uses
    # (it drives the loop client-side), and it is why each turn must be saved here.
    _commit(entry)
    return _view(entry, latest=latest)


@app.post("/api/sessions/{sid}/answer")
def answer(sid: str, body: dict):
    """Resume a run parked on a human-played character's answer.

    The text is authored by the person playing that character. It is the only
    route by which an answer enters the run — the agent has no way to supply one.
    """
    entry = _get(sid)
    session: GameSession = entry["session"]
    if not session.awaiting_human:
        raise HTTPException(409, "This session is not awaiting a human answer.")
    text = (body or {}).get("answer", "")
    if not str(text).strip():
        raise HTTPException(400, "An answer is required.")
    try:
        latest = session.provide_human_answer(str(text))
    except (InvestigationComplete, ValueError, TypeError) as exc:
        raise HTTPException(409, str(exc))
    # The person's answer is buffered and consumed inside provide_human_answer, in
    # this one request, so the responder needs no persistence — but the state the
    # replayed action produced does.
    _commit(entry)
    return _view(entry, latest=latest)


@app.post("/api/sessions/{sid}/run")
def run(sid: str):
    """Run to completion in a single request.

    Kept for programmatic clients. Both web views drive /step instead. A step may
    itself contain several inference/review calls, so either endpoint can hit a
    hosting platform's request-duration ceiling. See docs/DEPLOYMENT.md.
    """
    entry = _get(sid)
    entry["session"].run()
    _commit(entry)
    # run() returns early when it hits a human question; the response carries
    # awaiting_human so the client knows to prompt rather than assume completion.
    return _view(entry, trace=entry["session"].audit)


@app.post("/api/sessions/{sid}/reset")
def reset(sid: str):
    entry = _get(sid)
    entry["session"] = _build_session(entry["case"], entry["mode"], entry["policy"])
    _commit(entry)
    return _view(entry)


@app.get("/api/sessions/{sid}/trace")
def trace(sid: str):
    entry = _get(sid)
    return {"trace": [a.model_dump() for a in entry["session"].audit],
            "events": list(entry["session"].events)}


@app.get("/api/sessions/{sid}/evaluation")
def evaluation(sid: str):
    entry = _get(sid)
    session: GameSession = entry["session"]
    if session.state.status in ("investigating", "awaiting_human"):
        raise HTTPException(409, "Investigation still running; the hidden truth stays "
                                 "hidden until it finishes.")
    ev = evaluate(entry["case"], session.state, session.run_metadata())
    return {
        "evaluation": ev,
        "hidden_truth": entry["case"].hidden_truth,
        "report": session.state.final_report.model_dump()
        if session.state.final_report else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class C:
    BOLD = "\033[1m"; RESET = "\033[0m"; BLUE = "\033[34m"; GREEN = "\033[32m"
    YELLOW = "\033[33m"; CYAN = "\033[36m"; MAGENTA = "\033[35m"; RED = "\033[31m"


def run_cli(case_id: str):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{C.RED}Error: ANTHROPIC_API_KEY is not set.{C.RESET}")
        sys.exit(1)
    cap: dict[str, Any] = {}
    t0 = time.time()
    last_final: dict[str, Any] | None = None
    # At a terminal a person can simply be asked, so the CLI blocks on stdin rather
    # than parking the run. Role card prints once per character, then each question.
    for ev in iter_agent(case_id, _capture=cap, responder=CLIResponder()):
        t = ev["type"]
        if t == "final":
            last_final = ev
        if t == "input":
            print(f"{C.BOLD}INPUT: {json.dumps(ev['data']['briefing'], indent=2)[:400]}…{C.RESET}")
        elif t == "think":
            print(f"{C.BLUE}🧠 THINK:{C.RESET} {ev['text']}")
        elif t == "tool" and ev.get("name") == "get_skill":
            got = ev["result"].get("name") or ev["result"].get("error")
            print(f"{C.CYAN}📖 SKILL:{C.RESET} get_skill({ev['input'].get('name')}) → {got}")
        elif t == "tool":
            print(f"{C.YELLOW}🔧 ACT:{C.RESET} {ev['name']}({json.dumps(ev['input'])}) "
                  f"[skill={ev.get('skill')}]")
            obs = ev["result"]["observation"]
            print(f"{C.GREEN}OBSERVE {'✓' if ev['ok'] else '✗'}:{C.RESET} {obs[:300]}")
            for ch in ev["result"]["state_changes"]:
                print(f"   • {ch}")
        elif t == "gate":
            if ev["phase"] == "judge":
                tag = f"{C.GREEN}accepted" if ev["passed"] else f"{C.RED}rejected"
                print(f"{C.CYAN}🛡️  JUDGE:{C.RESET} conclusion "
                      f"{json.dumps(ev['proposal'].get('verdict'))} → {tag}{C.RESET}")
                for i in ev["issues"]:
                    print(f"   {C.RED}- {i}{C.RESET}")
            elif ev["phase"] == "revise":
                print(f"{C.YELLOW}↻ REVISE: critique returned to the agent…{C.RESET}")
        elif t == "final":
            print(f"\n{C.MAGENTA}{'─' * 70}\nEND ({ev['reason']}): {ev['text']}{C.RESET}")
            u = ev["usage"]
            print(f"{C.CYAN}📊 USAGE:{C.RESET} {u['llm_calls']} LLM calls, "
                  f"{u['input_tokens']} in / {u['output_tokens']} out tokens")
            md = ev["structured"]["run_metadata"]
            print(f"{C.CYAN}🔎 PROVENANCE:{C.RESET} fully_llm_driven={md['fully_llm_driven']} "
                  f"llm={md['llm_decisions']} fallback={md['fallback_decisions']}")
    # Out-of-band evaluation against hidden truth:
    session = cap["session"]
    ev = evaluate(cap["case"], session.state, session.run_metadata())
    print(f"{C.CYAN}🏁 EVALUATOR:{C.RESET} score={ev['total_score']} "
          f"status_match={ev['status_match']} actions={ev['actions_used']}/{ev['budget']}")

    # Persistent run record (history) — for later analysis, eval and training.
    usage = (last_final or {}).get("usage") or {}
    proposal = ((last_final or {}).get("structured") or {}).get("agent_proposal")
    # record_from_trace returns the RECORD (a dict), not a path — it has already
    # written the file by this point. Locate it by run_id rather than rebuilding the
    # filename here, so the naming scheme stays owned by history.save_record.
    rec = history.record_from_trace(cap["case"], session, ev, usage=usage,
                                    seconds=time.time() - t0, agent_proposal=proposal)
    base = Path(__file__).resolve().parent
    hits = sorted((base / "outputs" / "runs").glob(f"*_{rec['run_id']}.json"))
    where = hits[-1].relative_to(base) if hits else f"outputs/runs/ (run_id {rec['run_id']})"
    print(f"{C.CYAN}💾 HISTORY:{C.RESET} {where}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "cases":
        for c in list_cases():
            print(f"{c['case_id']}: {c['title']} ({c['difficulty']}, budget {c['budget']})")
    elif cmd == "run":
        case_id = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else "case-011"
        run_cli(case_id)
    elif cmd == "serve":
        import uvicorn
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
    else:
        print("Usage: python app.py [run --case case-011 | serve | cases]")
        sys.exit(1)


if __name__ == "__main__":
    main()
