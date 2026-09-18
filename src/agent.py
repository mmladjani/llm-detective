"""Native tool_use agent — the default LLM driver (src/agent.py).

The model receives the Anthropic tool
schemas, picks ONE tool per turn as a `tool_use` block, Python executes it through
the engine (`GameSession.execute_decision` — legality, budget, state, audit), and
the result goes back as `tool_result`. The model also:

  * loads skills itself (`get_skill` — a free action, visible in the trace), and
  * concludes via `submit_conclusion`, which an independent judge (src/judge.py)
    checks against the evidence on record; an unsupported conclusion comes back as
    a critique and the agent revises (bounded by MAX_REVISES).

`iter_agent(case_id)` is a generator yielding the shared event vocabulary
(input / think / tool / gate / final) used by the CLI and web views.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from . import affordances
from . import judge as judge_mod
from . import skills as skills_mod
from .cases import Case, load_case
from .investigator import Decision, Investigator, NextAction
from .loop import GameSession

BASE = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
MAX_REVISES = int(os.environ.get("AGENT_MAX_REVISES", "2"))
MAX_MODEL_CALLS = int(os.environ.get("AGENT_MAX_MODEL_CALLS", "40"))

# Extended thinking is enabled by default and surfaced as `think` events.
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4000"))
THINKING_BUDGET = int(os.environ.get("AGENT_THINKING_BUDGET", "2000"))


# The transcript is append-only; prompt caching can reuse its stable prefix.
# AGENT_CACHE_TTL optionally sets the provider cache lifetime. Caching does not
# compact the conversation, and investigator counters exclude cache token fields.
CACHE_TTL = os.environ.get("AGENT_CACHE_TTL", "").strip()


def cache_config() -> dict[str, Any]:
    """The `cache_control` parameter. Set AGENT_CACHE_TTL=1h for a deployed UI."""
    config: dict[str, Any] = {"type": "ephemeral"}
    if CACHE_TTL:
        config["ttl"] = CACHE_TTL
    return config


def thinking_config(max_tokens: int = MAX_TOKENS,
                    budget: int = THINKING_BUDGET) -> dict[str, Any] | None:
    """The `thinking` parameter, or None when it cannot be satisfied.

    The API requires max_tokens to exceed the thinking budget (the budget is drawn
    from the same allowance), so a misconfigured pair must disable thinking rather
    than fail the run. Set AGENT_THINKING_BUDGET=0 to turn it off.
    """
    if budget <= 0 or max_tokens <= budget:
        return None
    return {"type": "enabled", "budget_tokens": budget}

WORLD_TOOLS: dict[str, dict[str, Any]] = {
    "read_incident_report": {"desc": "Read the public incident report: scope, suspects, "
                                     "locations, objects and the time window.", "args": {}},
    "interview_character": {"desc": "Interview a suspect and record their account.",
                            "args": {"suspect": "string"}},
    "interview_human": {"desc": "Ask a human-played character one question. A real person "
                                "answers — you supply only the question, never the answer. "
                                "Use 'topic' to group questions about the same subject so "
                                "shifting stories become visible.",
                         "args": {"suspect": "string", "question": "string",
                                  "topic": "string"}},
    "inspect_location": {"desc": "Inspect a location for physical traces.",
                         "args": {"location": "string"}},
    "analyze_object": {"desc": "Analyze an object for how it was used.",
                       "args": {"object": "string"}},
    "check_access_log": {"desc": "Check who accessed a location during the window.",
                         "args": {"location": "string"}},
    "review_camera": {"desc": "Review camera footage covering a location.",
                      "args": {"location": "string"}},
    "verify_alibi": {"desc": "Test a suspect's stated alibi against the records.",
                     "args": {"suspect": "string"}},
    "compare_testimonies": {"desc": "Compare two suspects' testimonies for conflicts.",
                            "args": {"a": "string", "b": "string"}},
}


def build_schemas() -> list[dict[str, Any]]:
    schemas = []
    for name, spec in WORLD_TOOLS.items():
        props = {k: {"type": v} for k, v in spec["args"].items()}
        schemas.append({
            "name": name,
            "description": spec["desc"],
            "input_schema": {"type": "object", "properties": props,
                             "required": list(props)},
        })
    schemas.append({
        "name": "get_skill",
        "description": "Load an investigation skill (a short playbook). Free action. "
                       "Available: " + ", ".join(sorted(skills_mod.SKILLS)) + ".",
        "input_schema": {"type": "object",
                         "properties": {"name": {"type": "string"}},
                         "required": ["name"]},
    })
    schemas.append({
        "name": "assess_evidence",
        "description": "Record YOUR reading of one piece of evidence you have "
                       "collected: who it implicates and how strongly. Free action, "
                       "costs no budget. Only available on cases where evidence "
                       "arrives unlabelled (briefing field \"inference\": \"model\") — "
                       "there the engine assigns no weights, so a hypothesis only "
                       "moves if you weigh the evidence yourself. You may re-assess "
                       "an item when later facts change what it means.",
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_id": {"type": "string",
                                "description": "Id of an evidence item a tool has "
                                               "already returned (e.g. 'al_mara_entry')."},
                "supports": {"type": "string",
                             "description": "The suspect id this evidence bears on, "
                                            "lowercase (e.g. 'mara')."},
                "weight": {"type": "number",
                           "description": "Positive = incriminating, negative = "
                                          "exculpatory, 0 = collected but not probative "
                                          "(a red herring). Magnitude at most 0.35: no "
                                          "single record is decisive on its own."},
                "rationale": {"type": "string",
                              "description": "Why this evidence carries that weight."},
            },
            "required": ["evidence_id", "supports", "weight", "rationale"],
        },
    })
    schemas.append({
        "name": "establish_fact",
        "description": "Infer or revise action, method, motive or time from a collected record. "
                       "Free, model mode only. Choose from fact_candidates in the briefing; "
                       "those options include decoys. The judge reviews your rationale.",
        "input_schema": {"type": "object", "properties": {
            "field": {"type": "string", "enum": ["action", "method", "motive", "time"]},
            "value": {"type": "string"}, "evidence_id": {"type": "string"},
            "rationale": {"type": "string"}},
            "required": ["field", "value", "evidence_id", "rationale"]},
    })
    schemas.append({
        "name": "submit_conclusion",
        "description": "Submit your final conclusion. An independent judge checks it "
                       "against the evidence on record; unsupported conclusions are "
                       "returned with a critique for revision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["solved", "unresolved"]},
                "culprit": {"type": ["string", "null"],
                            "description": "The suspect id in lowercase (e.g. 'mara'), "
                                           "or null for unresolved."},
                "reasoning": {"type": "string"},
                "summary": {"type": "string",
                            "description": "Your account of the case in your own "
                                           "words — what happened, who did it, and "
                                           "which evidence establishes it. This is "
                                           "carried into the final report and scored "
                                           "for groundedness, so cite only ids and "
                                           "facts your tools actually returned."},
            },
            "required": ["verdict", "reasoning"],
        },
    })
    return schemas


def _critique_body(verdict: dict[str, Any]) -> str:
    """The judge's objection as the agent reads it.

    One builder for both rejection paths (revise-budget-remaining and
    budget-remaining), which previously carried two copies of this formatting and so
    drifted apart whenever either gained a section.
    """
    parts = ["Conclusion NOT accepted. Issues:\n- " + "\n- ".join(verdict["issues"])]
    parts.append("\n\nRepair the investigation state, not just the final prose. "
                 "If a stored evidence rationale overclaims, call assess_evidence again "
                 "to replace it; if an inferred fact is unsupported, revise it with "
                 "establish_fact or gather support. Explain disagreements with the "
                 "review using exact collected observations, never invented evidence. "
                 "A lie alone does not prove guilt; a credential alone does not identify its user.")
    if verdict.get("crossexam_questions"):
        parts.append("\n\nCross-examination questions to address:\n- "
                     + "\n- ".join(verdict["crossexam_questions"]))
    if verdict.get("advisories"):
        # Advisory findings did NOT block this conclusion, and saying so matters: an
        # agent told only "rejected" learns to soften prose until objections stop,
        # which is not the behaviour this loop is meant to reward.
        parts.append("\n\nAlso worth strengthening — these did NOT block the "
                     "conclusion, so do not rewrite the reasoning just to satisfy "
                     "them:\n- " + "\n- ".join(verdict["advisories"]))
    cm = verdict.get("confidence_mismatch")
    if cm and cm.get("suspicious"):
        parts.append(f"\n\nConfidence note: stated {cm['stated']*100:.0f}%, "
                     f"bookkeeping score {cm['evidence']*100:.0f}%. "
                     "The score is heuristic, not a calibrated probability or proof.")
    return "".join(parts)


_ID_ARGS = ("suspect", "location", "object", "a", "b", "culprit",
            "supports", "evidence_id")


def _canon_args(args: dict) -> dict:
    """Canonicalize arguments to id form ('Mara' -> 'mara', 'Archive Room' ->
    'archive_room'). The model naturally writes display names from the briefing while
    the tools and the judge work with ids; this is input normalization, not help with
    solving the case."""
    import re as _re
    out = dict(args)
    for k in _ID_ARGS:
        if isinstance(out.get(k), str):
            out[k] = _re.sub(r"[^a-z0-9]+", "_", out[k].strip().lower()).strip("_")
    return out


def load_system_prompt() -> str:
    return (BASE / "system_prompt.md").read_text(encoding="utf-8")


class AgentInvestigator(Investigator):
    """Anthropic tool_use conversation adapted to the engine's decide() interface.

    One `decide()` call = advance the conversation until the model proposes the next
    WORLD action (get_skill loops internally as a free action; submit_conclusion goes
    through the judge and may loop as a revise). Provenance is honest: every returned
    Decision is source="llm".
    """

    name = "llm"

    # This driver owns `submit_conclusion`, so the engine must not declare the case
    # solved on its behalf — see GameSession.defer_solved_to_agent.
    concludes_explicitly = True

    def __init__(self, case: Case, client=None, model: str = MODEL,
                 max_revises: int = MAX_REVISES, conclusion_judge=None,
                 max_tokens: int = MAX_TOKENS,
                 thinking: dict[str, Any] | None = None):
        self.skills_used: list[str] = []  # Track skills loaded during this investigation
        if client is None:
            from anthropic import Anthropic
            client = Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        # Explicit None means "caller did not choose"; fall back to the env-driven
        # default. An explicit {} / disabled config from a caller is respected.
        self._thinking = thinking if thinking is not None else thinking_config(max_tokens)
        # Layer-2 LLM reviewer, applied only after the deterministic gate passes.
        # None -> deterministic-only (default; keeps offline tests hermetic). The real
        # run wires one in via iter_agent; a test can inject a fake to exercise it.
        self._conclusion_judge = conclusion_judge
        self.provider = "anthropic"
        self.session: Optional[GameSession] = None   # bound by driver
        self._schemas = build_schemas()
        self._system = load_system_prompt()
        self._messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": ("PUBLIC BRIEFING (no solution inside):\n"
                        + json.dumps(case.public_briefing(), indent=2)
                        + "\n\nLead the investigation. One tool per turn."),
        }]
        self._pending_id: Optional[str] = None
        self._pending_extras: list[str] = []
        self._extra_ids: list[str] = []
        self._nudged = False
        self.active_skill: Optional[str] = None
        self._max_revises = max_revises
        self.revises_left = max_revises
        self.usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
        # Drained by the event driver (iter_agent):
        self.think_log: list[str] = []
        self.event_log: list[dict[str, Any]] = []    # skill loads + judge verdicts
        # Monotonic counter stamped on every think text and event as it is produced, so
        # the two lists can be merged back into the order they actually happened in.
        # Without it a reader cannot tell whether a thought preceded or followed the
        # judge verdict it refers to.
        self._seq = 0
        self._think_seq: list[int] = []
        self._event_seq: list[int] = []
        self.final_proposal: Optional[dict[str, Any]] = None

    # ---- helpers --------------------------------------------------------------
    def _track(self, resp):
        u = getattr(resp, "usage", None)
        self.usage["llm_calls"] += 1
        if u is not None:
            self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0

    def _last_result_payload(self) -> str:
        s = self.session
        entry = s.audit[-1] if s.audit else None
        payload = {
            "ok": entry.tool_ok if entry else False,
            "observation": entry.observation if entry else "",
            "state_changes": entry.state_changes if entry else [],
            "hypotheses": entry.hypotheses_snapshot if entry else [],
            "remaining_budget": s.state.remaining_budget,
            "catalog": s.state.index,
            "facts": s.state.facts_revealed,
            "fact_assertions": s.state.fact_assertions,
            "open_questions": [q.model_dump() for q in s.state.open_questions],
        }
        # In model-inference mode nothing moves the hypotheses until the agent weighs
        # the evidence, so the outstanding ids travel with every result. Without this
        # the agent would have to remember ids across a long transcript, and forgetting
        # one looks identical to deciding it was irrelevant.
        unassessed = s.unassessed_evidence()
        if unassessed:
            payload["unassessed_evidence"] = unassessed
            payload["note"] = (
                "This case ships no authored weights: these items are collected but "
                "carry no attribution until you call assess_evidence on them."
            )
        # The evidence now clears the bar. The engine used to act on this itself and
        # end the run; now it only reports it, so the agent still has to decide
        # whether the record really supports a verdict — and to say what that verdict
        # is. Deliberately phrased as a state of the evidence, not an instruction to
        # conclude: "unresolved" stays a legitimate answer here.
        if s.state.conclusion_ready:
            payload["conclusion_ready"] = True
            payload["note_conclusion"] = (
                "The evidence on record now meets the bar for a conclusion "
                "(confidence, independent lines, lead margin, no open questions). "
                "This is a heuristic signal, not proof of correctness: call "
                "submit_conclusion when you can justify a verdict, or keep "
                "investigating if you think the record is misleading."
            )
        # Budget is spent: the model gets one free move to conclude. Without this
        # signal it does not know this is its last chance and wastes it on a tool.
        if s.state.remaining_budget <= 0:
            payload["final_turn"] = True
            payload["instruction"] = (
                "Budget exhausted. This is your final turn and it is free: call "
                "submit_conclusion now with the verdict the evidence supports. "
                "World actions are no longer available."
            )
        return json.dumps(payload)

    def _tool_result(self, tool_use_id: str, content: str) -> dict[str, Any]:
        blocks = [{"type": "tool_result", "tool_use_id": tool_use_id,
                   "content": content}]
        # The Anthropic API requires a tool_result for every tool_use from the previous reply;
        # we reject the extra (batched) moves with an explanation.
        for tid in getattr(self, "_extra_ids", []) or []:
            blocks.append({"type": "tool_result", "tool_use_id": tid,
                           "is_error": True,
                           "content": "One tool call per turn; this extra call was "
                                      "ignored."})
        self._extra_ids = []
        return {"role": "user", "content": blocks}

    # ---- ordered logging ------------------------------------------------------
    def _log_event(self, ev: dict[str, Any]) -> None:
        """Record a skill load or judge verdict, stamped with its place in time."""
        self._seq += 1
        self.event_log.append(ev)
        self._event_seq.append(self._seq)
        if self.session is not None:
            self.session.record_event(ev)

    def drain_think(self) -> list[tuple[int, str]]:
        """Return [(seq, text)] and clear. Pairs each thought with when it happened."""
        out = list(zip(self._think_seq, self.think_log))
        self.think_log.clear()
        self._think_seq.clear()
        return out

    def drain_events(self) -> list[tuple[int, dict[str, Any]]]:
        """Return [(seq, event)] and clear."""
        out = list(zip(self._event_seq, self.event_log))
        self.event_log.clear()
        self._event_seq.clear()
        return out

    # ---- the decision ---------------------------------------------------------
    def decide(self, state) -> Decision:
        assert self.session is not None, "AgentInvestigator: bind .session first"

        # Feed the previous world action's result back to the model.
        if self._pending_id is not None:
            # New evidence restores the right to submit a conclusion: the revise budget exists to
            # prevent resubmitting the SAME conclusion without new information, not to
            # punish an agent that went and found something in the meantime.
            last = self.session.audit[-1] if self.session.audit else None
            if last is not None and last.tool_ok and last.state_changes:
                self.revises_left = self._max_revises
            self._extra_ids = getattr(self, "_pending_extras", [])
            self._messages.append(self._tool_result(self._pending_id,
                                                    self._last_result_payload()))
            self._pending_id = None
            self._pending_extras = []

        for _ in range(MAX_MODEL_CALLS):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self._model, max_tokens=self._max_tokens,
                    system=self._system, tools=self._schemas,
                    messages=self._messages,
                    cache_control=cache_config(),
                )
                if self._thinking:
                    kwargs["thinking"] = self._thinking
                resp = self._client.messages.create(**kwargs)
            except Exception as exc:  # an API error must not crash the session
                return Decision(action=None, source="llm", llm_called=True,
                                validation_status="failed",
                                failure_reason=f"api_exception: {type(exc).__name__}: {exc}",
                                execution_error=True)
            self._track(resp)

            # Extended-thinking blocks are the agent's actual deliberation, so they
            # belong in the trace alongside its plain-text remarks. They are kept out
            # of `texts` because that feeds NextAction.question, which records the
            # question the agent asked itself — a whole reasoning dump is not that.
            for b in resp.content:
                if getattr(b, "type", "") == "thinking":
                    reasoning = (getattr(b, "thinking", "") or "").strip()
                    if reasoning:
                        self._seq += 1
                        self.think_log.append(reasoning)
                        self._think_seq.append(self._seq)
                        self.session.record_event({"type": "think", "text": reasoning})
            texts = [b.text.strip() for b in resp.content
                     if getattr(b, "type", "") == "text" and b.text.strip()]
            for t in texts:
                self._seq += 1
                self.think_log.append(t)
                self._think_seq.append(self._seq)
                self.session.record_event({"type": "think", "text": t})
            tool_uses = [b for b in resp.content
                         if getattr(b, "type", "") == "tool_use"]
            tool_use = tool_uses[0] if tool_uses else None
            # the API requires a tool_result for EVERY tool_use — we reject the extras explicitly.
            self._extra_ids = [b.id for b in tool_uses[1:]]

            if tool_use is None:
                if not self._nudged:
                    self._nudged = True
                    self._messages.append({"role": "assistant", "content": resp.content})
                    self._messages.append({"role": "user", "content":
                                           "Do not stop in plain text. Either call a tool "
                                           "or call submit_conclusion."})
                    continue
                return Decision(action=None, source="llm", llm_called=True,
                                validation_status="ok", should_stop=True,
                                stop_reason="model stopped without submit_conclusion")

            self._messages.append({"role": "assistant", "content": resp.content})
            name, args = tool_use.name, _canon_args(dict(tool_use.input or {}))

            if name == "get_skill":
                result = skills_mod.read_skill(args.get("name", ""))
                if result.get("ok"):
                    self.active_skill = result["name"]
                    if result["name"] not in self.skills_used:
                        self.skills_used.append(result["name"])
                self._log_event({"type": "tool", "name": "get_skill",
                                       "input": args, "result": result,
                                       "ok": bool(result.get("ok"))})
                self._messages.append(self._tool_result(tool_use.id, json.dumps(result)))
                continue  # free action — model immediately picks the next move

            if name in ("assess_evidence", "establish_fact"):
                # The agent's own inference. Free like get_skill, but unlike get_skill
                # it MUTATES state, so the engine applies it and it lands in the audit
                # — the trace has to show whose reading moved the hypotheses.
                apply = (self.session.apply_assessment if name == "assess_evidence"
                         else self.session.establish_fact)
                entry = apply(
                    args,
                    Decision(action=None, source="llm", llm_called=True,
                             validation_status="ok"),
                )
                # NOT _log_event'd: apply_assessment already appended an AuditEntry,
                # and the event driver emits every audited action. Logging it here as
                # well would show the assessment twice in the trace.
                self._messages.append(self._tool_result(tool_use.id, self._last_result_payload()))
                continue  # free action — model immediately picks the next move

            if name == "submit_conclusion":
                args["verdict"] = str(args.get("verdict") or "").strip().lower()
                # Layer 1: deterministic gate (always). Layer 2: LLM reviewer, only if
                # layer 1 passed and a reviewer is wired in (real runs). Either layer's
                # objection becomes a critique the agent must address.
                # Negative means the final permitted retry was already rejected.
                # This counter is persisted and renewed only by new world-state
                # progress above: repeated submissions must not lottery the reviewer.
                det = ({"ok": False, "issues": [
                    "Review retry limit reached. Gather new evidence before submitting again."]}
                    if self.revises_left < 0 else
                    judge_mod.check_conclusion(self.session.state, args))
                issues = list(det["issues"])
                crossexam_questions = list(det.get("crossexam_questions", []))
                confidence_mismatch = det.get("confidence_mismatch")
                llm_ok = None
                llm_error = None
                finding_audit = []
                advisories: list[str] = []
                if det["ok"] and self._conclusion_judge is not None:
                    lj = self._conclusion_judge(self.session.state, args)
                    llm_ok = bool(lj.get("ok", True))
                    llm_error = lj.get("error")
                    finding_audit = lj.get("finding_audit", [])
                    advisories = [str(a) for a in (lj.get("advisories") or [])]
                    if not llm_ok:
                        issues.extend(f"[reviewer] {i}" for i in
                                      (lj.get("issues") or ["Review did not approve the proposal."]))
                verdict = {"ok": not issues, "issues": issues,
                           "advisories": advisories,
                           "crossexam_questions": crossexam_questions,
                           "confidence_mismatch": confidence_mismatch,
                           "lead": det.get("lead"), "review_error": llm_error,
                           "finding_audit": finding_audit}
                self._log_event({"type": "gate", "phase": "judge",
                                       "proposal": args, "passed": verdict["ok"],
                                       "issues": verdict["issues"],
                                       "advisories": advisories,
                                       "confidence_mismatch": verdict.get("confidence_mismatch"),
                                       "deterministic_ok": det["ok"], "llm_ok": llm_ok,
                                       "review_error": llm_error, "finding_audit": finding_audit})
                if not verdict["ok"] and self.revises_left > 0:
                    self.revises_left -= 1
                    self._log_event({"type": "gate", "phase": "revise",
                                           "issues": verdict["issues"],
                                           "crossexam_questions": verdict.get("crossexam_questions", [])})
                    critique = json.dumps({
                        "ok": False,
                        "judge": _critique_body(verdict) +
                                 "\n\nGather the missing evidence or answer these questions, "
                                 "then submit again."})
                    self._messages.append(self._tool_result(tool_use.id, critique))
                    continue  # revise — the loop goes on

                # Revise budget spent BUT the investigation budget is not: a rejected conclusion
                # must not end an investigation that still has moves available.
                # Send the agent back into the world — new evidence restores its right to submit.
                if not verdict["ok"] and self.session.state.remaining_budget > 0:
                    self.revises_left = -1
                    legal_now = [f"{la['tool']}({la['key']})" for la
                                 in affordances.legal_actions(self.session.state)][:12]
                    self._log_event({"type": "gate", "phase": "revise",
                                           "issues": verdict["issues"],
                                           "crossexam_questions": verdict.get("crossexam_questions", [])})
                    self._messages.append(self._tool_result(tool_use.id, json.dumps({
                        "ok": False,
                        "judge": _critique_body(verdict) +
                                 "\n\nSTOP RESUBMITTING. Resubmitting the same "
                                 "conclusion without new evidence will not change the "
                                 "verdict. You still have "
                                 + str(self.session.state.remaining_budget)
                                 + " action(s) left — take one of these now: "
                                 + (", ".join(legal_now) or "(none)")
                                 + ". Gathering new evidence restores your right to "
                                   "submit."})))
                    continue
                # Accepted (or revises exhausted -> recorded honestly).
                self.final_proposal = {"proposal": args, "judge": verdict,
                                       "forced": not verdict["ok"]}
                # Hand the agent's narrative to the engine BEFORE it synthesizes, so
                # the report carries the agent's own words instead of discarding them.
                self.session.record_agent_conclusion(args, verdict)
                return Decision(action=None, source="llm", llm_called=True,
                                validation_status="ok", should_stop=True,
                                stop_reason=f"submit_conclusion: {args.get('verdict')}"
                                            + (" (judge objections stand)"
                                               if not verdict["ok"] else ""))

            if name not in WORLD_TOOLS:
                self._messages.append(self._tool_result(
                    tool_use.id, json.dumps({"ok": False,
                                             "error": f"unknown tool {name!r}"})))
                continue

            # A world action: hand it to the engine; result returns next decide().
            self._pending_id = tool_use.id
            self._pending_extras = list(self._extra_ids)
            self._extra_ids = []
            question = texts[-1] if texts else ""
            return Decision(
                action=NextAction(self.active_skill or "unspecified",
                                  question, name, args),
                source="llm", llm_called=True, validation_status="ok",
            )

        return Decision(action=None, source="llm", llm_called=True,
                        validation_status="failed",
                        failure_reason="model call budget exhausted in decide()",
                        should_stop=True, stop_reason="model_call_budget")


# ---------------------------------------------------------------------------
# Event driver: the shared event vocabulary (input/think/tool/gate/final)
# ---------------------------------------------------------------------------
def iter_agent(case_id: str, client=None, model: str = MODEL,
               _capture: Optional[dict] = None, responder=None,
               max_tokens: int = MAX_TOKENS,
               thinking: Optional[dict[str, Any]] = None):
    case = load_case(case_id)
    inv = AgentInvestigator(case, client=client, model=model,
                            max_tokens=max_tokens, thinking=thinking)
    # Real run (no injected client): layer the LLM reviewer over the deterministic gate,
    # reusing the agent's own client. When a client is injected (tests) the reviewer is
    # left off so scripted runs stay hermetic; a test wires its own fake if it wants it.
    if client is None:
        inv._conclusion_judge = judge_mod.make_llm_conclusion_judge(inv._client, model)
    session = GameSession(case, investigator=inv, requested_mode="llm",
                          responder=responder)
    inv.session = session
    if _capture is not None:  # eval needs the finished session for the evaluator
        _capture["case"] = case
        _capture["session"] = session

    yield {"type": "input", "data": {"case_id": case_id, "mode": "llm",
                                     "briefing": case.public_briefing()}}

    step = 0
    emitted = 0
    while session.state.status == "investigating":
        session.step()
        step += 1
        # Compatibility buffers are mirrored into the session's persistent ordered
        # stream at occurrence time, including events inside a single decide().
        inv.drain_think()
        inv.drain_events()

        # Emit thoughts, free inferences, world actions and judge feedback in causal
        # order. Do not infer actions from the single entry returned by session.step().
        while emitted < len(session.events):
            ev = session.events[emitted]
            emitted += 1
            yield {**ev, "step": step}

    report = session.state.final_report
    yield {"type": "final",
           "text": report.investigation_summary if report else "",
           "reason": session.state.status,
           "structured": {
               "report": report.model_dump() if report else None,
               "agent_proposal": inv.final_proposal,
               "run_metadata": session.run_metadata(),
           },
           "usage": dict(inv.usage)}
    # The session object stays available to the caller (evaluation needs it).
    return
