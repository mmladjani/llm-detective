"""Eval: all built-in cases x N runs -> score, actions, fallback rate, consistency.

Fallback/invalid decisions do NOT count as an LLM success (llm_integrity) — that is the main
lesson of this app: evaluation honesty.

    python eval/run_all.py --mode rule_based --runs 1   # offline, no key needed
    export ANTHROPIC_API_KEY=sk-ant-...
    python eval/run_all.py --runs 3                     # the live agent
    python eval/run_all.py --compare --runs 3           # both, side by side

`--compare` evaluates both drivers on the same selected cases, not the different
AI and Legacy case lists shown in the demo UI. Model-inference cases use a deliberately
naive uniform/lexical baseline on the same unlabelled records; authored cases retain
their checklist policy. Keep case version, public information, budget and evaluator
fixed; control human answers when testing interactive cases. Results measure these
policies on these cases, not LLM superiority over every deterministic approach.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.agent import iter_agent          # noqa: E402
from src.evaluator import evaluate        # noqa: E402
from src.cases import list_cases, load_case  # noqa: E402
from src.loop import GameSession          # noqa: E402
from src.human_responder import CLIResponder  # noqa: E402
from src import history                   # noqa: E402
from eval.model_baseline import run_naive_model_baseline  # noqa: E402


def _result(session, e, md, u, seconds) -> dict:
    return {
        "status": session.state.status,
        "score": e["total_score"],
        "status_match": e["status_match"],
        "llm_integrity_ok": e.get("llm_integrity_ok"),
        "actions": e["actions_used"],
        "budget": e["budget"],
        "fallback_decisions": md["fallback_decisions"],
        "fully_llm_driven": md["fully_llm_driven"],
        "inference_mode": md.get("inference_mode"),
        "assessments": md.get("assessments"),
        "summary_fabrications": len(
            (e.get("agent_summary_grade") or {}).get("fabricated_entities", [])),
        "seconds": round(seconds, 1),
        "tokens_in": u.get("input_tokens"),
        "tokens_out": u.get("output_tokens"),
        "llm_calls": u.get("llm_calls"),
    }


def run_once(case_id: str, save_history: bool = True) -> dict:
    t0 = time.time()
    cap: dict = {}
    final = None
    for ev in iter_agent(case_id, _capture=cap, responder=CLIResponder()):
        if ev["type"] == "final":
            final = ev
    session = cap["session"]
    e = evaluate(cap["case"], session.state, session.run_metadata())
    u = (final or {}).get("usage") or {}
    if save_history:
        proposal = ((final or {}).get("structured") or {}).get("agent_proposal")
        history.record_from_trace(cap["case"], session, e, usage=u,
                                  seconds=time.time() - t0, agent_proposal=proposal)
    return _result(session, e, session.run_metadata(), u, time.time() - t0)


def run_once_baseline(case_id: str) -> dict:
    """The fixed deterministic policy. No key, no network, no variance.

    Model cases use uniform weights and lexical candidate matching, not oracle labels.
    """
    t0 = time.time()
    case = load_case(case_id)
    if case.inference_mode == "model":
        session = run_naive_model_baseline(case)
    else:
        session = GameSession(case, requested_mode="rule_based")
        session.run()
    e = evaluate(case, session.state, session.run_metadata())
    result = _result(session, e, session.run_metadata(), {}, time.time() - t0)
    result["baseline_policy"] = "uniform_lexical" if case.inference_mode == "model" else "authored_checklist"
    return result


def _avg(values: list) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


def _summarize(label: str, results: dict[str, list[dict]]) -> None:
    print(f"\n--- {label} ---")
    for cid, runs in results.items():
        statuses = Counter(r["status"] for r in runs)
        top, top_n = statuses.most_common(1)[0]
        avg = _avg([r["score"] for r in runs])
        integrity = sum(1 for r in runs if r.get("llm_integrity_ok"))
        print(f"{cid}: agreement={top_n / len(runs):.2f} (most common: {top}) "
              f"score avg={'n/a' if avg is None else f'{avg:.0f}'} "
              f"integrity_ok={integrity}/{len(runs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--case", default=None, help="a single case only (case-001…)")
    ap.add_argument("--mode", choices=("llm", "rule_based"), default="llm",
                    help="which driver to evaluate (rule_based needs no API key)")
    ap.add_argument("--compare", action="store_true",
                    help="run BOTH drivers and print the score delta per case")
    ap.add_argument("--include-authored", action="store_true",
                    help="include legacy authored cases in live LLM evaluation")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be at least 1")

    # Batch runs must not unexpectedly wait for a human. Select an interactive
    # case explicitly with --case; its questions then use the CLI responder.
    case_ids = [args.case] if args.case else [c["case_id"] for c in list_cases()
                if not c["human_played"] and (c["inference"] == "model"
                or args.include_authored or (args.mode == "rule_based" and not args.compare))]
    run_llm = args.compare or args.mode == "llm"
    run_base = args.compare or args.mode == "rule_based"

    llm_results: dict[str, list[dict]] = {}
    base_results: dict[str, list[dict]] = {}

    for cid in case_ids:
        if run_base:
            # Deterministic: one run is the whole distribution.
            base_results[cid] = [run_once_baseline(cid)]
            b = base_results[cid][0]
            print(f"{cid} rule_based: status={b['status']} score={b['score']}")
        if run_llm:
            llm_results[cid] = []
            for i in range(1, args.runs + 1):
                r = run_once(cid)
                llm_results[cid].append(r)
                print(f"{cid} llm run {i}/{args.runs}: status={r['status']} "
                      f"score={r['score']} actions={r['actions']}/{r['budget']} "
                      f"fallback={r['fallback_decisions']} "
                      f"tokens={r['tokens_in']}/{r['tokens_out']}")

    if run_base:
        _summarize("RULE-BASED BASELINE", base_results)
    if run_llm:
        _summarize("LLM SUMMARY", llm_results)

    if args.compare:
        print("\n--- LLM vs BASELINE (score delta) ---")
        print(f"{'case':<12} {'baseline':>9} {'llm avg':>9} {'delta':>8}  note")
        for cid in case_ids:
            b = base_results[cid][0]["score"]
            l = _avg([r["score"] for r in llm_results[cid]])
            if b is None:
                note = base_results[cid][0].get("skipped", "")
                print(f"{cid:<12} {'n/a':>9} "
                      f"{'n/a' if l is None else f'{l:9.0f}'} {'—':>8}  {note}")
                continue
            delta = (l or 0) - b
            note = ("no measured score advantage over this baseline"
                    if delta <= 0 else "")
            print(f"{cid:<12} {b:>9} {l:9.0f} {delta:+8.0f}  {note}")

    payload = {"llm": llm_results, "rule_based": base_results}
    out = BASE / "outputs" / f"eval_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
