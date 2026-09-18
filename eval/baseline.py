"""Phase 0 — measurement foundation.

Run ALL cases × N, for both the AGENT (llm) and the BASELINE (rule_based), compute
the central KPI (agent-vs-baseline gap), and write:
  * every run to history (outputs/runs/…),
  * a machine summary (outputs/baseline_<ts>.json),
  * a readable report (outputs/baseline_report.md).

Without ANTHROPIC_API_KEY: only the baseline (rule_based) runs, and the agent columns are N/A —
still useful as a lower bound and as a pipeline check.

    python eval/baseline.py --runs 3
    python eval/baseline.py --runs 5 --baseline-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.cases import list_cases, load_case          # noqa: E402
from src.evaluator import evaluate                    # noqa: E402
from src.investigator import RuleBasedInvestigator    # noqa: E402
from src.loop import GameSession                       # noqa: E402
from src import history                                # noqa: E402
from eval import metrics                               # noqa: E402


def _row(case, session, evaluation, usage, seconds, mode) -> dict:
    """A flat row for metrics (single-level, easy to aggregate)."""
    correct = evaluation.get("correct") or {}
    return {
        "case_id": case.case_id,
        "mode": mode,
        "status": session.state.status,
        "expected_status": evaluation.get("expected_status"),
        "score": evaluation.get("total_score"),
        "status_match": evaluation.get("status_match"),
        "culprit_correct": bool(correct.get("culprit")),
        "actions_used": evaluation.get("actions_used"),
        "budget": evaluation.get("budget"),
        "correct_unresolved_handling": evaluation.get("correct_unresolved_handling"),
        "llm_integrity_ok": evaluation.get("llm_integrity_ok"),
        "seconds": round(seconds, 2),
        "input_tokens": (usage or {}).get("input_tokens"),
        "output_tokens": (usage or {}).get("output_tokens"),
        "llm_calls": (usage or {}).get("llm_calls"),
    }


def run_baseline_once(case_id: str, save: bool) -> dict:
    case = load_case(case_id)
    s = GameSession(case, requested_mode="rule_based")
    s.investigator = RuleBasedInvestigator(s.toolbox)
    t0 = time.time()
    s.run()
    secs = time.time() - t0
    e = evaluate(case, s.state, s.run_metadata())
    if save:
        history.record_from_trace(case, s, e, usage={}, seconds=secs, include_trace=True)
    return _row(case, s, e, {}, secs, "rule_based")


def run_agent_once(case_id: str, save: bool) -> dict:
    from src.agent import iter_agent  # local: requires anthropic + a key
    cap: dict = {}
    final = None
    t0 = time.time()
    for ev in iter_agent(case_id, _capture=cap):
        if ev["type"] == "final":
            final = ev
    secs = time.time() - t0
    case, session = cap["case"], cap["session"]
    e = evaluate(case, session.state, session.run_metadata())
    usage = (final or {}).get("usage") or {}
    proposal = ((final or {}).get("structured") or {}).get("agent_proposal")
    if save:
        history.record_from_trace(case, session, e, usage=usage, seconds=secs,
                                  agent_proposal=proposal, include_trace=True)
    return _row(case, session, e, usage, secs, "llm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--case", default=None)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--include-generated", action="store_true",
                    help="include cases/generated/ too (complex cases, Phase 1)")
    ap.add_argument("--no-save", action="store_true", help="do not write history records")
    args = ap.parse_args()

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    do_agent = have_key and not args.baseline_only
    save = not args.no_save

    case_ids = ([args.case] if args.case
                else [c["case_id"] for c in list_cases(include_generated=args.include_generated)])
    agent_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for cid in case_ids:
        for i in range(1, args.runs + 1):
            b = run_baseline_once(cid, save)
            baseline_rows.append(b)
            line = (f"{cid} run {i}/{args.runs}  baseline: status={b['status']:16} "
                    f"score={b['score']:3} actions={b['actions_used']}/{b['budget']}")
            if do_agent:
                a = run_agent_once(cid, save)
                agent_rows.append(a)
                line += (f"  | agent: status={a['status']:16} score={a['score']:3} "
                         f"actions={a['actions_used']}/{a['budget']} "
                         f"integrity={a['llm_integrity_ok']}")
            print(line)

    # ---- KPI ----
    gap = (metrics.agent_vs_baseline_gap(agent_rows, baseline_rows)
           if agent_rows else {"baseline": metrics.summarize_mode(baseline_rows),
                               "agent": None})
    per_case = {}
    for cid in case_ids:
        b = [r for r in baseline_rows if r["case_id"] == cid]
        a = [r for r in agent_rows if r["case_id"] == cid]
        per_case[cid] = (metrics.agent_vs_baseline_gap(a, b) if a
                         else {"baseline": metrics.summarize_mode(b), "agent": None})

    ts = int(time.time())
    payload = {"runs_per_case": args.runs, "have_agent": bool(agent_rows),
               "overall": gap, "per_case": per_case}
    (BASE / "outputs" / f"baseline_{ts}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = _report_md(payload)
    (BASE / "outputs" / "baseline_report.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nSaved: outputs/baseline_{ts}.json + outputs/baseline_report.md")
    if save:
        print(f"History: outputs/runs/ ({len(baseline_rows)+len(agent_rows)} records)")


def _fmt(v):
    return "N/A" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def _report_md(payload: dict) -> str:
    o = payload["overall"]
    a = o.get("agent")
    b = o.get("baseline")
    lines = ["# trace — Baseline report (Phase 0)", ""]
    lines.append(f"Runs per case: **{payload['runs_per_case']}**  ·  "
                 f"agent included: **{'yes' if payload['have_agent'] else 'no (no key)'}**")
    lines.append("")
    lines.append("## Central KPI — agent vs baseline")
    lines.append("")
    if a:
        lines.append("| metric | agent (llm) | baseline (rule_based) | gap |")
        lines.append("|---|---|---|---|")
        rows = [("accuracy", "accuracy", "accuracy_gap"),
                ("avg_score", "avg_score", "score_gap"),
                ("steps_to_convergence", "steps_to_convergence", "steps_gap"),
                ("false_accusation_rate", "false_accusation_rate", "false_accusation_gap")]
        for label, key, gapkey in rows:
            lines.append(f"| {label} | {_fmt(a[key])} | {_fmt(b[key])} | {_fmt(o.get(gapkey))} |")
        lines.append("")
        lines.append("> `steps_gap` < 0 = the agent reaches the correct conclusion with fewer moves "
                     "(more efficient).  `accuracy_gap` > 0 = the agent is more accurate.  This is proof "
                     "that the loop pays off.")
    else:
        lines.append("_The agent was not run (no ANTHROPIC_API_KEY). Only the baseline is shown "
                     "as a lower bound._")
        lines.append("")
        lines.append("| metrika | baseline (rule_based) |")
        lines.append("|---|---|")
        for k in ("runs", "accuracy", "avg_score", "steps_to_convergence",
                  "false_accusation_rate", "honest_unresolved_rate"):
            lines.append(f"| {k} | {_fmt(b[k])} |")
    lines.append("")
    lines.append("## Per case")
    lines.append("")
    lines.append("| case | baseline score | baseline steps | agent score | agent steps | steps_gap |")
    lines.append("|---|---|---|---|---|---|")
    for cid, g in payload["per_case"].items():
        gb = g["baseline"]; ga = g.get("agent")
        lines.append(f"| {cid} | {_fmt(gb['avg_score'])} | {_fmt(gb['steps_to_convergence'])} "
                     f"| {_fmt(ga['avg_score']) if ga else 'N/A'} "
                     f"| {_fmt(ga['steps_to_convergence']) if ga else 'N/A'} "
                     f"| {_fmt(g.get('steps_gap')) if ga else 'N/A'} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
