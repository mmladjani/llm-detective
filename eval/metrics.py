"""Metrics over run records (history records or raw results).
Agent-versus-baseline differences are meaningful only for matched cases and settings.

All functions operate on a list of dicts with the fields produced by eval/baseline.py:
    {case_id, mode, status, expected_status, score, status_match, actions_used,
     budget, correct_unresolved_handling, culprit_correct, seconds,
     input_tokens, output_tokens}
Deliberately no pandas dependency — stdlib is enough and testable.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable


def _num(xs: Iterable[Any]) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float))]


def avg(xs: Iterable[Any]) -> float | None:
    v = _num(xs)
    return round(mean(v), 2) if v else None


def accuracy(rows: list[dict]) -> float | None:
    """Share of runs with the correct outcome: right culprit on solved, or a correct honest-unresolved."""
    judged = [r for r in rows if r.get("status_match") is not None]
    if not judged:
        return None
    ok = 0
    for r in judged:
        if r.get("expected_status") == "solved":
            ok += 1 if r.get("culprit_correct") else 0
        else:
            ok += 1 if r.get("correct_unresolved_handling") else 0
    return round(ok / len(judged), 3)


def false_accusation_rate(rows: list[dict]) -> float | None:
    """On UNsolvable cases: how often the agent accused someone anyway."""
    unsolvable = [r for r in rows if r.get("expected_status") != "solved"]
    if not unsolvable:
        return None
    bad = sum(1 for r in unsolvable if r.get("correct_unresolved_handling") is False)
    return round(bad / len(unsolvable), 3)


def honest_unresolved_rate(rows: list[dict]) -> float | None:
    unsolvable = [r for r in rows if r.get("expected_status") != "solved"]
    if not unsolvable:
        return None
    good = sum(1 for r in unsolvable if r.get("correct_unresolved_handling") is True)
    return round(good / len(unsolvable), 3)


def steps_to_convergence(rows: list[dict]) -> float | None:
    """Average number of actions used on SOLVED runs with the correct culprit —
    a measure of efficiency (how quickly the agent reaches the right conclusion)."""
    solved_ok = [r for r in rows
                 if r.get("expected_status") == "solved" and r.get("culprit_correct")]
    return avg([r.get("actions_used") for r in solved_ok])


def summarize_mode(rows: list[dict]) -> dict[str, Any]:
    """Summary for one set of runs (usually one mode: llm or rule_based)."""
    return {
        "runs": len(rows),
        "accuracy": accuracy(rows),
        "avg_score": avg([r.get("score") for r in rows]),
        "steps_to_convergence": steps_to_convergence(rows),
        "avg_actions": avg([r.get("actions_used") for r in rows]),
        "false_accusation_rate": false_accusation_rate(rows),
        "honest_unresolved_rate": honest_unresolved_rate(rows),
        "avg_tokens_in": avg([r.get("input_tokens") for r in rows]),
        "avg_tokens_out": avg([r.get("output_tokens") for r in rows]),
        "avg_seconds": avg([r.get("seconds") for r in rows]),
    }


def agent_vs_baseline_gap(agent_rows: list[dict], baseline_rows: list[dict]) -> dict[str, Any]:
    """CENTRAL KPI. A positive `accuracy_gap` or lower `steps` = the loop pays off.

    steps_gap < 0 means: the agent reaches the correct conclusion with FEWER moves than the
    brute-force baseline (that's the good side). accuracy_gap > 0 means the agent is more accurate.
    """
    a = summarize_mode(agent_rows)
    b = summarize_mode(baseline_rows)

    def diff(key: str):
        if a.get(key) is None or b.get(key) is None:
            return None
        return round(a[key] - b[key], 3)

    return {
        "agent": a,
        "baseline": b,
        "accuracy_gap": diff("accuracy"),
        "score_gap": diff("avg_score"),
        "steps_gap": diff("steps_to_convergence"),   # negative = agent more efficient
        "false_accusation_gap": diff("false_accusation_rate"),  # negative = agent better
    }


def group_by_case(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("case_id", "?"), []).append(r)
    return out


# --------------------------------------------------------------------------- #
# Information quality (info-gain) over the audit trace
# --------------------------------------------------------------------------- #
def _real_actions(trace: list[dict]) -> list[dict]:
    """Real moves: without the 'closing' synthesis and without empty (tool=None) entries."""
    return [e for e in (trace or [])
            if e.get("decision_source") != "closing" and e.get("tool")]


def step_info_quality(trace: list[dict]) -> dict[str, Any]:
    """Did the agent pick moves that changed the stored state?

    Productive move = successful and it produced a state change or moved a hypothesis
    (`state_changes` or `confidence_deltas`). Wasted = successful but informationally
    empty (e.g. analyzing a decoy object), or illegal/failed.

    - wasted_rate: share of wasted moves (lower = better planning).
    - info_efficiency: average number of hypothesis moves per action (higher = better).
    """
    acts = _real_actions(trace)
    if not acts:
        return {"actions": 0, "productive": 0, "wasted": 0,
                "wasted_rate": None, "info_efficiency": None}
    productive = wasted = total_deltas = 0
    for e in acts:
        changes = e.get("state_changes") or []
        deltas = e.get("confidence_deltas") or []
        total_deltas += len(deltas)
        if e.get("tool_ok") and (changes or deltas):
            productive += 1
        else:
            wasted += 1
    n = len(acts)
    return {"actions": n, "productive": productive, "wasted": wasted,
            "wasted_rate": round(wasted / n, 3),
            "info_efficiency": round(total_deltas / n, 2)}


def info_quality_over_records(records: Iterable[dict]) -> dict[str, Any]:
    """Aggregate info-quality across several history records (which carry a `trace`)."""
    per = [step_info_quality(r.get("trace") or []) for r in records]
    per = [p for p in per if p["actions"]]
    if not per:
        return {"runs": 0, "avg_wasted_rate": None, "avg_info_efficiency": None,
                "avg_actions": None}
    return {
        "runs": len(per),
        "avg_wasted_rate": round(mean(p["wasted_rate"] for p in per), 3),
        "avg_info_efficiency": round(mean(p["info_efficiency"] for p in per), 2),
        "avg_actions": round(mean(p["actions"] for p in per), 2),
    }
