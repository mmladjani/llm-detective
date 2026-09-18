"""Run history — a persistent record of EVERY run, for later analysis, eval and training.

Each run is saved as one standalone JSON file in ``outputs/runs/``, plus one compact line
in ``outputs/runs/index.jsonl`` (for quick scanning without opening every record). The
record is DELIBERATELY app-agnostic: the fields ``app`` / ``task`` / ``outcome`` mean the
same thing in both trace and wiki-claim, so the same schema and the same analysis tools
can be shared across the apps.

Why this way:
  * one file per run -> easy to append, nothing is overwritten, safe for parallelism;
  * index.jsonl -> `pandas.read_json(lines=True)` or `jq` can aggregate immediately;
  * the full trace in the record -> enough for SFT/DPO examples and post-hoc eval, not just the score.

The record is created AFTER the run (out-of-band), so it may contain the hidden truth — it
serves as a label for eval/training and never goes back into the agent's context.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

SCHEMA_VERSION = 1

BASE = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE / "outputs" / "runs"
INDEX_PATH = RUNS_DIR / "index.jsonl"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def build_record(
    *,
    app: str,
    task: dict[str, Any],
    mode: str,
    outcome: dict[str, Any],
    provenance: Optional[dict[str, Any]] = None,
    usage: Optional[dict[str, Any]] = None,
    seconds: Optional[float] = None,
    trace: Optional[list[dict[str, Any]]] = None,
    final: Optional[dict[str, Any]] = None,
    labels: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the canonical run record. All fields are plain JSON (dict/list/scalar).

    Parameters (app-agnostic):
      app        -> "trace" | "wiki-claim" | ...
      task       -> what was solved: {id, title, difficulty, budget, expected_*}
      mode       -> "llm" | "rule_based" | "llm_schema" | ...
      outcome    -> the measurable result: {status/verdict, score, status_match, ...}
      provenance -> honesty metadata (run_metadata: fully_llm_driven, fallback...)
      usage      -> {input_tokens, output_tokens, llm_calls}
      seconds    -> run duration
      trace      -> list of steps (audit) — for post-hoc analysis and training examples
      final      -> the final report / agent proposal
      labels     -> the hidden truth etc. (label for eval/training)
      extra      -> anything app-specific
    """
    run_id = uuid.uuid4().hex[:12]
    rec = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "app": app,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_ts": time.time(),
        "mode": mode,
        "task": task or {},
        "outcome": outcome or {},
        "provenance": provenance or {},
        "usage": usage or {},
        "seconds": seconds,
        "final": final,
        "labels": labels or {},
    }
    if trace is not None:
        rec["trace"] = trace
    if extra:
        rec["extra"] = extra
    return rec


def _index_row(rec: dict[str, Any]) -> dict[str, Any]:
    """One flat line for index.jsonl — only what's needed for filtering/aggregation."""
    t = rec.get("task", {})
    o = rec.get("outcome", {})
    p = rec.get("provenance", {})
    u = rec.get("usage", {})
    return {
        "run_id": rec["run_id"],
        "app": rec.get("app"),
        "created_at": rec.get("created_at"),
        "mode": rec.get("mode"),
        "model": p.get("model"),
        "task_id": t.get("id"),
        "difficulty": t.get("difficulty"),
        "budget": t.get("budget"),
        "status": o.get("status") or o.get("verdict"),
        "score": o.get("score"),
        "status_match": o.get("status_match"),
        "actions_used": o.get("actions_used"),
        "llm_integrity_ok": o.get("llm_integrity_ok"),
        "fully_llm_driven": p.get("fully_llm_driven"),
        "fallback_decisions": p.get("fallback_decisions"),
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "seconds": rec.get("seconds"),
    }


def save_record(rec: dict[str, Any], runs_dir: Path | None = None) -> Path:
    """Write the record as a standalone file and append a line to index.jsonl. Return the path."""
    d = runs_dir or RUNS_DIR
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(rec.get("created_ts") or time.time()))
    task_id = (rec.get("task", {}) or {}).get("id") or "run"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(task_id))
    fname = f"{ts}_{rec.get('app','app')}_{rec.get('mode','?')}_{safe}_{rec['run_id']}.json"
    path = d / fname
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    index = (d / "index.jsonl")
    with index.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_index_row(rec), ensure_ascii=False) + "\n")
    return path


def save_run(**kwargs) -> Path:
    """Shortcut: build_record(**kwargs) -> save_record(...)."""
    runs_dir = kwargs.pop("runs_dir", None)
    return save_record(build_record(**kwargs), runs_dir=runs_dir)


# --------------------------------------------------------------------------- #
# Reading / analysis
# --------------------------------------------------------------------------- #
def iter_records(runs_dir: Path | None = None) -> Iterator[dict[str, Any]]:
    """Iterate over all full records (sorted by creation time)."""
    d = runs_dir or RUNS_DIR
    if not d.exists():
        return
    files = sorted(d.glob("*.json"))
    for p in files:
        if p.name == "index.jsonl":
            continue
        try:
            yield json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue


def load_index(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load the compact index lines (fast, without the full traces)."""
    d = runs_dir or RUNS_DIR
    idx = d / "index.jsonl"
    if not idx.exists():
        return []
    rows = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --------------------------------------------------------------------------- #
# Trace adapter — turn a finished GameSession into a canonical record
# --------------------------------------------------------------------------- #
def record_from_trace(
    case: Any,
    session: Any,
    evaluation: dict[str, Any],
    *,
    usage: Optional[dict[str, Any]] = None,
    seconds: Optional[float] = None,
    agent_proposal: Optional[dict[str, Any]] = None,
    include_trace: bool = True,
    save: bool = True,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    """Map trace's (case, session, evaluation) onto an app-agnostic run record.

    include_trace=False -> record without the full audit (outcome only), for light batch evals.
    save=False          -> just return the record (used by tests to avoid touching disk).
    """
    md = session.run_metadata()
    state = session.state
    report = getattr(state, "final_report", None)
    final = {
        "report": report.model_dump() if report is not None else None,
        "agent_proposal": agent_proposal,
        "summary": getattr(report, "investigation_summary", "") if report else "",
    }
    outcome = {
        "status": state.status,
        "score": evaluation.get("total_score"),
        "expected_status": evaluation.get("expected_status"),
        "status_match": evaluation.get("status_match"),
        "correct": evaluation.get("correct"),
        "unsupported_claims": evaluation.get("unsupported_claims"),
        "missed_required_evidence": evaluation.get("missed_required_evidence"),
        "actions_used": evaluation.get("actions_used"),
        "correct_unresolved_handling": evaluation.get("correct_unresolved_handling"),
        "llm_integrity_ok": evaluation.get("llm_integrity_ok"),
    }
    task = {
        "id": case.case_id,
        "title": getattr(case, "title", None),
        "difficulty": getattr(case, "difficulty", None),
        "budget": case.budget,
        "expected_status": case.evaluation.get("expected_status"),
    }
    trace = None
    if include_trace:
        trace = [a.model_dump() for a in getattr(session, "audit", [])]

    rec = build_record(
        app="trace",
        task=task,
        mode=md.get("requested_mode") or "unknown",
        outcome=outcome,
        provenance=md,
        usage=usage or {},
        seconds=seconds,
        trace=trace,
        extra={"events": list(getattr(session, "events", []))} if include_trace else None,
        final=final,
        labels={"hidden_truth": dict(case.hidden_truth),
                "required_evidence": list(case.required_evidence),
                # How honest the human player actually was, per answer. Engine-side
                # like hidden_truth: derived from the character's secret knowledge, so
                # it belongs with the labels and never with the trace the agent saw.
                # Empty for cases with no human-played character.
                "human_interviews": list(
                    getattr(session.toolbox, "human_interview_audit", []) or []),
                },
    )
    if save:
        save_record(rec, runs_dir=runs_dir)
    return rec


def summarize(rows: Iterable[dict[str, Any]] | None = None,
              runs_dir: Path | None = None) -> dict[str, Any]:
    """Aggregate by (app, mode): number of runs, average score, status agreement,
    integrity rate, average tokens. Works over the index lines."""
    data = list(rows) if rows is not None else load_index(runs_dir)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in data:
        groups.setdefault((r.get("app"), r.get("mode")), []).append(r)

    out: dict[str, Any] = {"total_runs": len(data), "groups": []}
    for (app, mode), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        scores = [r["score"] for r in rs if isinstance(r.get("score"), (int, float))]
        integ = [r for r in rs if r.get("llm_integrity_ok") is not None]
        matches = [r for r in rs if r.get("status_match") is not None]
        out["groups"].append({
            "app": app,
            "mode": mode,
            "runs": len(rs),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
            "status_match_rate": round(sum(1 for r in matches if r["status_match"]) / len(matches), 2)
                                 if matches else None,
            "integrity_ok_rate": round(sum(1 for r in integ if r["llm_integrity_ok"]) / len(integ), 2)
                                 if integ else None,
        })
    return out
