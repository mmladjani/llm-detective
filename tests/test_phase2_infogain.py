"""Phase 2: info-gain metric (quality of move selection) + skill registration.
Offline — a fabricated trace + one real rule_based run, no key."""

from __future__ import annotations

from src.cases import load_case
from src.loop import GameSession
from src.investigator import RuleBasedInvestigator
from src import skills as skills_mod
from src import history
from eval import metrics


def test_information_gain_skill_registered():
    assert "information_gain" in skills_mod.SKILLS
    got = skills_mod.read_skill("information_gain")
    assert got["ok"] and "information" in got["content"].lower()


def test_step_info_quality_flags_waste():
    trace = [
        # productive: moved a hypothesis
        {"tool": "check_access_log", "tool_ok": True, "decision_source": "llm",
         "state_changes": ["x"], "confidence_deltas": [{"suspect": "mara", "delta": 0.28}]},
        # wasted: successful but informationally empty (e.g. a decoy object)
        {"tool": "analyze_object", "tool_ok": True, "decision_source": "llm",
         "state_changes": [], "confidence_deltas": []},
        # illegal/failed: wasted
        {"tool": "verify_alibi", "tool_ok": False, "decision_source": "llm",
         "state_changes": [], "confidence_deltas": []},
        # 'closing' is not counted
        {"tool": None, "decision_source": "closing", "state_changes": ["stop"],
         "confidence_deltas": []},
    ]
    q = metrics.step_info_quality(trace)
    assert q["actions"] == 3            # closing excluded
    assert q["productive"] == 1
    assert q["wasted"] == 2
    assert q["wasted_rate"] == round(2 / 3, 3)
    assert q["info_efficiency"] == round(1 / 3, 2)


def test_rule_based_run_is_mostly_productive():
    case = load_case("case-001")
    s = GameSession(case, requested_mode="rule_based")
    s.investigator = RuleBasedInvestigator(s.toolbox)
    s.run()
    trace = [a.model_dump() for a in s.audit]
    q = metrics.step_info_quality(trace)
    # Baseline front-loads high-info physical evidence -> no wasted moves.
    assert q["actions"] >= 3
    assert q["wasted_rate"] == 0.0
    assert q["info_efficiency"] > 0


def test_info_quality_over_records(tmp_path):
    runs = tmp_path / "runs"
    for cid in ("case-001", "case-002"):
        case = load_case(cid)
        s = GameSession(case, requested_mode="rule_based")
        s.investigator = RuleBasedInvestigator(s.toolbox)
        s.run()
        e = {"total_score": 0}  # evaluator dict not needed for trace-based metric
        from src.evaluator import evaluate
        e = evaluate(case, s.state, s.run_metadata())
        history.record_from_trace(case, s, e, seconds=0.1, runs_dir=runs)
    recs = list(history.iter_records(runs_dir=runs))
    agg = metrics.info_quality_over_records(recs)
    assert agg["runs"] == 2
    assert agg["avg_wasted_rate"] == 0.0
