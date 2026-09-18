"""The expanded generator (more suspects, herrings, decoy objects),
the history store (round-trip), and metrics (agent-vs-baseline gap). All offline, no key."""

from __future__ import annotations

import pytest

from src.generator import build_and_verify, build_case, validate_recipe
from src.loop import GameSession
from src.investigator import RuleBasedInvestigator
from src.evaluator import evaluate
from src.cases import load_case, list_cases
from src import history
from eval import metrics

NAMES8 = ["Mara", "Idris", "Lena", "Tomas", "Sara", "Petar", "Nina", "Goran"]


# ---------------- Generator ----------------
def test_recipe_allows_up_to_eight_suspects():
    r = validate_recipe({"incident_type": "theft", "suspects": NAMES8, "culprit": "random"})
    assert len(r["suspects"]) == 8
    with pytest.raises(ValueError):
        validate_recipe({"suspects": NAMES8 + ["Extra"]})  # 9 -> too many


@pytest.mark.parametrize("nsus", [3, 5, 8])
@pytest.mark.parametrize("twist", ["none", "lying_witness", "insufficient"])
def test_expanded_cases_pass_self_check(nsus, twist):
    recipe = {"incident_type": "sabotage", "suspects": NAMES8[:nsus], "culprit": "random",
              "twist": twist, "num_red_herrings": 2, "decoy_objects": 3,
              "multi_day": True, "budget": 14, "seed": nsus}
    case = build_and_verify(recipe)  # raises if the deterministic solver cannot solve it
    assert case.budget == 14


def test_decoy_objects_appear_but_are_not_required():
    case = build_case({"incident_type": "theft", "suspects": NAMES8[:4], "culprit": "random",
                       "twist": "none", "decoy_objects": 3, "budget": 12, "seed": 7})
    objs = case.index["objects"]
    assert len(objs) == 2 + 3          # 2 incident objects + 3 decoys
    for req in case.required_evidence:  # decoy evidence is not required
        assert req in ("al_culprit", "cam_culprit", "obj_method")


def test_more_suspects_grows_action_space():
    def space(case):
        S = len(case.index["suspects"]); L = len(case.index["locations"]); O = len(case.index["objects"])
        return 1 + 2 * S + S * (S - 1) // 2 + 3 * L + O
    small = build_case({"incident_type": "theft", "suspects": NAMES8[:3], "twist": "none",
                        "budget": 11, "seed": 1})
    big = build_case({"incident_type": "theft", "suspects": NAMES8[:8], "twist": "none",
                      "num_red_herrings": 3, "decoy_objects": 4, "budget": 14, "seed": 2})
    assert space(big) >= 2 * space(small)   # At least twice the action space


# ---------------- history store ----------------
def _run_baseline(case_id: str):
    case = load_case(case_id)
    s = GameSession(case, requested_mode="rule_based")
    s.investigator = RuleBasedInvestigator(s.toolbox)
    s.run()
    e = evaluate(case, s.state, s.run_metadata())
    return case, s, e


def test_history_round_trip(tmp_path):
    runs = tmp_path / "runs"
    case, s, e = _run_baseline("case-001")
    rec = history.record_from_trace(case, s, e, seconds=0.1, runs_dir=runs)
    assert rec["app"] == "trace"
    assert rec["labels"]["hidden_truth"]["culprit"]        # label for eval/training
    assert rec["trace"] and isinstance(rec["trace"], list)  # full trace for analysis
    assert rec["extra"]["events"] == s.events

    loaded = list(history.iter_records(runs_dir=runs))
    assert len(loaded) == 1 and loaded[0]["run_id"] == rec["run_id"]
    assert loaded[0]["extra"]["events"] == s.events
    idx = history.load_index(runs_dir=runs)
    assert idx and idx[0]["task_id"] == "case-001"


def test_history_summarize(tmp_path):
    runs = tmp_path / "runs"
    for cid in [c["case_id"] for c in list_cases()]:
        case, s, e = _run_baseline(cid)
        history.record_from_trace(case, s, e, seconds=0.1, runs_dir=runs)
    summ = history.summarize(runs_dir=runs)
    assert summ["total_runs"] == len(list_cases())
    grp = summ["groups"][0]
    assert grp["mode"] == "rule_based" and grp["runs"] == len(list_cases())


# ---------------- metrics ----------------
def test_agent_vs_baseline_gap_shapes():
    baseline = [{"case_id": "c1", "expected_status": "solved", "culprit_correct": True,
                 "status_match": True, "score": 80, "actions_used": 8, "budget": 12,
                 "correct_unresolved_handling": None}]
    agent = [{"case_id": "c1", "expected_status": "solved", "culprit_correct": True,
              "status_match": True, "score": 92, "actions_used": 5, "budget": 12,
              "correct_unresolved_handling": None}]
    gap = metrics.agent_vs_baseline_gap(agent, baseline)
    assert gap["score_gap"] == 12
    assert gap["steps_gap"] == -3          # agent 3 moves more efficient
    assert gap["accuracy_gap"] == 0.0


def test_false_accusation_rate():
    rows = [{"expected_status": "unresolved", "correct_unresolved_handling": True},
            {"expected_status": "unresolved", "correct_unresolved_handling": False}]
    assert metrics.false_accusation_rate(rows) == 0.5
    assert metrics.honest_unresolved_rate(rows) == 0.5
