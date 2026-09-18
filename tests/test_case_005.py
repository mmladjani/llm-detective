"""case-005 (The Observatory Blind Spot) - adversarial stress case.

This case is intentionally mean to the investigator:
  * a planted Arun trail becomes the early lead,
  * two Celia signals share the same underlying badge source and must not count twice,
  * open questions block an otherwise strong conclusion,
  * the rule-based baseline solves it only at the edge of the budget.
"""

from __future__ import annotations

import json

from src.cases import list_cases, load_case
from src.evaluator import evaluate
from src.judge import check_conclusion
from src.loop import GameSession


def test_case_005_is_registered_and_briefing_hides_solution():
    ids = [c["case_id"] for c in list_cases()]
    assert "case-005" in ids

    case = load_case("case-005")
    briefing = json.dumps(case.public_briefing()).lower()
    for secret in ("spoofed_calibration_script", "hide_data_fabrication"):
        assert secret not in briefing
    assert "hidden_truth" not in case.public_briefing()


def test_case_005_planted_trail_temporarily_leads():
    case = load_case("case-005")
    session = GameSession(case)
    session.step()  # read_incident_report
    session.step()  # check_access_log
    session.step()  # review_camera

    assert session.state.leading().suspect == "arun"
    assert session.state.hypothesis_for("arun").confidence > session.state.hypothesis_for("celia").confidence


def test_case_005_json_source_groups_collapse_independence():
    case = load_case("case-005")
    session = GameSession(case)
    session.step()  # read_incident_report
    session.step()  # check_access_log: Celia badge signal
    session.step()  # review_camera: Celia lanyard signal, same source_group

    celia = session.state.hypothesis_for("celia")
    assert {"al_celia_badge", "cam_celia_lanyard"} <= set(celia.supporting_evidence)
    assert celia.independent_types == ["badge_controller"]


def test_case_005_open_questions_block_premature_celia_accusation():
    case = load_case("case-005")
    session = GameSession(case)
    for _ in range(7):
        session.step()

    assert session.state.leading().suspect == "celia"
    assert session.state.hypothesis_for("celia").confidence >= 0.70
    assert {q.suspect for q in session.state.open_questions} == {"arun", "bianca"}

    verdict = check_conclusion(
        session.state,
        {"verdict": "solved", "culprit": "celia", "reasoning": "strong physical trail"},
    )
    assert not verdict["ok"]
    assert any("Open questions remain" in issue for issue in verdict["issues"])


def test_case_005_is_solved_at_the_budget_edge():
    case = load_case("case-005")
    session = GameSession(case)
    session.run()
    report = session.state.final_report
    evaluation = evaluate(case, session.state)

    assert session.state.status == "solved"
    assert report.culprit == case.hidden_truth["culprit"] == "celia"
    assert report.method == "spoofed_calibration_script"
    assert report.motive == "hide_data_fabrication"
    assert len([t for t in session.state.tools_used if t["ok"]]) == case.budget
    assert session.state.remaining_budget == 0
    assert evaluation["missed_required_evidence"] == []
    assert evaluation["total_score"] >= 85
