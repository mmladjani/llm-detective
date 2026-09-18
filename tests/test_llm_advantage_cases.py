"""Cases where semantic prioritization beats the fixed deterministic baseline.

These are not live-model tests. The scripted investigator below represents the kind of
choice an LLM can make from the visible trace: jump to the clue-rich target instead of
following the baseline's fixed physical-scan order.
"""

from __future__ import annotations

import json

from src.cases import list_cases, load_case
from src.evaluator import evaluate
from src.investigator import Investigator, NextAction
from src.loop import GameSession


class ClueDrivenInvestigator(Investigator):
    name = "llm_clue_driven"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def next_action(self, state):
        if self.i >= len(self.actions):
            return None
        skill, question, tool, arguments = self.actions[self.i]
        self.i += 1
        return NextAction(skill, question, tool, arguments)


CASE_006_PATH = [
    ("timeline_reconstruction", "Read the report and identify wording traps.", "read_incident_report", {}),
    ("statement_validation", "Clear Quinn's blue-folder language trap first.", "verify_alibi", {"suspect": "quinn"}),
    ("access_path_analysis", "Check who used the relay workstation.", "check_access_log", {"location": "mailroom_security_desk"}),
    ("physical_evidence_analysis", "Inspect the desk for motive and action traces.", "inspect_location", {"location": "mailroom_security_desk"}),
    ("physical_evidence_analysis", "Analyze the named printer queue instead of generic desk clutter.", "analyze_object", {"object": "printer_queue"}),
]


CASE_007_PATH = [
    ("timeline_reconstruction", "Read the report and identify the named export artifact.", "read_incident_report", {}),
    ("physical_evidence_analysis", "Go straight to the scheduler terminal named in the report.", "analyze_object", {"object": "scheduler_terminal"}),
    ("statement_validation", "Resolve Luca's harmless-label claim.", "verify_alibi", {"suspect": "luca"}),
    ("access_path_analysis", "Check the terminal access token.", "check_access_log", {"location": "scheduling_office"}),
    ("physical_evidence_analysis", "Inspect the office for motive and action context.", "inspect_location", {"location": "scheduling_office"}),
]


def _run_clue_driven(case_id, path):
    case = load_case(case_id)
    investigator = ClueDrivenInvestigator(path)
    session = GameSession(case, investigator=investigator, requested_mode="llm")
    session.run()
    return case, session


def _actions_used(session):
    return len([t for t in session.state.tools_used if t["ok"]])


def test_llm_advantage_cases_are_registered_and_hide_solution():
    ids = [c["case_id"] for c in list_cases()]
    assert {"case-006", "case-007"} <= set(ids)

    for cid, secrets in {
        "case-006": ("printer_queue_macro", "cover_badge_clone_sales"),
        "case-007": ("scheduler_terminal_export", "manipulate_triage_metrics"),
    }.items():
        case = load_case(cid)
        briefing = json.dumps(case.public_briefing()).lower()
        assert "hidden_truth" not in case.public_briefing()
        for secret in secrets:
            assert secret not in briefing


def test_case_006_fixed_policy_wastes_budget_but_clue_driven_solves():
    case = load_case("case-006")
    baseline = GameSession(case)
    baseline.run()
    assert baseline.state.status != "solved"
    assert _actions_used(baseline) == case.budget
    assert "printer_queue" not in [t["arg_key"] for t in baseline.state.tools_used]

    _, strategic = _run_clue_driven("case-006", CASE_006_PATH)
    report = strategic.state.final_report
    score = evaluate(case, strategic.state)

    assert strategic.state.status == "solved"
    assert report.culprit == "petra"
    assert report.method == "printer_queue_macro"
    assert _actions_used(strategic) < _actions_used(baseline)
    assert score["missed_required_evidence"] == []
    assert score["total_score"] >= 85


def test_case_007_fixed_policy_wastes_budget_but_clue_driven_solves():
    case = load_case("case-007")
    baseline = GameSession(case)
    baseline.run()
    assert baseline.state.status != "solved"
    assert _actions_used(baseline) == case.budget
    assert "scheduler_terminal" not in [t["arg_key"] for t in baseline.state.tools_used]

    _, strategic = _run_clue_driven("case-007", CASE_007_PATH)
    report = strategic.state.final_report
    score = evaluate(case, strategic.state)

    assert strategic.state.status == "solved"
    assert report.culprit == "luca"
    assert report.method == "scheduler_terminal_export"
    assert _actions_used(strategic) < _actions_used(baseline)
    assert score["missed_required_evidence"] == []
    assert score["total_score"] >= 85
