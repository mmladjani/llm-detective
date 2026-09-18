"""case-004 (The Gallery Hall Theft) — a demo case for REVISE / self-correction.

A lying witness + a red herring: the obvious early accusation is insufficiently supported, so
the deterministic judge rejects it. These offline tests assert that:
  * the case exists and loads (statically in cases/),
  * it is solvable (the rule-based solver closes it with the correct culprit — the truth boundary holds),
  * a premature accusation is REJECTED, and the scripted agent goes reject -> revise -> outcome.
"""

from __future__ import annotations

from src.agent import AgentInvestigator
from src.cases import list_cases, load_case
from src.judge import check_conclusion
from src.loop import GameSession
from tests.test_agent_native import FakeClient, FakeResponse, ToolUseBlock, TextBlock


def test_case_004_is_registered_and_public_briefing_hides_truth():
    ids = [c["case_id"] for c in list_cases()]
    assert "case-004" in ids
    case = load_case("case-004")
    pb = case.public_briefing()
    # the public briefing must not carry the culprit or solution fields
    assert "culprit" not in pb and case.hidden_truth["culprit"] not in str(pb)


def test_case_004_is_solvable_truth_boundary_holds():
    case = load_case("case-004")
    s = GameSession(case)
    s.run()
    assert s.state.status == "solved"
    assert s.state.final_report.culprit == case.hidden_truth["culprit"]


def test_case_004_premature_accusation_is_rejected():
    """An early accusation on a thin state (only the report read) must fail."""
    case = load_case("case-004")
    s = GameSession(case)
    s.step()  # read_incident_report
    culprit = case.hidden_truth["culprit"]
    v = check_conclusion(s.state, {"verdict": "solved", "culprit": culprit,
                                   "reasoning": "deluje sumnjivo"})
    assert not v["ok"]
    assert len(v["issues"]) >= 2


def test_case_004_agent_reject_then_revise_flow():
    """Scripted agent: premature accusation -> judge rejects -> revise -> honest outcome."""
    script = [
        FakeResponse([ToolUseBlock("read_incident_report", {})]),
        FakeResponse([TextBlock("Early, but I'll try."),
                      ToolUseBlock("submit_conclusion",
                                   {"verdict": "solved",
                                    "culprit": load_case("case-004").hidden_truth["culprit"],
                                    "reasoning": "prvi utisak"})]),
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved",
                                    "reasoning": "nedovoljno nezavisnih linija zasad"})]),
    ]
    case = load_case("case-004")
    inv = AgentInvestigator(case, client=FakeClient(script))
    session = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = session
    while session.state.status == "investigating":
        session.step()
    gates = [e for e in inv.event_log if e.get("type") == "gate"]
    judged = [e for e in gates if e.get("phase") == "judge"]
    assert judged and judged[0]["passed"] is False           # early accusation rejected
    assert any(e.get("phase") == "revise" for e in gates)    # critique returned
