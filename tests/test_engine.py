"""Behavioural tests for the deduction engine.

These target the product's promises, not implementation trivia:
the hidden-truth boundary, deterministic tools, budget accounting, the audit trail, the
three case outcomes, and the evaluator.
"""

import json

import pytest

from src.cases import CASES_DIR, Case, load_case
from src.evaluator import evaluate
from src.loop import GameSession, InvestigationComplete
from src.tools import ToolBox


def solve(case_id):
    case = load_case(case_id)
    session = GameSession(case)
    session.run()
    return case, session


# 1. Hidden truth never reaches the investigator's input surface.
def test_hidden_truth_not_in_investigator_input():
    case = load_case("case-001")
    from src.methods import METHOD_DEFINITIONS
    public, board = case.public_briefing(), case.catalog()
    # A universal vocabulary may name a method, but never selects one as true.
    assert public.pop("method_definitions") == METHOD_DEFINITIONS
    assert board.pop("method_definitions") == METHOD_DEFINITIONS
    briefing = json.dumps(public).lower()
    catalog = json.dumps(board).lower()
    truth = case.hidden_truth
    # Suspect names are necessarily public (they are named as persons of interest).
    # What must NOT leak is the *solution*: which action/method/motive occurred.
    for secret in [truth["method"], truth["motive"], truth["action"]]:
        assert secret not in briefing
        assert secret not in catalog
    # The catalog must not silently rank or single out the culprit.
    assert "culprit" not in catalog and "hidden" not in catalog
    # And the public briefing carries none of the evaluator/solution keys.
    assert "hidden_truth" not in case.public_briefing()
    assert "required_evidence" not in case.public_briefing()
    assert "evaluation" not in case.public_briefing()


@pytest.mark.parametrize("case_id", ["case-001", "case-011", "case-012", "case-013"])
def test_public_contract_does_not_depend_on_hidden_truth_or_tool_contents(case_id):
    import copy
    case = load_case(case_id)
    changed = copy.deepcopy(case._data)
    changed["hidden_truth"] = {"culprit": "SECRET_SENTINEL", "method": "SECRET_SENTINEL"}
    changed["required_evidence"] = ["SECRET_SENTINEL"]
    changed["evaluation"] = {"expected_status": "SECRET_SENTINEL"}
    for tool, targets in changed["tools"].items():
        if tool != "read_incident_report":
            for target in targets:
                targets[target] = {"observation": "SECRET_SENTINEL", "evidence": []}
    altered = Case(changed)
    assert altered.public_briefing() == case.public_briefing()
    assert altered.catalog() == case.catalog()


# 2. Tools expose only permitted evidence (never the whole case / hidden truth).
def test_tool_result_excludes_hidden_truth():
    case = load_case("case-001")
    tb = ToolBox(case)
    result = tb.call("read_incident_report")
    blob = json.dumps(result, default=str).lower()
    assert "hidden_truth" not in blob
    assert case.hidden_truth["motive"] not in blob
    # An access-log call returns only its own observation + evidence.
    r2 = tb.call("check_access_log", {"location": "archive_room"})
    assert set(r2.keys()) >= {"tool", "observation", "evidence", "ok"}
    assert "required_evidence" not in json.dumps(r2, default=str)


# 3. Tool responses are deterministic.
def test_tools_are_deterministic():
    tb = ToolBox(load_case("case-001"))
    a = tb.call("check_access_log", {"location": "archive_room"})
    b = tb.call("check_access_log", {"location": "archive_room"})
    assert json.dumps(a, default=str) == json.dumps(b, default=str)


# 4. Budget decreases by exactly one per successful action.
def test_budget_decrements():
    case = load_case("case-001")
    session = GameSession(case)
    start = session.state.remaining_budget
    session.step()  # one successful tool call
    assert session.state.remaining_budget == start - 1


# 5. State updates are recorded (facts / evidence / hypotheses grow).
def test_state_updates_recorded():
    _, session = solve("case-001")
    st = session.state
    assert st.known_facts
    assert st.evidence_collected
    assert st.hypotheses
    assert st.facts_revealed  # action/method/etc discovered from evidence


# 6. An audit entry is created for every iteration.
def test_audit_entry_per_iteration():
    _, session = solve("case-001")
    action_entries = [a for a in session.audit if a.tool is not None]
    successful_calls = [t for t in session.state.tools_used if t["ok"]]
    assert len(action_entries) == len(successful_calls)
    assert session.audit[-1].skill == "final_case_synthesis"  # closing entry


# 7. Rule-based investigator solves Case 1 correctly and efficiently.
def test_case1_solved():
    case, session = solve("case-001")
    r = session.state.final_report
    assert session.state.status == "solved"
    assert r.culprit == case.hidden_truth["culprit"] == "mara"
    assert r.method == case.hidden_truth["method"]
    assert r.action == case.hidden_truth["action"]
    used = len([t for t in session.state.tools_used if t["ok"]])
    assert used <= 6


# 8. Case 2: the unrelated liar is NOT accused; the real culprit is found.
def test_case2_does_not_accuse_liar():
    case, session = solve("case-002")
    r = session.state.final_report
    assert r.culprit == case.hidden_truth["culprit"] == "idris"
    assert r.culprit != "tomas"
    tomas = session.state.hypothesis_for("tomas")
    assert tomas.confidence < 0.5  # the lie never raised his guilt
    # The lie was actually encountered and set aside.
    assert any("tomas" in c.lower() for c in session.state.unverified_claims)


# 9. Case 3: honest unresolved outcome, no accusation.
def test_case3_unresolved():
    _, session = solve("case-003")
    assert session.state.status in ("unresolved", "budget_exhausted")
    r = session.state.final_report
    assert r.culprit is None
    assert r.action is None and r.method is None


# 10. Evaluator scores correct and incorrect conclusions properly.
def test_evaluator_scores():
    case1, s1 = solve("case-001")
    e1 = evaluate(case1, s1.state)
    assert e1["correct"]["culprit"] is True
    assert e1["total_score"] >= 80
    assert e1["missed_required_evidence"] == []

    case3, s3 = solve("case-003")
    e3 = evaluate(case3, s3.state)
    assert e3["correct_unresolved_handling"] is True
    assert e3["total_score"] >= 70

    # A fabricated wrong conclusion must score poorly.
    s1.state.final_report.culprit = "lena"
    s1.state.final_report.method = "nonsense"
    bad = evaluate(case1, s1.state)
    assert bad["correct"]["culprit"] is False
    assert bad["total_score"] < e1["total_score"]


# 11. Invalid tool calls do not crash the run.
def test_invalid_tool_call_does_not_crash():
    tb = ToolBox(load_case("case-001"))
    r = tb.call("interview_character", {"suspect": "does_not_exist"})
    assert r["ok"] is False and r["error"]
    r2 = tb.call("teleport", {})  # unknown tool
    assert r2["ok"] is False
    # A session survives a bogus manual call.
    session = GameSession(load_case("case-001"))
    session.toolbox.call("analyze_object", {"object": "nope"})
    session.run()
    assert session.state.status in ("solved", "unresolved", "budget_exhausted")


# 12. An investigator cannot continue after completion.
def test_cannot_continue_after_completion():
    _, session = solve("case-001")
    assert session.state.status != "investigating"
    with pytest.raises(InvestigationComplete):
        session.step()


# Bonus: all case files load and are internally consistent.
def test_all_cases_load():
    for path in CASES_DIR.glob("case_*.json"):
        data = json.loads(path.read_text())
        case = Case(data)
        assert case.hidden_truth["culprit"] in case.index["suspects"]
        assert case.index["location"]
