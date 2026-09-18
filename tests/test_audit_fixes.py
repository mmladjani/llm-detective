"""Regression tests added during the technical audit.

Grouped by the four audit areas: hidden-truth boundary, confidence/evidence logic,
LLM validation & fallback honesty, and loop safety & state integrity.
"""

import copy
import json

import pytest

from src.cases import Case, load_case
from src.evaluator import evaluate
from src.investigator import (
    Investigator,
    LLMInvestigator,
    NextAction,
    RuleBasedInvestigator,
)
from src.loop import GameSession
from src.state import EvidenceItem

POISON = "ZZZ_POISON_TOKEN_42"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seeded(case_id="case-001") -> GameSession:
    """A session that has read the incident report (hypotheses seeded)."""
    s = GameSession(load_case(case_id))
    s.step()
    return s


def ev(id, supports=None, weight=0.0, itype="misc", relates=True, group=None):
    return EvidenceItem(id=id, description=f"desc-{id}", supports=supports,
                        weight=weight, independent_type=itype,
                        relates_to_incident=relates, source_group=group)


def perfect_decider_for(session):
    """A stand-in 'perfect' LLM: emits, as valid schema JSON, exactly the action the
    rule-based policy would take. Lets us prove a fully-LLM-driven run end to end."""
    rb = RuleBasedInvestigator(session.toolbox)

    def dec(_prompt):
        na = rb.next_action(session.state)
        if na is None:
            return json.dumps({"should_stop": True, "stop_reason": "nothing left"})
        return json.dumps({
            "selected_skill": na.skill, "next_question": na.question,
            "tool_name": na.tool, "tool_arguments": na.arguments,
            "should_stop": False, "stop_reason": None,
        })

    return dec


def llm(session, raw, mode="demo"):
    inv = LLMInvestigator(session.toolbox, briefing=session.case.public_briefing(),
                          mode=mode, raw_decider=raw)
    session.investigator = inv
    session.requested_mode = "llm"
    return session


# ===========================================================================
# 1. HIDDEN-TRUTH BOUNDARY
# ===========================================================================
def poisoned_session():
    c = load_case("case-001")
    data = copy.deepcopy(c._data)
    data["hidden_truth"] = {k: POISON for k in data["hidden_truth"]}
    return GameSession(Case(data)), Case(data)


def test_poison_token_never_visible_before_completion():
    session, case = poisoned_session()
    # Walk every iteration; at no point may the token appear in anything the
    # investigator/API/frontend can see (state, audit, run metadata).
    while session.state.status == "investigating":
        session.step()
        visible = json.dumps({
            "state": session.state.model_dump(),
            "audit": [a.model_dump() for a in session.audit],
            "run_metadata": session.run_metadata(),
        }, default=str)
        assert POISON not in visible
    # The report itself must not carry the poison truth either.
    assert POISON not in json.dumps(session.state.final_report.model_dump())
    # It is only reachable through the case's hidden_truth (the reveal path).
    assert all(v == POISON for v in case.hidden_truth.values())


def test_rulebased_input_surface_has_no_secrets():
    # The rule-based investigator only ever reads state (index/facts/hypotheses),
    # never the case object.
    s = seeded()
    rb = s.investigator
    assert not hasattr(rb, "_case")  # holds a toolbox, not the case
    # Its decision is a function of state alone.
    d = rb.decide(s.state)
    assert d.source == "rule_based"


# ===========================================================================
# 2. CONFIDENCE & EVIDENCE LOGIC
# ===========================================================================
def test_duplicate_evidence_not_counted_twice():
    s = seeded()
    e = ev("dup", supports="mara", weight=0.3, itype="access")
    s._apply_result(s.state, {"ok": True, "evidence": [e]})
    h = s.state.hypothesis_for("mara")
    c1, n1 = h.confidence, len(h.supporting_evidence)
    s._apply_result(s.state, {"ok": True, "evidence": [e]})  # same id again
    assert h.confidence == c1
    assert len(h.supporting_evidence) == n1 == 1


def test_contradiction_lowers_confidence():
    s = seeded()
    s._apply_result(s.state, {"ok": True, "evidence": [ev("s", "mara", 0.3, "access")]})
    up = s.state.hypothesis_for("mara").confidence
    s._apply_result(s.state, {"ok": True, "evidence": [ev("c", "mara", -0.25, "alibi")]})
    assert s.state.hypothesis_for("mara").confidence < up


def test_same_source_two_tools_is_one_independent_line():
    s = seeded()
    s._apply_result(s.state, {"ok": True, "evidence": [
        ev("a", "mara", 0.3, itype="access", group="badge_system"),
        ev("b", "mara", 0.3, itype="camera", group="badge_system"),
    ]})
    h = s.state.hypothesis_for("mara")
    assert h.independent_types == ["badge_system"]  # collapsed to one


def test_confidence_is_clamped():
    s = seeded()
    s._apply_result(s.state, {"ok": True, "evidence": [ev("big", "mara", 5.0, "access")]})
    assert s.state.hypothesis_for("mara").confidence <= 1.0
    assert s.state.hypothesis_for("mara").confidence == pytest.approx(0.98)
    s._apply_result(s.state, {"ok": True, "evidence": [ev("neg", "mara", -9.0, "x")]})
    assert s.state.hypothesis_for("mara").confidence >= 0.0
    assert s.state.hypothesis_for("mara").confidence == 0.0


def test_high_confidence_single_source_does_not_solve():
    s = seeded()
    s._apply_result(s.state, {
        "ok": True,
        "evidence": [ev("one", "mara", 0.6, "access")],
        "reveals": {"action": "replaced_manuscript", "method": "copied_access_card"},
    })
    h = s.state.hypothesis_for("mara")
    assert h.confidence >= 0.70 and len(h.independent_types) == 1
    # Even though confidence is high and action/method known, one independent line
    # must NOT satisfy the two-independent-evidence stopping rule.
    assert s._check_stop(s.state) != "solved"


def test_unresolved_question_blocks_solved():
    s = seeded()
    s._apply_result(s.state, {
        "ok": True,
        "evidence": [ev("a", "mara", 0.35, "access"), ev("b", "mara", 0.35, "camera")],
        "reveals": {"action": "replaced_manuscript", "method": "copied_access_card"},
        "raises": [{"suspect": "mara", "note": "loose end"}],
    })
    # Strong, two independent lines, action+method known — but an open question remains.
    assert s._check_stop(s.state) != "solved"


def test_unrelated_lie_does_not_raise_guilt():
    s = seeded()
    before = s.state.hypothesis_for("idris").confidence
    changes, deltas = s._apply_result(
        s.state, {"ok": True, "evidence": [ev("lie", "idris", 0.5, "alibi", relates=False)]}
    )
    assert s.state.hypothesis_for("idris").confidence == before  # unchanged
    d = [x for x in deltas if x.evidence_id == "lie"][0]
    assert d.reason == "unrelated_discounted" and d.previous == d.new


def test_confidence_deltas_are_structured_and_attributable():
    _, s = None, seeded()
    _, deltas = s._apply_result(s.state, {"ok": True, "evidence": [ev("z", "mara", 0.3, "access")]})
    d = deltas[0]
    assert d.suspect == "mara" and d.evidence_id == "z"
    assert d.reason == "support" and d.weight == 0.3
    assert d.new > d.previous
    # And it flows into the audit trace during a real run.
    s2 = GameSession(load_case("case-001"))
    s2.run()
    assert any(e.confidence_deltas for e in s2.audit)


def test_weights_are_data_driven_not_hardcoded_to_suspects():
    # The engine never references specific culprit names; swapping the culprit in a
    # case's data changes the outcome without any code change.
    import re
    src = open("src/loop.py").read()
    for name in ["mara", "idris", "noor", "tomas"]:
        assert not re.search(rf"['\"]{name}['\"]", src)


# ===========================================================================
# 3. LLM VALIDATION & FALLBACK HONESTY
# ===========================================================================
BAD_OUTPUTS = {
    "invalid_json": "not json {",
    "empty": "",
    "invalid_schema": json.dumps({"foo": "bar"}),
    "unknown_skill": json.dumps({"selected_skill": "telepathy", "tool_name": "check_access_log",
                                 "tool_arguments": {"location": "archive_room"}}),
    "unknown_tool": json.dumps({"selected_skill": "access_path_analysis", "tool_name": "hack",
                                "tool_arguments": {}}),
    "invalid_args": json.dumps({"selected_skill": "access_path_analysis",
                                "tool_name": "check_access_log", "tool_arguments": {}}),
    "unavailable_target": json.dumps({"selected_skill": "physical_evidence_analysis",
                                      "tool_name": "analyze_object",
                                      "tool_arguments": {"object": "ghost"}}),
}


@pytest.mark.parametrize("label,payload", list(BAD_OUTPUTS.items()))
def test_demo_fallback_is_never_silent(label, payload):
    s = GameSession(load_case("case-001"))
    llm(s, raw=lambda _p, _pay=payload: _pay, mode="demo")
    entry = s.step()
    assert entry.decision_source == "rule_based_fallback"
    assert entry.fallback_used is True
    assert entry.llm_called is True
    assert entry.validation_status == "failed"
    assert entry.failure_reason  # a concrete reason is recorded


def test_api_exception_falls_back_visibly():
    def boom(_p):
        raise TimeoutError("simulated timeout")

    s = GameSession(load_case("case-001"))
    llm(s, raw=boom, mode="demo")
    entry = s.step()
    assert entry.fallback_used and "api_exception" in entry.failure_reason


def test_premature_stop_is_recorded_as_llm_decision():
    s = GameSession(load_case("case-001"))
    s.step()  # read report first so there is state
    llm(s, raw=lambda _p: json.dumps({"should_stop": True, "stop_reason": "giving up"}))
    entry = s.step()
    assert entry.decision_source == "llm"
    assert s.state.status != "investigating"  # loop honored the stop


def test_evaluation_mode_does_not_mask_invalid_output():
    s = GameSession(load_case("case-001"))
    llm(s, raw=lambda _p: "garbage", mode="evaluation")
    s.run()
    assert s.state.status == "execution_error"
    meta = s.run_metadata()
    assert meta["invalid_decisions"] >= 1
    assert meta["fallback_decisions"] == 0
    assert meta["fully_llm_driven"] is False
    result = evaluate(s.case, s.state, run_metadata=meta)
    assert result["total_score"] == 0
    assert result["llm_integrity_ok"] is False


def test_fully_llm_driven_run_is_reported_honestly():
    s = GameSession(load_case("case-001"), requested_mode="llm")
    s.investigator = LLMInvestigator(s.toolbox, briefing=s.case.public_briefing(),
                                     mode="demo", raw_decider=perfect_decider_for(s))
    s.run()
    meta = s.run_metadata()
    assert s.state.status == "solved"
    assert meta["fallback_decisions"] == 0 and meta["invalid_decisions"] == 0
    assert meta["llm_decisions"] >= 1
    assert meta["fully_llm_driven"] is True
    assert evaluate(s.case, s.state, run_metadata=meta)["llm_integrity_ok"] is True


def test_fallback_heavy_run_not_reported_as_fully_llm():
    s = GameSession(load_case("case-001"))
    llm(s, raw=lambda _p: "still not json", mode="demo")
    s.run()
    meta = s.run_metadata()
    assert meta["fallback_decisions"] >= 1
    assert meta["fully_llm_driven"] is False
    assert evaluate(s.case, s.state, run_metadata=meta)["llm_integrity_ok"] is False


# ===========================================================================
# 4. LOOP SAFETY & STATE INTEGRITY
# ===========================================================================
class AdversarialInvestigator(Investigator):
    """Always emits a valid-tool call with a fresh, unavailable target."""
    name = "adversarial"

    def __init__(self):
        self.n = 0

    def next_action(self, state):
        self.n += 1
        return NextAction("physical_evidence_analysis", "q", "analyze_object",
                          {"object": f"ghost_{self.n}"})


def test_illegal_action_limit_terminates():
    """Illegal moves are free (a name error must not bring down the case), so an agent
    stuck in a loop no longer spends budget — so its own counter stops it."""
    s = GameSession(load_case("case-001"), investigator=AdversarialInvestigator())
    trace = s.run()
    assert s.state.status == "illegal_action_limit"
    assert s.state.final_report is not None
    assert s.state.iteration <= s.max_iterations
    assert len(trace) <= s.max_iterations + 2


def test_hard_iteration_limit_still_terminates(monkeypatch):
    """The hard iteration ceiling is defense in depth — it must hold even when the counter
    illegal moves is not the one that stops the run."""
    monkeypatch.setattr("src.loop.MAX_ILLEGAL_ACTIONS", 10_000)
    s = GameSession(load_case("case-001"), investigator=AdversarialInvestigator())
    trace = s.run()
    assert s.state.status == "execution_error"
    assert s.state.final_report is not None
    assert s.state.iteration <= s.max_iterations
    assert len(trace) <= s.max_iterations + 2


def test_budget_never_negative_and_illegal_calls_are_free():
    """An illegal move does not touch the world, so we don't charge for it — but the
    budget must never be allowed to go negative either."""
    s = GameSession(load_case("case-001"), investigator=AdversarialInvestigator())
    start = s.state.remaining_budget
    s.step()  # illegal call
    assert s.state.remaining_budget == start
    assert s.state.illegal_actions == 1
    s.run()
    assert s.state.remaining_budget >= 0


def test_tool_exception_is_controlled():
    s = GameSession(load_case("case-001"))
    orig = s.toolbox.call

    def raising(tool, args=None):
        raise ValueError("boom")

    s.toolbox.call = raising
    # Must not raise; run ends in a terminal state with a recorded failed call.
    s.run()
    assert s.state.status != "investigating"
    assert any(a.tool is not None and a.tool_ok is False for a in s.audit)
    s.toolbox.call = orig


def test_invalid_tool_call_does_not_corrupt_state():
    s = seeded()
    snapshot = s.state.model_dump_json()
    s._apply_result(s.state, {"ok": False, "error": "nope", "evidence": []})
    assert s.state.model_dump_json() == snapshot  # a failed result changes nothing


def test_evaluator_does_not_mutate_state():
    s = GameSession(load_case("case-001"))
    s.run()
    before = s.state.model_dump_json()
    evaluate(s.case, s.state, run_metadata=s.run_metadata())
    assert s.state.model_dump_json() == before


def test_one_tool_entry_per_iteration():
    s = GameSession(load_case("case-001"))
    s.run()
    tool_entries = [a for a in s.audit if a.tool is not None]
    assert len(tool_entries) == s.state.iteration


def test_terminal_status_and_stop_reason_consistent():
    for cid in ["case-001", "case-002", "case-003"]:
        s = GameSession(load_case(cid))
        s.run()
        assert s.state.final_report.status == s.state.status
        assert s.audit[-1].skill == "final_case_synthesis"
        assert s.audit[-1].continue_investigation is False
