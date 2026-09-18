"""Scope of the two checks that read the agent's PROSE.

Neither knows the hidden truth; both ask whether what the agent WROTE matches what its
tools actually returned. They split the work by what is machine-checkable:

  * the evaluator screens ids and clock times deterministically, offline;
  * the LLM reviewer (judge.py layer 2) handles invented prose, which no regex can see.

These tests pin the boundary, and the blocking/advisory split that keeps the reviewer
from rejecting a conclusion merely because it would have phrased it more weakly.
"""

from __future__ import annotations

import json

from src.cases import load_case
from src.evaluator import _grade_agent_summary
from src.judge import _split_findings, make_llm_conclusion_judge
from src.loop import GameSession

from tests.test_agent_native import FakeResponse, TextBlock


def _solved_session() -> GameSession:
    s = GameSession(load_case("case-001"))
    s.run()
    assert s.state.status == "solved"
    return s


class _Reviewer:
    """Minimal client returning one canned reviewer reply."""

    def __init__(self, text: str):
        self._text = text

        class _Messages:
            @staticmethod
            def create(**_kw):
                payload = json.loads(_kw["messages"][0]["content"])
                if "blocking_findings" in payload:
                    return FakeResponse([TextBlock(json.dumps({"decisions": [
                        {"index": i, "disposition": "uphold", "reason": finding}
                        for i, finding in enumerate(payload["blocking_findings"])]}))])
                parsed = json.loads(text)
                for finding in parsed.get("findings", []):
                    if finding.get("type") == "blocking":
                        finding.update(_claim_reference())
                return FakeResponse([TextBlock(json.dumps(parsed))])

        self.messages = _Messages()


def _claim_reference():
    return {"claim_id": "conclusion.culprit", "claim_quote": "mara",
            "evidence_ids": [], "repair": "Revise the unsupported claim."}


def _blocking(detail):
    return json.dumps({"findings": [{"type": "blocking", "detail": detail, **_claim_reference()}]})


# --- the blocking / advisory split ---------------------------------------------
def test_split_findings_separates_blocking_from_advisory():
    blocking, advisory = _split_findings({"findings": [
        {"type": "blocking", "detail": "cites a badge scan no tool returned"},
        {"type": "advisory", "detail": "presence is weaker than proof of the act"},
    ]})
    assert blocking == ["cites a badge scan no tool returned"]
    assert advisory == ["presence is weaker than proof of the act"]


def test_split_findings_accepts_the_older_string_shape():
    """Injected test reviewers still return {"ok":..., "issues":[str]}; a bare string
    blocks, exactly as every finding did before the split existed."""
    blocking, advisory = _split_findings({"ok": False, "issues": ["unsupported leap"]})
    assert blocking == ["unsupported leap"]
    assert advisory == []


def test_an_advisory_only_review_does_not_block():
    """The regression this guards: an 'I would have said it more weakly' objection used
    to reject a conclusion as hard as a fabricated fact, which taught the agent to pass
    by softening prose rather than by investigating."""
    s = _solved_session()
    judge = make_llm_conclusion_judge(
        _Reviewer('{"findings": [{"type": "advisory", '
                  '"detail": "fiber contact is not proof of use"}]}'), "m")

    out = judge(s.state, {"verdict": "solved", "culprit": "mara", "reasoning": "x"})

    assert out["ok"] is True
    assert out["issues"] == []
    assert out["advisories"] == ["fiber contact is not proof of use"]


def test_a_blocking_review_still_rejects():
    s = _solved_session()
    judge = make_llm_conclusion_judge(
        _Reviewer('{"findings": [{"type": "blocking", '
                  '"detail": "quotes a two-building badge hit that is not on record"}]}'), "m")

    out = judge(s.state, {"verdict": "solved", "culprit": "mara", "reasoning": "x"})

    assert out["ok"] is False
    assert out["issues"][0].startswith("quotes a two-building badge hit that is not on record")


def test_a_refusal_with_no_findings_still_says_something_actionable():
    s = _solved_session()
    judge = make_llm_conclusion_judge(_Reviewer('{"ok": false, "findings": []}'), "m")

    out = judge(s.state, {"verdict": "solved", "culprit": "mara", "reasoning": "x"})

    assert out["ok"] is False
    assert out["issues"], "a rejection the agent cannot act on is not a rejection"


def test_explicit_refusal_is_not_overridden_by_advisory_findings():
    s = _solved_session()
    judge = make_llm_conclusion_judge(_Reviewer(
        '{"ok": false, "findings": [{"type":"advisory", "detail":"weak phrasing"}]}'), "m")
    assert judge(s.state, {"verdict": "solved", "culprit": "mara"})["ok"] is False


def test_critique_requests_state_repair_not_only_prose_changes():
    from src.agent import _critique_body
    critique = _critique_body({"issues": ["Unsupported stored assessment."]})
    assert "assess_evidence" in critique
    assert "establish_fact" in critique
    assert "not just the final prose" in critique


def test_live_control_fixtures_are_legal_and_do_not_send_expected_labels():
    """Fixture plumbing only; semantic accuracy requires running the live harness."""
    import json
    from eval.reviewer_controls import controls
    from src.judge import _evidence_digest
    from tests.test_agent_native import FakeClient
    for name, state, proposal, expected in controls():
        assert state.remaining_budget >= 0
        assert state.inference_mode == "model"
        assert not state.open_questions
        assert all(e.attribution == "model" for e in state.evidence_collected)
        client = FakeClient([FakeResponse([TextBlock('{"findings": []}')])])
        make_llm_conclusion_judge(client, "offline")(state, proposal)
        payload = json.loads(client.messages.calls[0]["messages"][0]["content"])
        assert set(payload) == {"proposed_conclusion", "evidence_record", "current_claims"}
        assert payload["evidence_record"] == _evidence_digest(state)
        assert "expected_accept" not in json.dumps(payload)
        assert "hidden_truth" not in json.dumps(payload)


def test_false_objection_can_be_withdrawn_only_with_a_record_quote():
    from eval.reviewer_controls import controls
    from tests.test_agent_native import FakeClient
    _, state, proposal, _ = controls()[0]
    quote = "The archive entry at 21:17 used the duplicate"
    client = FakeClient([
        FakeResponse([TextBlock(_blocking("No record says the duplicate was used."))]),
        FakeResponse([TextBlock(json.dumps({"decisions": [{"index": 0,
            "disposition": "withdraw", "record_quote": quote,
            "reason": "The observation explicitly identifies the credential used."}]}))]),
    ])
    review = make_llm_conclusion_judge(client, "offline")(state, proposal)
    assert review["ok"]
    assert len(client.messages.calls) == 2
    assert review["finding_audit"][0]["original_finding"].startswith("No record says the duplicate was used.")
    assert review["finding_audit"][0]["record_quote"] == quote
    assert review["advisories"]


def test_finding_audit_fails_closed_on_missing_decisions_or_invented_quotes():
    from eval.reviewer_controls import controls
    from tests.test_agent_native import FakeClient
    _, state, proposal, _ = controls()[0]
    for decisions in [[], [{"index": 0, "disposition": "withdraw", "reason": "x",
                           "record_quote": "Invented observation not collected."}],
                      [{"index": True, "disposition": "uphold", "reason": "x"}]]:
        client = FakeClient([
            FakeResponse([TextBlock(_blocking("A defect."))]),
            FakeResponse([TextBlock(json.dumps({"decisions": decisions}))]),
        ])
        review = make_llm_conclusion_judge(client, "offline")(state, proposal)
        assert not review["ok"]
        assert review["error"] == "finding_audit_failed"


def test_audit_can_narrow_a_partly_wrong_finding_but_not_accept_it():
    from eval.reviewer_controls import controls
    from tests.test_agent_native import FakeClient
    _, state, proposal, _ = controls()[0]
    client = FakeClient([
        FakeResponse([TextBlock(_blocking("Mixed objection."))]),
        FakeResponse([TextBlock('{"decisions":[{"index":0,"disposition":"uphold",'
                                 '"reason":"A material invented claim remains."}]}')]),
    ])
    review = make_llm_conclusion_judge(client, "offline")(state, proposal)
    assert not review["ok"]
    assert review["issues"] == ["A material invented claim remains."]


def test_current_claim_map_replaces_the_retracted_assessment():
    from eval.reviewer_controls import controls
    from src.judge import _current_claims, _evidence_digest
    fixtures = {name: (state, proposal) for name, state, proposal, _ in controls()}
    bad, proposal = fixtures["unsupported_current_assessment"]
    repaired, _ = fixtures["corrected_current_assessment"]
    before = _current_claims(bad, proposal)["evidence.obj_card.assessment"]
    assert "proves Mara personally manufactured" in before
    after = _current_claims(repaired, proposal)["evidence.obj_card.assessment"]
    assert "who made or supplied the copy remains unknown" in after
    assert before not in json.dumps(_evidence_digest(repaired))
    assert repaired.remaining_budget == bad.remaining_budget


def test_method_contract_is_shared_without_selected_answers():
    from src.agent import AgentInvestigator
    from src.judge import _evidence_digest
    from tests.test_agent_native import FakeClient
    for cid in ("case-011", "case-012", "case-013"):
        case = load_case(cid)
        s = GameSession(case)
        s.step()
        inv = AgentInvestigator(case, client=FakeClient([]))
        definitions = case.public_briefing()["method_definitions"]
        assert set(definitions) == set(case.fact_candidates["method"])
        assert _evidence_digest(s.state)["method_definitions"] == definitions
        assert s.state.index["method_definitions"] == definitions
        for definition in definitions.values():
            assert definition in inv._messages[0]["content"]


def test_unanchored_or_malformed_blocking_claims_fail_closed():
    from eval.reviewer_controls import controls
    from tests.test_agent_native import FakeClient
    _, state, proposal, _ = controls()[0]
    for fields in [
        {},
        {**_claim_reference(), "claim_id": "old_history.assessment"},
        {**_claim_reference(), "claim_quote": "Mara personally manufactured the copy"},
        {**_claim_reference(), "evidence_ids": ["invented_record"]},
        {**_claim_reference(), "repair": ""},
    ]:
        reply = {"findings": [{"type": "blocking", "detail": "Unsupported claim.", **fields}]}
        client = FakeClient([FakeResponse([TextBlock(json.dumps(reply))])])
        review = make_llm_conclusion_judge(client, "offline")(state, proposal)
        assert not review["ok"]
        assert review["error"] == "unparseable_review"
        assert len(client.messages.calls) == 1


def test_label_scope_audit_uses_a_checked_definition_not_fabricated_evidence():
    from eval.reviewer_controls import controls
    from tests.test_agent_native import FakeClient
    _, state, proposal, _ = controls()[0]
    definition = "Does NOT assert who manufactured or obtained the copy."
    good = {"index": 0, "disposition": "withdraw", "basis": "method_definition",
            "method_label": "copied_access_card", "definition_quote": definition,
            "reason": "The objection adds a maker requirement absent from the method label."}
    for change, expected in [({}, True),
                             ({"method_label": "borrowed_key"}, False),
                             ({"definition_quote": "The suspect is always guilty."}, False),
                             ({"definition_quote": ""}, False),
                             ({"basis": "agent_opinion"}, False)]:
        client = FakeClient([
            FakeResponse([TextBlock(_blocking("The selected method requires proof of manufacture."))]),
            FakeResponse([TextBlock(json.dumps({"decisions": [{**good, **change}]}))]),
        ])
        review = make_llm_conclusion_judge(client, "offline")(state, proposal)
        assert review["ok"] == expected
        if not expected:
            assert review["error"] == "finding_audit_failed"


# --- the evaluator's deterministic groundedness screen --------------------------
def test_groundedness_scans_the_reasoning_not_only_the_summary():
    """The reasoning is where the case is argued, and where invented detail shows up.
    Grading only the tidy summary left the chain of inference unchecked."""
    s = _solved_session()
    report = s.state.final_report
    report.agent_summary = "mara swapped the manuscript."
    report.agent_reasoning = "the ghost_corridor camera settles it."

    grade = _grade_agent_summary(report, s.state)

    assert "ghost_corridor" in grade["fabricated_entities"]


def test_a_clock_time_no_tool_returned_is_flagged():
    s = _solved_session()
    report = s.state.final_report
    report.agent_summary = ""
    report.agent_reasoning = "the second entry at 03:45 proves the return trip."

    grade = _grade_agent_summary(report, s.state)

    assert grade["invented_times"] == ["03:45"]


def test_times_that_are_on_the_record_are_not_flagged():
    s = _solved_session()
    report = s.state.final_report
    report.agent_summary = ""
    report.agent_reasoning = "her card opened the archive at 21:17."

    grade = _grade_agent_summary(report, s.state)

    assert grade["invented_times"] == []
