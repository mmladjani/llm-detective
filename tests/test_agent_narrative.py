"""The agent's own words survive into the report, and get checked.

Before this, `submit_conclusion`'s reasoning was stored on the investigator and then
dropped: `_synthesize` rebuilt the summary from engine state with an f-string, so the
report a reader saw contained no sentence the model had written. These tests pin down
that the narrative now lands in the report and that the evaluator screens it for
groundedness — an invented entity is penalized like an unsupported evidence claim.

Also covers extended thinking: the deliberation has to reach the trace as `think`
events, or "the agent reasons" is an assertion rather than something you can read.
"""

from __future__ import annotations

import pytest

from src.agent import AgentInvestigator, iter_agent, thinking_config
from src.cases import load_case
from src.evaluator import evaluate
from src.loop import GameSession

from tests.test_agent_native import (
    FakeClient,
    FakeResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
)


class ThinkingBlock:
    type = "thinking"

    def __init__(self, thinking):
        self.thinking = thinking


def _run(script, case_id="case-001"):
    case = load_case(case_id)
    inv = AgentInvestigator(case, client=FakeClient(script))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    while s.state.status == "investigating":
        s.step()
    return case, s, inv


def _conclude(summary: str, verdict: str = "unresolved", culprit=None):
    args = {"verdict": verdict, "reasoning": "Thin evidence.", "summary": summary}
    if culprit:
        args["culprit"] = culprit
    return [
        FakeResponse([ToolUseBlock("read_incident_report", {})]),
        FakeResponse([ToolUseBlock("submit_conclusion", args)]),
    ]


# --- the narrative reaches the report ------------------------------------------
def test_agent_summary_lands_in_the_report():
    _, s, _ = _run(_conclude("I could not establish who swapped the manuscript."))
    report = s.state.final_report
    assert report.agent_summary == "I could not establish who swapped the manuscript."
    assert report.agent_reasoning == "Thin evidence."
    # The engine's factual recap is kept alongside, not overwritten: they are scored
    # on different things and a reader must be able to tell them apart.
    assert report.investigation_summary
    assert report.investigation_summary != report.agent_summary


def test_report_without_an_agent_narrative_is_still_complete():
    """Rule-based runs write no narrative; that must not produce a broken report."""
    case = load_case("case-001")
    s = GameSession(case, requested_mode="rule_based")
    s.run()
    assert s.state.final_report.agent_summary == ""
    assert s.state.final_report.investigation_summary


# --- the evaluator screens it --------------------------------------------------
def test_grounded_summary_is_not_penalized():
    case, s, _ = _run(_conclude(
        "The archive_room yielded nothing tying mara to the swap."))
    result = evaluate(case, s.state, s.run_metadata())
    grade = result["agent_summary_grade"]
    assert grade["present"] is True
    assert grade["fabricated_entities"] == []


def test_fabricated_entity_in_the_summary_is_penalized():
    """A corridor no tool ever reported is the narrative form of a made-up fact.

    Scored on case-003, where the expected outcome IS "unresolved": an honest
    abstention there scores 90, so the penalty is visible as plain arithmetic instead
    of disappearing into the clamp at zero that a failing run sits against.
    """
    case, s, _ = _run(_conclude(
        "The thief left via the service_stairwell using a cloned master_badge."),
        case_id="case-003")
    result = evaluate(case, s.state, s.run_metadata())
    assert set(result["agent_summary_grade"]["fabricated_entities"]) == {
        "service_stairwell", "master_badge"}

    clean_case, clean_s, _ = _run(_conclude("Nothing was established."),
                                  case_id="case-003")
    clean = evaluate(clean_case, clean_s.state, clean_s.run_metadata())
    assert clean["total_score"] == 90
    # Same run, same evidence, same verdict — the only difference is the invention.
    assert result["total_score"] == 80


def test_ids_the_run_actually_collected_are_not_flagged():
    """Groundedness is per-run, not per-case: an id that exists in the case file but
    was never obtained is still a claim the agent cannot support."""
    case = load_case("case-001")
    script = [
        FakeResponse([ToolUseBlock("read_incident_report", {})]),
        FakeResponse([ToolUseBlock("check_access_log", {"location": "archive_room"})]),
        FakeResponse([ToolUseBlock("submit_conclusion", {
            "verdict": "unresolved", "reasoning": "One line only.",
            "summary": "The al_mara_entry record puts mara in the archive_room, "
                       "but that is a single line of evidence."})]),
    ]
    inv = AgentInvestigator(case, client=FakeClient(script))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    while s.state.status == "investigating":
        s.step()

    assert any(e.id == "al_mara_entry" for e in s.state.evidence_collected)
    grade = evaluate(case, s.state, s.run_metadata())["agent_summary_grade"]
    assert grade["fabricated_entities"] == []


def test_citing_evidence_the_run_never_collected_is_flagged():
    case, s, _ = _run(_conclude(
        "The obj_card analysis proves it."))  # never analyzed in this run
    grade = evaluate(case, s.state, s.run_metadata())["agent_summary_grade"]
    assert grade["fabricated_entities"] == ["obj_card"]


def test_grade_reports_whether_the_summary_names_the_culprit():
    """Reported, not punished: incoherence is worth seeing even when unscored."""
    case, s, _ = _run(_conclude("Nothing was established."))
    grade = evaluate(case, s.state, s.run_metadata())["agent_summary_grade"]
    # An unresolved verdict has no culprit to name.
    assert grade["mentions_culprit"] is None


def test_rule_based_run_is_scored_exactly_as_before():
    """The screen must not silently rescore the existing corpus.

    A rule-based run produces no narrative, so it can fabricate nothing and its score
    has to be untouched by this feature. case-001 scored 90 before the change.
    """
    case = load_case("case-001")
    s = GameSession(case, requested_mode="rule_based")
    s.run()
    result = evaluate(case, s.state, s.run_metadata())
    assert result["total_score"] == 90
    assert result["agent_summary_grade"]["present"] is False


def test_grade_is_present_on_every_result_shape():
    """Callers must never have to branch on key presence."""
    case = load_case("case-001")
    s = GameSession(case, requested_mode="rule_based")
    s.run()
    assert "agent_summary_grade" in evaluate(case, s.state)


# --- extended thinking ---------------------------------------------------------
def test_thinking_is_requested_by_default():
    case = load_case("case-001")
    client = FakeClient([FakeResponse([ToolUseBlock("read_incident_report", {})])])
    inv = AgentInvestigator(case, client=client)
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    s.step()
    sent = client.messages.calls[0]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    # Room to actually reason, and more than the thinking budget.
    assert sent["max_tokens"] > sent["thinking"]["budget_tokens"]


def test_thinking_can_be_disabled():
    case = load_case("case-001")
    client = FakeClient([FakeResponse([ToolUseBlock("read_incident_report", {})])])
    inv = AgentInvestigator(case, client=client, thinking={})
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    s.step()
    assert "thinking" not in client.messages.calls[0]


@pytest.mark.parametrize("max_tokens,budget", [(1000, 2000), (2000, 2000), (4000, 0)])
def test_unsatisfiable_thinking_config_disables_rather_than_fails(max_tokens, budget):
    """The API requires max_tokens > budget_tokens; a bad pair must not kill the run."""
    assert thinking_config(max_tokens, budget) is None


def test_thinking_blocks_reach_the_trace():
    """The deliberation is the product here — it has to be readable, not implied."""
    case = load_case("case-001")
    script = [
        FakeResponse([ThinkingBlock("Access logs first: they are hardest to fake."),
                      TextBlock("Reading the report."),
                      ToolUseBlock("read_incident_report", {})]),
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved", "reasoning": "thin",
                                    "summary": "Nothing established."})]),
    ]
    events = list(iter_agent("case-001", client=FakeClient(script)))
    thoughts = [e["text"] for e in events if e["type"] == "think"]
    assert "Access logs first: they are hardest to fake." in thoughts
    # The plain-text remark is kept too, and the thinking precedes it.
    assert "Reading the report." in thoughts
    assert thoughts.index("Access logs first: they are hardest to fake.") < \
        thoughts.index("Reading the report.")


def test_reasoning_dump_does_not_become_the_recorded_question():
    """NextAction.question records the question the agent asked itself; a whole
    reasoning trace is not that, so thinking must stay out of it."""
    case = load_case("case-001")
    script = [
        FakeResponse([ThinkingBlock("A long deliberation " * 20),
                      TextBlock("Who had access?"),
                      ToolUseBlock("read_incident_report", {})]),
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved", "reasoning": "thin"})]),
    ]
    _, s, _ = _run(script)
    first = s.audit[0]
    assert first.decision == "Who had access?"
