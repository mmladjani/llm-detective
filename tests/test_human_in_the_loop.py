"""The session parks on a human question and resumes on the answer.

Tool-level tests live in test_interview_human_tool.py. These cover the loop:
that a run genuinely suspends, that suspending costs no budget, and that
resuming executes the action exactly once.
"""

import pytest

from src.human_responder import CLIResponder, PendingResponder, ScriptedResponder
from src.investigator import Decision, Investigator, NextAction
from src.loop import GameSession, InvestigationComplete
from tests.test_interview_human_tool import make_case


class ScriptedInvestigator(Investigator):
    """Plays a fixed list of actions, so the loop is what's under test."""

    name = "scripted"

    def __init__(self, actions):
        self._actions = list(actions)
        self.calls = 0

    def decide(self, state) -> Decision:
        self.calls += 1
        if not self._actions:
            return Decision(action=None, source="rule_based", validation_status="ok",
                            should_stop=True, stop_reason="script finished")
        tool, args = self._actions.pop(0)
        return Decision(action=NextAction("unspecified", "q", tool, args),
                        source="rule_based", validation_status="ok")


REPORT = ("read_incident_report", {})


def ask(suspect="butler", question="Where were you?", topic="alibi"):
    return ("interview_human", {"suspect": suspect, "question": question, "topic": topic})


def started(*actions, responder):
    """A session that has already read the report, so the board (and who is
    human-played) is known — interviews are gated on that, as they should be."""
    session = GameSession(make_case(),
                          investigator=ScriptedInvestigator([REPORT, *actions]),
                          responder=responder)
    session.step()  # read_incident_report
    return session


def test_session_parks_and_charges_no_budget():
    session = started(ask(), responder=PendingResponder())
    budget_before = session.state.remaining_budget
    audit_before = len(session.audit)

    entry = session.step()

    assert entry is None, "parking is not an audited action"
    assert session.awaiting_human
    assert session.state.status == "awaiting_human"
    assert session.state.remaining_budget == budget_before, "parking must be free"
    assert len(session.audit) == audit_before, "nothing happened in the world yet"


def test_pending_question_carries_the_role_card():
    session = started(ask(), responder=PendingResponder())
    session.step()

    pending = session.pending_question()
    assert pending["suspect"] == "butler"
    assert pending["character_name"] == "Edmund"
    assert pending["question"] == "Where were you?"
    assert "WHAT YOU KNOW:" in pending["role_card"]
    assert "YOUR SECRETS" in pending["role_card"]


def test_resume_executes_the_action_once():
    session = started(ask(), responder=PendingResponder())
    budget_before = session.state.remaining_budget
    session.step()

    entry = session.provide_human_answer("I was in the pantry.")

    assert not session.awaiting_human
    assert session.state.pending_human is None
    assert entry.tool == "interview_human"
    assert "I was in the pantry." in entry.observation
    assert session.state.remaining_budget == budget_before - 1, "charged exactly once"
    assert len([a for a in session.audit if a.tool == "interview_human"]) == 1


def test_stepping_while_parked_is_refused():
    session = started(ask(), responder=PendingResponder())
    session.step()

    with pytest.raises(InvestigationComplete, match="awaiting a human answer"):
        session.step()


def test_answering_when_not_parked_is_refused():
    session = started(ask(), responder=PendingResponder())

    with pytest.raises(InvestigationComplete):
        session.provide_human_answer("nobody asked me")


def test_run_halts_at_the_pause_instead_of_looping():
    """run() must not spin forever against a question it cannot answer."""
    session = started(ask(), ask(question="And then?"), responder=PendingResponder())
    calls_before = session.investigator.calls

    session.run()

    assert session.awaiting_human
    assert session.investigator.calls == calls_before + 1, "run stopped at the pause"


def test_two_questions_park_separately():
    session = started(
        ask(question="Where were you?", topic="movements"),
        ask(question="You're sure?", topic="movements"),
        responder=PendingResponder(),
    )

    session.step()
    session.provide_human_answer("I left at 11:30pm")
    session.step()
    assert session.pending_question()["question"] == "You're sure?"
    entry = session.provide_human_answer("Actually, I left before 11:00pm")

    # The agent is told the story shifted — but not that it was a lie.
    assert "conflicts with what" in entry.observation
    assert "likely_lying" not in repr(entry.observation)


def test_blocking_responder_never_parks():
    """With a CLI responder the run proceeds without any pause at all."""
    typed = iter(["In the pantry."])
    session = started(ask(), responder=CLIResponder(
        input_fn=lambda _: next(typed), output_fn=lambda _: None))

    entry = session.step()

    assert not session.awaiting_human
    assert "In the pantry." in entry.observation


def test_report_does_not_leak_the_characters_knowledge_or_secrets():
    """read_incident_report says WHO is human-played, never what they know."""
    session = started(responder=PendingResponder())

    report = session.audit[0]
    catalog = session.state.index
    blob = repr(report.observation) + repr(catalog)

    assert "You saw Alice leave the study" not in blob
    assert "bribed_by_alice" not in blob
    assert "false alibi" not in blob

    butler = next(c for c in catalog["characters"] if c["id"] == "butler")
    assert butler["human_played"] is True          # the agent may know this
    assert "knowledge" not in butler               # but not this
    assert "secrets" not in butler
    assert "motivation" not in butler


def test_engine_keeps_the_truthfulness_record_for_the_evaluator():
    session = started(
        ask(question="Did she leave?", topic="timing"),
        responder=ScriptedResponder(["She never left the study at all."]),
    )

    session.step()

    audit = session.toolbox.human_interview_audit
    assert len(audit) == 1
    assert audit[0]["answer"] == "She never left the study at all."
    assert audit[0]["likely_lying"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- regressions from the first real human-played run --------------------------

def test_denial_is_caught_even_when_the_topic_is_the_agents_own_label():
    """The agent invents its own topic string; scoring must not depend on it.

    In the first live run of case-008 the agent used topic="ward_4_observations"
    while the character's knowledge keys are saw_rhodes_enter / heard_the_lock /
    saw_the_holdall / corridor_was_empty. Nothing matched, so a flat denial from a
    player who had witnessed everything scored 0.5 / likely_lying=False.
    """
    from src.human_interview import check_statement_consistency

    pike = {
        "id": "pike", "name": "Pike", "human_played": True,
        "knowledge": {
            "saw_rhodes_enter": "You saw Rhodes badge into the dispensary corridor at about 03:15.",
            "corridor_was_empty": "Nobody else came down that corridor between 03:00 and 03:40.",
        },
        "secrets": {"your_log_is_false": "You wrote 'nothing to report'."},
        "motivation": "Frightened of losing your job.",
    }

    a = check_statement_consistency(
        "i haven't seen anything, i was sleeping", pike,
        topic="ward_4_observations",                       # agent's own label
        question="What did you observe on Ward 4 corridor between 03:00 and 03:40?",
    )

    assert a["likely_lying"] is True
    assert a["consistency_score"] < 0.4
    assert any("saw_rhodes_enter" in n for n in a["consistency_notes"])


def test_contradictions_span_different_agent_topics():
    """A story that shifts across two differently-labelled questions must be caught."""
    from src.human_interview import check_statement_consistency

    char = {"id": "x", "name": "X", "human_played": True,
            "knowledge": {"k": "something"}, "secrets": {"s": "y"}, "motivation": "m"}
    prior = {"first_question": ["I left at 11:30pm"]}

    a = check_statement_consistency(
        "Actually I left before 11:00pm", char,
        topic="a_completely_different_label",              # different topic on purpose
        previous_statements=prior,
        question="Are you sure about the time?",
    )

    assert a["contradictions"] == ["I left at 11:30pm"]


def test_interview_audit_is_persisted_with_the_labels():
    """The engine's honesty assessment must survive the run, for the evaluator."""
    from src.evaluator import evaluate
    from src.history import record_from_trace
    from src.human_responder import ScriptedResponder
    import tempfile, pathlib

    session = started(ask(question="Did you see Rhodes?", topic="anything"),
                      responder=ScriptedResponder(["I never saw a thing."]))
    session.step()
    while session.state.status == "investigating":
        try:
            session.step()
        except Exception:
            break

    with tempfile.TemporaryDirectory() as d:
        rec = record_from_trace(session.case, session,
                                evaluate(session.case, session.state),
                                seconds=0.1, runs_dir=pathlib.Path(d))

    audit = rec["labels"]["human_interviews"]
    assert audit, "the human-interview assessment was not persisted"
    assert audit[0]["answer"] == "I never saw a thing."
    assert "likely_lying" in audit[0]
