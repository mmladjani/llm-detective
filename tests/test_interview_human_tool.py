"""The agent asks; a person answers.

These tests exist because an earlier version of this tool took the answer as a
tool ARGUMENT, which let the model write both halves of the interview. The
first two tests below are the regression guard for that bug; the rest cover the
pause/resume handoff and the hidden-truth boundary.
"""

import pytest

from src.cases import Case
from src.human_interview import ENGINE_ONLY_FIELDS
from src.human_responder import (
    CLIResponder,
    HumanInputRequired,
    PendingResponder,
    ScriptedResponder,
)
from src.tools import ToolBox


def make_case():
    """Minimal case with one human-played character and one ordinary suspect."""
    return Case({
        "case_id": "test-human-001",
        "title": "Test Case with Human Player",
        "difficulty": "standard",
        "short_description": "Test.",
        "incident_report": "A crime occurred.",
        "budget": 10,
        "index": {
            "suspects": ["butler", "alice"],
            "location": "house",
            "objects": [],
            "characters": [
                {
                    "id": "butler",
                    "name": "Edmund",
                    "role": "butler",
                    "human_played": True,
                    "knowledge": {
                        "timing": "You saw Alice leave the study at 11:30pm",
                    },
                    "secrets": {"bribed_by_alice": "Alice paid you to say 10:30pm"},
                    "motivation": "Paid to provide a false alibi",
                },
                {"id": "alice", "name": "Alice", "role": "suspect"},
            ],
            "locations": [{"id": "house", "name": "House"}],
        },
        "hidden_truth": {"culprit": "alice"},
        "tools": {
            "read_incident_report": {"observation": "Crime occurred", "evidence": []},
            "interview_character": {
                "alice": {"observation": "Alice's account", "evidence": []},
            },
        },
    })


# --- regression guard: the agent must not be able to author the answer ----------

def test_agent_cannot_supply_the_answer():
    """A 'response' argument must be ignored, not used as the answer.

    This is the original bug: if the tool honours an agent-written response, the
    model interviews itself and every downstream check is theatre.
    """
    responder = ScriptedResponder(["I was in the pantry all evening."])
    box = ToolBox(make_case(), responder=responder)

    result = box.call("interview_human", {
        "suspect": "butler",
        "question": "Where were you?",
        "topic": "alibi",
        "response": "I confess, I did it!",   # agent tries to write the answer
    })

    assert result["ok"]
    assert result["interview"]["answer"] == "I was in the pantry all evening."
    assert "confess" not in result["observation"]


def test_question_is_required_and_answer_comes_from_responder():
    responder = ScriptedResponder(["Around eleven, I think."])
    box = ToolBox(make_case(), responder=responder)

    missing = box.call("interview_human", {"suspect": "butler", "topic": "alibi"})
    assert not missing["ok"]
    assert "question" in missing["error"].lower()

    ok = box.call("interview_human", {
        "suspect": "butler", "question": "What time did she leave?", "topic": "timing",
    })
    assert ok["ok"]
    assert ok["interview"]["answer"] == "Around eleven, I think."
    # The responder saw the agent's actual question.
    assert responder.asked == [("butler", "What time did she leave?")]


# --- hidden-truth boundary -----------------------------------------------------

def test_truthfulness_never_reaches_the_agent():
    """Lie detection is derived from hidden knowledge, so it stays engine-side."""
    # This answer flatly contradicts the butler's knowledge — engine-side it scores
    # as a lie. The agent must still not be handed that conclusion.
    responder = ScriptedResponder(["She never left the study at all."])
    box = ToolBox(make_case(), responder=responder)

    result = box.call("interview_human", {
        "suspect": "butler", "question": "Did she leave?", "topic": "timing",
    })

    flat = repr(result)
    for field in ENGINE_ONLY_FIELDS:
        assert field not in result
        assert field not in result.get("interview", {})
        assert field not in flat, f"{field} leaked into the agent-visible result"

    # The engine still recorded it, for the evaluator.
    assert len(box.human_interview_audit) == 1
    assert box.human_interview_audit[0]["likely_lying"] is True


def test_contradictions_do_reach_the_agent():
    """Conflicts with the character's OWN earlier answers are fair game."""
    responder = ScriptedResponder([
        "I left the kitchen at 11:30pm",
        "Actually, I left before 11:00pm",
    ])
    box = ToolBox(make_case(), responder=responder)

    box.call("interview_human", {
        "suspect": "butler", "question": "When did you leave?", "topic": "movements",
    })
    second = box.call("interview_human", {
        "suspect": "butler", "question": "You're sure about the time?", "topic": "movements",
    })

    assert second["interview"]["contradiction_count"] > 0
    assert "I left the kitchen at 11:30pm" in second["interview"]["contradictions"]
    assert "conflicts with what" in second["observation"]


# --- routing -------------------------------------------------------------------

def test_model_interview_has_no_precomputed_contradiction_or_truthfulness():
    case = make_case()
    case._data["inference"] = "model"
    box = ToolBox(case, responder=ScriptedResponder([
        "I left at 11:30pm", "Actually, I left before 11:00pm"]))
    args = {"suspect": "butler", "question": "When did you leave?", "topic": "timing"}
    first = box.call("interview_human", args)
    second = box.call("interview_human", args)
    assert first["interview"]["answer"] == "I left at 11:30pm"
    assert second["observation"] == 'Edmund answers: "Actually, I left before 11:00pm"'
    assert "contradictions" not in second["interview"]
    assert "contradiction_count" not in second["interview"]
    assert not (set(ENGINE_ONLY_FIELDS) & second["interview"].keys())
    assert len(box.human_interview_audit) == 2


def test_non_human_character_is_rejected():
    box = ToolBox(make_case(), responder=ScriptedResponder(["unused"]))

    result = box.call("interview_human", {
        "suspect": "alice", "question": "Did you do it?", "topic": "guilt",
    })

    assert not result["ok"]
    assert "not human-played" in result["error"].lower()
    assert "interview_character" in result["error"]


def test_unknown_character_is_rejected():
    box = ToolBox(make_case(), responder=ScriptedResponder(["unused"]))

    result = box.call("interview_human", {
        "suspect": "nobody", "question": "Hello?", "topic": "x",
    })

    assert not result["ok"]
    assert "not found" in result["error"].lower()


def test_human_played_suspects_listed():
    box = ToolBox(make_case())
    assert box.human_played_suspects() == ["butler"]


def test_testimony_is_not_evidence():
    """An interview yields a claim to corroborate, never a scored evidence item."""
    box = ToolBox(make_case(), responder=ScriptedResponder(["I saw everything."]))

    result = box.call("interview_human", {
        "suspect": "butler", "question": "What did you see?", "topic": "witness",
    })

    assert result["evidence"] == []


# --- responders ----------------------------------------------------------------

def test_pending_responder_raises_then_returns_after_submit():
    responder = PendingResponder()
    character = {"id": "butler", "name": "Edmund", "human_played": True,
                 "knowledge": {"a": "b"}, "secrets": {"c": "d"}, "motivation": "m"}

    with pytest.raises(HumanInputRequired) as excinfo:
        responder.ask("butler", "Where were you?", character)

    pending = excinfo.value
    assert pending.suspect == "butler"
    assert pending.question == "Where were you?"
    assert "Edmund" in pending.role_card

    responder.submit("butler", "Where were you?", "In the pantry.")
    assert responder.ask("butler", "Where were you?", character) == "In the pantry."


def test_pending_responder_rejects_empty_answer():
    responder = PendingResponder()
    with pytest.raises(ValueError):
        responder.submit("butler", "q", "   ")


def test_cli_responder_prints_role_card_once_and_reads_stdin():
    typed = iter(["In the pantry.", "About eleven."])
    printed: list[str] = []
    responder = CLIResponder(input_fn=lambda _: next(typed),
                             output_fn=lambda s: printed.append(str(s)))
    character = {"id": "butler", "name": "Edmund", "role": "butler",
                 "human_played": True, "knowledge": {"timing": "you saw her leave"},
                 "secrets": {"bribed": "paid off"}, "motivation": "money"}

    first = responder.ask("butler", "Where were you?", character)
    second = responder.ask("butler", "What time?", character)

    assert first == "In the pantry."
    assert second == "About eleven."
    assert sum(1 for line in printed if "ROLE:" in line) == 1


def test_scripted_responder_fails_loudly_when_exhausted():
    responder = ScriptedResponder(["only one"])
    character = {"id": "butler", "name": "Edmund", "human_played": True}

    responder.ask("butler", "q1", character)
    with pytest.raises(IndexError):
        responder.ask("butler", "q2", character)


def test_scripted_responder_routes_per_suspect():
    responder = ScriptedResponder({"butler": ["b1"], "maid": ["m1"]})
    character = {"id": "x", "name": "X", "human_played": True}

    assert responder.ask("maid", "q", character) == "m1"
    assert responder.ask("butler", "q", character) == "b1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
