"""Tests for human-as-actor interview mechanics and lie detection."""

import pytest
from src.human_interview import (
    InterviewState,
    check_statement_consistency,
    simulate_human_response,
    _statements_contradict,
    extract_time_markers,
    parse_time_claims,
)
from src.human_characters import EXAMPLE_CHARACTERS


def test_interview_state_tracks_statements():
    """InterviewState should track character statements over time."""
    state = InterviewState()

    state.add_statement("alibi", "I was in the kitchen")
    state.add_statement("alibi", "I was in the kitchen around 11pm")
    state.add_statement("motive", "I had no reason to harm them")

    assert len(state.statements["alibi"]) == 2
    assert len(state.statements["motive"]) == 1
    assert "I was in the kitchen" in state.statements["alibi"]


def test_find_contradictions_detects_negation():
    """Should detect when new statement negates previous statement."""
    state = InterviewState()

    state.add_statement("location", "I was in the kitchen")

    contradictions = state.find_contradictions("location", "I was not in the kitchen")
    assert len(contradictions) > 0


def test_find_contradictions_detects_time_shift():
    """Should detect when character changes claimed time from specific to before."""
    state = InterviewState()

    state.add_statement("time", "I left at 11:30pm")

    # This is a contradiction: said 11:30pm, now says before 11:00pm
    contradictions = state.find_contradictions("time", "Actually, I left before 11:00pm")
    assert len(contradictions) > 0


def test_check_consistency_truthful_response():
    """Truthful statement aligned with character knowledge should score high."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    # Use the actual knowledge statement
    statement = butler["knowledge"]["saw_alice_leave"]
    analysis = check_statement_consistency(statement, butler, "saw_alice_leave")

    assert analysis["consistency_score"] >= 0.7
    assert analysis["is_truthful_estimate"] is True
    assert analysis["likely_lying"] is False


def test_check_consistency_lying_response():
    """Lying statement contradicting knowledge should score low."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    # Directly contradict the known fact
    lying_statement = "No, I didn't see Alice leave the study. She stayed inside."
    analysis = check_statement_consistency(lying_statement, butler, "saw_alice_leave")

    assert analysis["consistency_score"] < 0.7
    assert analysis["is_truthful_estimate"] is False
    assert analysis["likely_lying"] is True


def test_check_consistency_evasive_response():
    """Evasive response about a known fact is suspicious (avoidance pattern)."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    # Being evasive about something you DO know is suspicious
    evasive_statement = "I'm not sure I remember clearly, it was quite late."
    analysis = check_statement_consistency(evasive_statement, butler, "saw_alice_leave")

    # Score should be low because they're avoiding a topic they clearly know about
    assert analysis["consistency_score"] < 0.5


def test_check_consistency_detects_secret_topic():
    """Should flag when statement relates to a character's secret."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    analysis = check_statement_consistency(
        "Yes, Alice gave me money",
        butler,
        "bribed_by_alice"
    )

    assert analysis["possibly_hiding_secret"] is True


def test_simulate_human_response_truthful():
    """Should generate truthful response matching character knowledge."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    response = simulate_human_response(butler, "Did you see Alice leave?", "truthful")

    assert "Alice leave" in response or "Yes" in response
    assert "didn't" not in response.lower()


def test_simulate_human_response_lying():
    """Should generate lying response contradicting knowledge."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    response = simulate_human_response(butler, "Did you see Alice leave?", "lying")

    assert "No" in response or "opposite" in response


def test_simulate_human_response_evasive():
    """Should generate evasive response."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    response = simulate_human_response(butler, "Did you see Alice leave?", "evasive")

    assert "not sure" in response.lower() or "remember" in response.lower()


def test_statements_contradict_explicit_negation():
    """Should detect explicit negation contradictions."""
    statement_a = "I was in the kitchen"
    statement_b = "I was not in the kitchen"

    assert _statements_contradict(statement_a, statement_b)


def test_statements_contradict_time_reversal():
    """Should detect when before/after are reversed."""
    statement_a = "I left before 11pm"
    statement_b = "I left after 11pm"

    assert _statements_contradict(statement_a, statement_b)


def test_statements_dont_contradict_when_similar():
    """Similar statements should not be flagged as contradicting."""
    statement_a = "I was in the kitchen around 11pm"
    statement_b = "I was in the kitchen at 11pm"

    # These are consistent, not contradictions
    assert not _statements_contradict(statement_a, statement_b)


@pytest.mark.parametrize("a,b", [
    # Same moment said two ways — must NOT be flagged.
    ("I left at half past ten", "I left at 10:30"),
    ("I saw him at 3:15", "I saw him at quarter past three"),
    ("I was in the kitchen around 11pm", "I was in the kitchen at 11pm"),
])
def test_equivalent_times_are_not_contradictions(a, b):
    assert not _statements_contradict(a, b)


@pytest.mark.parametrize("a,b", [
    # Genuinely incompatible — and none of these share a substring, so a
    # character-matching detector misses every one.
    ("About quarter past three.", "Actually it was after four."),
    ("I left at 11:30pm", "Actually, I left before 11:00pm"),
    ("I left before 11pm", "I left after 11pm"),
    ("It was half past two", "It was ten o'clock"),
])
def test_incompatible_times_are_contradictions(a, b):
    assert _statements_contradict(a, b)


def test_parse_time_claims_reads_spoken_forms():
    assert ("at", 10 * 60 + 30) in parse_time_claims("I left at half past ten")
    assert ("at", 3 * 60 + 15) in parse_time_claims("about quarter past three")
    assert ("after", 4 * 60) in parse_time_claims("it was after four")
    assert ("before", 11 * 60) in parse_time_claims("before eleven")


def test_extract_time_markers_finds_clock_times():
    """Should extract HH:MM time patterns."""
    text = "I left at 11:30pm"

    markers = extract_time_markers(text)

    assert "11:30" in markers


def test_extract_time_markers_finds_before_after():
    """Should extract before/after markers."""
    text = "I arrived before midnight and left after 1am"

    markers = extract_time_markers(text)

    assert "before" in markers
    assert "after" in markers


def test_consistency_with_previous_statements():
    """Should detect contradictions when previous statements are provided."""
    butler = EXAMPLE_CHARACTERS["butler_example"]

    previous = {
        "alibi": ["I was in the kitchen all evening", "I didn't leave the kitchen"]
    }

    # New statement contradicts previous
    new_statement = "I left the kitchen and went to the library"
    analysis = check_statement_consistency(
        new_statement,
        butler,
        "alibi",
        previous
    )

    # Score should be lower due to contradiction
    assert analysis["consistency_score"] < 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
