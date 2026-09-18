"""Tests for cross-examination question generation in judge layer."""

import pytest
from src.judge import check_conclusion, _generate_crossexam_questions
from src.state import InvestigationState, Hypothesis, EvidenceItem


def test_crossexam_missing_method():
    """Agent accuses a suspect but hasn't established METHOD."""
    state = InvestigationState(case_id="test", remaining_budget=10)

    # Seed with one hypothesis
    state.hypotheses.append(
        Hypothesis(id="H1", claim="alice is guilty", suspect="alice", confidence=0.85)
    )
    state.hypotheses[0].independent_types = ["alibi_check", "timeline"]

    # Establish MOTIVE and OPPORTUNITY but NOT METHOD
    state.facts_revealed["motive"] = "jealousy"
    state.facts_revealed["opportunity"] = "at_party"
    # NOTE: method is NOT in facts_revealed

    proposal = {
        "verdict": "solved",
        "culprit": "alice",
        "reasoning": "Alice had motive (jealousy) and was at the party. She's guilty."
    }

    result = check_conclusion(state, proposal)

    # Gate fails due to missing evidence
    assert result["ok"] is False

    # Cross-exam questions are generated (not generic "method not established" messages)
    questions = result.get("crossexam_questions", [])
    assert len(questions) > 0
    assert any("method" in q.lower() for q in questions)


def test_crossexam_missing_motive():
    """Agent accuses a suspect but hasn't established MOTIVE."""
    state = InvestigationState(case_id="test", remaining_budget=10)

    state.hypotheses.append(
        Hypothesis(id="H1", claim="bob is guilty", suspect="bob", confidence=0.80)
    )
    state.hypotheses[0].independent_types = ["dna", "witness"]

    # Establish METHOD and OPPORTUNITY but NOT MOTIVE
    state.facts_revealed["method"] = "poisoned"
    state.facts_revealed["opportunity"] = "had_access"

    proposal = {
        "verdict": "solved",
        "culprit": "bob",
        "reasoning": "Bob had access and poison was used. He must be guilty."
    }

    result = check_conclusion(state, proposal)

    assert result["ok"] is False
    questions = result.get("crossexam_questions", [])
    assert len(questions) > 0
    assert any("motive" in q.lower() for q in questions)


def test_crossexam_all_fields_present():
    """No cross-exam questions when all required fields are established."""
    state = InvestigationState(case_id="test", remaining_budget=10)

    state.hypotheses.append(
        Hypothesis(id="H1", claim="charlie is guilty", suspect="charlie", confidence=0.85)
    )
    state.hypotheses[0].independent_types = ["dna", "camera"]

    # All required fields present
    state.facts_revealed["method"] = "stabbed"
    state.facts_revealed["motive"] = "revenge"
    state.facts_revealed["opportunity"] = "at_scene"

    proposal = {
        "verdict": "solved",
        "culprit": "charlie",
        "reasoning": "DNA match, camera footage, clear motive. Charlie did it."
    }

    result = check_conclusion(state, proposal)

    # Gate should still reject if missing other criteria, but no METHOD question
    questions = result.get("crossexam_questions", [])
    # Empty list or no method-related questions
    assert not any("method" in q.lower() for q in questions if q)


def test_generate_crossexam_direct():
    """Test the cross-exam question generator directly."""
    state = InvestigationState(case_id="test", remaining_budget=10)
    state.facts_revealed["motive"] = "greed"
    # Missing method and opportunity

    proposal = {
        "verdict": "solved",
        "culprit": "diana",
        "reasoning": "She needed money."
    }

    questions = _generate_crossexam_questions(state, proposal)

    # Should have questions for missing method and opportunity
    assert len(questions) == 2  # method + opportunity
    assert any("method" in q.lower() for q in questions)
    assert any("opportunity" in q.lower() for q in questions)


def test_adversarial_judge_is_scoped_to_claims_the_record_does_not_contain():
    """The reviewer stays adversarial — but about groundedness, not epistemics.

    This used to assert the prompt enumerated named fallacies ("Post-hoc ergo propter
    hoc", "circular reasoning"). Measured on live case-011 runs, that list is exactly
    what the reviewer reached for: it returned "post-hoc reasoning flaw" and "circular
    reasoning" against conclusions whose only real defect was that the case cannot
    offer a confession, and rejected one run five times in a row. Worse, it also
    objected to a phrase the agent had quoted verbatim from the record.

    So the prompt now scopes it to the single thing it is uniquely able to check — the
    agent's prose against the evidence record — and forbids re-arguing the
    deterministic gate. Behaviour that can be exercised offline lives in
    test_reviewer_scope.py; this pins the instructions the live reviewer is given.
    """
    from src import judge as judge_mod

    prompt = judge_mod._LLM_JUDGE_SYSTEM.lower()

    assert "skeptical" in prompt
    # Never an oracle: it must decide without knowing the answer.
    assert "ground truth" in prompt and "not guess" in prompt
    # Scoped to groundedness: claims the record does not carry.
    assert "does not contain" in prompt or "does not support" in prompt
    # ...and explicitly out of the deterministic gate's business.
    assert "numeric confidence and leading rank do not decide acceptance" in prompt
    assert "original tool observations" in prompt


def test_confidence_mismatch_detection():
    """The mismatch check must compare the agent's stated confidence against the
    system's OWN number for the suspect it is accusing.

    It used to compare the scraped prose figure against a bespoke
    `0.3 + facts*0.1 + independent*0.1` formula that nothing else in the engine used —
    so a "mismatch" was one guess disagreeing with a second, unrelated guess. It now
    reads the accused hypothesis's `confidence`, which is the same quantity
    CONFIDENCE_THRESHOLD, LEAD_MARGIN and the final report all act on.
    """
    from src.judge import _extract_confidence_from_reasoning, _confidence_for_culprit

    state = InvestigationState(case_id="test", remaining_budget=10)
    state.hypotheses.append(
        Hypothesis(id="H1", claim="alice is guilty", suspect="alice", confidence=0.75)
    )
    state.hypotheses[0].independent_types = ["dna", "alibi_check"]

    # Establish all three facts
    state.facts_revealed["method"] = "poisoned"
    state.facts_revealed["motive"] = "revenge"
    state.facts_revealed["opportunity"] = "at_scene"

    # Agent says 95% confident
    reasoning = "Alice is definitely guilty. I'm 95% confident based on all the evidence."
    stated = _extract_confidence_from_reasoning(reasoning)
    assert stated == 0.95

    # The engine's own number for the accused: alice's hypothesis sits at 0.75.
    evidence = _confidence_for_culprit(state, "alice")
    assert evidence == 0.75
    assert evidence < stated  # 95% claimed, 75% earned

    # A name no hypothesis carries has nothing to compare against.
    assert _confidence_for_culprit(state, "nobody") is None

    # Mismatch should be flagged
    diff = abs(stated - evidence)
    assert diff > 0.0


def test_extract_confidence_levels():
    """Test extraction of different confidence levels from reasoning."""
    from src.judge import _extract_confidence_from_reasoning

    # Explicit percentages
    assert _extract_confidence_from_reasoning("I'm 75% confident") == 0.75
    assert _extract_confidence_from_reasoning("90% sure Alice did it") == 0.90

    # High confidence keywords
    assert _extract_confidence_from_reasoning("Absolutely, Alice is guilty") >= 0.8
    assert _extract_confidence_from_reasoning("Clearly Alice is the culprit") >= 0.8

    # Medium confidence keywords
    assert 0.5 <= _extract_confidence_from_reasoning("Probably Alice is guilty") <= 0.7

    # Low confidence keywords
    assert _extract_confidence_from_reasoning("Possibly Alice is guilty") < 0.6


def test_adversarial_judge_prompt_demands_the_blocking_advisory_split():
    """Only a claim the record does not carry may FAIL a conclusion.

    An "I would have put that more weakly" objection is advisory. Before the split
    both rejected equally hard, and the measured effect was an agent that learned to
    pass by softening its prose rather than by investigating further — the opposite of
    what this loop is meant to reward.
    """
    from src import judge as judge_mod

    prompt = judge_mod._LLM_JUDGE_SYSTEM.lower()

    assert "blocking" in prompt and "advisory" in prompt
    # The response shape _split_findings parses.
    assert "findings" in prompt
    # The reviewer must be told what NOT to flag, or it invents a standard of proof
    # the corpus can never satisfy and blocks on it.
    assert "not a finding" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
