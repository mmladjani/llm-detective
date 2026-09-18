"""Tests for human-as-actor character role definitions."""

import pytest
from src.human_characters import (
    validate_character_schema,
    build_character_role_card,
    EXAMPLE_CHARACTERS,
    CHARACTER_ARCHETYPES,
)


def test_validate_character_missing_required_fields():
    """Character missing required fields should fail validation."""
    char = {"name": "Alice"}  # missing 'id'
    is_valid, error = validate_character_schema(char)
    assert not is_valid
    assert "id" in error.lower()


def test_validate_human_played_missing_knowledge():
    """Human-played character must have knowledge field."""
    char = {
        "id": "alice",
        "name": "Alice",
        "human_played": True,
        "secrets": {"some_secret": "value"},
        "motivation": "greed",
    }
    is_valid, error = validate_character_schema(char)
    assert not is_valid
    assert "knowledge" in error.lower()


def test_validate_human_played_missing_secrets():
    """Human-played character must have secrets field."""
    char = {
        "id": "alice",
        "name": "Alice",
        "human_played": True,
        "knowledge": {"saw_something": "yes"},
        "motivation": "greed",
    }
    is_valid, error = validate_character_schema(char)
    assert not is_valid
    assert "secrets" in error.lower()


def test_validate_human_played_missing_motivation():
    """Human-played character must have motivation field."""
    char = {
        "id": "alice",
        "name": "Alice",
        "human_played": True,
        "knowledge": {"saw_something": "yes"},
        "secrets": {"secret": "value"},
    }
    is_valid, error = validate_character_schema(char)
    assert not is_valid
    assert "motivation" in error.lower()


def test_validate_valid_human_played_character():
    """Valid human-played character passes validation."""
    char = {
        "id": "alice",
        "name": "Alice",
        "role": "suspect",
        "human_played": True,
        "knowledge": {"saw_something": "yes"},
        "secrets": {"secret": "value"},
        "motivation": "self preservation",
    }
    is_valid, error = validate_character_schema(char)
    assert is_valid
    assert error is None


def test_validate_ai_character_not_human_played():
    """AI-only character (not human_played) has fewer requirements."""
    char = {
        "id": "bob",
        "name": "Bob",
        "role": "witness",
    }
    is_valid, error = validate_character_schema(char)
    assert is_valid
    assert error is None


def test_build_role_card_empty_for_ai_character():
    """AI-only character should not get a role card."""
    char = {
        "id": "bob",
        "name": "Bob",
        "role": "witness",
    }
    card = build_character_role_card(char)
    assert card == ""


def test_build_role_card_includes_all_sections():
    """Role card should include knowledge, secrets, and motivation."""
    card = build_character_role_card(EXAMPLE_CHARACTERS["butler_example"])

    assert "ROLE:" in card
    assert "Edmund" in card
    assert "WHAT YOU KNOW:" in card
    assert "YOUR SECRETS" in card  # Includes "(decide when/if to reveal)"
    assert "YOUR MOTIVATION:" in card
    assert "REMEMBER:" in card


def test_build_role_card_includes_knowledge_items():
    """Role card should list all knowledge items."""
    card = build_character_role_card(EXAMPLE_CHARACTERS["butler_example"])

    assert "saw_alice_leave" in card
    assert "11:30pm" in card
    assert "saw_victim_with_wine" in card


def test_build_role_card_includes_secrets():
    """Role card should list all secrets but not reveal them as truth."""
    card = build_character_role_card(EXAMPLE_CHARACTERS["butler_example"])

    assert "bribed_by_alice" in card
    assert "lie_about_time" in card
    assert "poison_bottle" in card.lower()


def test_example_characters_are_valid():
    """All example characters should pass validation."""
    for char_name, char_data in EXAMPLE_CHARACTERS.items():
        is_valid, error = validate_character_schema(char_data)
        assert is_valid, f"{char_name} failed validation: {error}"


def test_archetypes_documented():
    """All character archetypes should be documented."""
    required_archetypes = ["butler", "neighbor", "family_member", "colleague", "acquaintance"]
    for archetype in required_archetypes:
        assert archetype in CHARACTER_ARCHETYPES
        assert "description" in CHARACTER_ARCHETYPES[archetype]
        assert "typical_knowledge" in CHARACTER_ARCHETYPES[archetype]
        assert "typical_secrets" in CHARACTER_ARCHETYPES[archetype]
        assert "typical_motivation" in CHARACTER_ARCHETYPES[archetype]


def test_human_character_cannot_be_culprit_upfront():
    """A human-played character's role should not spoil the culprit identity.

    The human is told their character type, but not if they're the guilty party.
    """
    char = EXAMPLE_CHARACTERS["butler_example"]
    card = build_character_role_card(char)

    # Card should NOT mention guilt or innocence
    assert "culprit" not in card.lower()
    assert "guilty" not in card.lower()
    assert "innocent" not in card.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
