"""Human-as-actor character roles and knowledge.

When a character is marked human_played=True, the agent interviews a real person who
plays that character. The human knows their role upfront:
  - What they witnessed (from 'knowledge')
  - Secrets/lies they might tell (from 'secrets')
  - Motivations to lie or stay quiet (from 'motivation')

The human then decides: tell the truth, lie, or stay silent about certain facts.
The agent must interview skillfully to detect contradictions and lies.

This module defines character archetypes and validates role schemas.
"""

from typing import Any, Optional

# Character role templates for common investigation scenarios
CHARACTER_ARCHETYPES = {
    "butler": {
        "description": "Service staff who was present",
        "typical_knowledge": ["saw_people_enter_exit", "heard_conversations", "noticed_unusual_items"],
        "typical_secrets": ["bribed_to_lie", "saw_something_incriminating", "protected_someone"],
        "typical_motivation": "paid_to_stay_quiet or fears_employer",
    },
    "neighbor": {
        "description": "Nearby resident who may have witnessed activity",
        "typical_knowledge": ["saw_arrivals", "heard_disturbances", "noticed_time"],
        "typical_secrets": ["incomplete_information", "mistaken_identity", "afraid_to_testify"],
        "typical_motivation": "fear_of_retaliation or social_pressure",
    },
    "family_member": {
        "description": "Relative with protective motivations",
        "typical_knowledge": ["routine_information", "relationship_dynamics", "financial_motives"],
        "typical_secrets": ["protecting_accused", "hiding_motive", "covering_up_past"],
        "typical_motivation": "loyalty or self_preservation",
    },
    "colleague": {
        "description": "Professional associate or coworker",
        "typical_knowledge": ["workplace_dynamics", "access_patterns", "behavioral_changes"],
        "typical_secrets": ["rivalry", "business_conflicts", "hidden_relationships"],
        "typical_motivation": "career_advancement or avoiding_blame",
    },
    "acquaintance": {
        "description": "Casual contact with partial information",
        "typical_knowledge": ["casual_observations", "hearsay", "incomplete_timeline"],
        "typical_secrets": ["unreliable_memory", "self_interest", "exaggeration"],
        "typical_motivation": "attention_seeking or self_protection",
    },
}


def validate_character_schema(char: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate a character definition for human-play.

    Returns (is_valid, error_message).
    """
    required_fields = ["id", "name"]
    for field in required_fields:
        if field not in char:
            return False, f"Missing required field: {field}"

    if char.get("human_played"):
        # If marked for human play, must have knowledge, secrets, motivation
        if not char.get("knowledge"):
            return False, "human_played characters must have 'knowledge' dict"
        if not char.get("secrets"):
            return False, "human_played characters must have 'secrets' dict"
        if not char.get("motivation"):
            return False, "human_played characters must have 'motivation' field"

    return True, None


def build_character_role_card(char: dict[str, Any]) -> str:
    """Generate a human-readable role card for a human player.

    This is what the human sees when they start playing this character.
    """
    if not char.get("human_played"):
        return ""

    lines = [
        f"ROLE: You are {char.get('name')} — {char.get('role', 'unknown role')}",
        "",
        "WHAT YOU KNOW:",
    ]

    for key, value in char.get("knowledge", {}).items():
        lines.append(f"  • {key}: {value}")

    lines.extend([
        "",
        "YOUR SECRETS (decide when/if to reveal):",
    ])

    for key, value in char.get("secrets", {}).items():
        lines.append(f"  • {key}: {value}")

    lines.extend([
        "",
        f"YOUR MOTIVATION: {char.get('motivation')}",
        "",
        "REMEMBER: You can tell the truth, lie, or stay silent. The detective must",
        "ask good follow-up questions to catch inconsistencies.",
    ])

    return "\n".join(lines)


# Example character definitions for a theft scenario
EXAMPLE_CHARACTERS = {
    "butler_example": {
        "id": "edmund",
        "name": "Edmund",
        "role": "butler",
        "human_played": True,
        "knowledge": {
            "saw_alice_leave": "You saw Alice leave the study at 11:30pm carrying a bag",
            "saw_victim_with_wine": "You saw the victim with a wine glass at 11:15pm",
            "heard_argument": "You heard raised voices from the study around 11:00pm",
        },
        "secrets": {
            "bribed_by_alice": True,
            "lie_about_time": "Alice asked you to say she left at 10:30pm, not 11:30pm",
            "saw_poison_bottle": "You saw a small bottle on Alice's nightstand labeled 'arsenic'",
        },
        "motivation": "Paid 500 pounds by Alice to provide a false alibi; fears her if you don't cooperate",
    },
    "neighbor_example": {
        "id": "margaret",
        "name": "Margaret",
        "role": "neighbor",
        "human_played": True,
        "knowledge": {
            "heard_scream": "You heard a woman's scream from the next house around 11:45pm",
            "saw_car_leave": "You saw a car leave the driveway at 11:50pm (couldn't see driver clearly)",
            "heard_sirens": "Police sirens arrived about 15 minutes later",
        },
        "secrets": {
            "bad_eyesight": "You weren't wearing glasses, so car description is unreliable",
            "mistook_time": "You're not entirely sure of the time; could have been 11:30 instead of 11:45",
            "didnt_see_driver": "You assumed it was Alice because her car is red, but didn't actually see her",
        },
        "motivation": "Wants to be helpful but also worries about implicating an innocent person; tends to exaggerate drama",
    },
}


if __name__ == "__main__":
    # Example usage
    print("Character Archetypes:")
    for archetype, info in CHARACTER_ARCHETYPES.items():
        print(f"\n{archetype.upper()}: {info['description']}")
        print(f"  Typical knowledge: {info['typical_knowledge']}")
        print(f"  Typical secrets: {info['typical_secrets']}")

    print("\n" + "=" * 60)
    print("Example Character Role Card:")
    print("=" * 60)
    print(build_character_role_card(EXAMPLE_CHARACTERS["butler_example"]))
