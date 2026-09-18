"""Interview mechanics for human-played characters.

When the agent interviews a human-played character, the human provides a response.
This module checks if the response is consistent with the character's knowledge and
secrets, detecting lies and contradictions.

The agent gets back a tool result that includes:
  - observation: What was said (human's response)
  - consistency_note: Whether response matches known facts
  - contradiction_detected: Whether response contradicts previous statements
  - evidence: Standard evidence items (always empty for human interviews)

The agent must notice patterns: if a human said X at time T1 but Y at time T2,
that's a contradiction the agent can use in cross-examination.
"""

from typing import Any, Optional


class InterviewState:
    """Tracks interview history with a human-played character to detect contradictions."""

    def __init__(self):
        self.statements: dict[str, list[str]] = {}  # topic -> list of statements over time

    def add_statement(self, topic: str, statement: str) -> None:
        """Record a statement made by the character."""
        if topic not in self.statements:
            self.statements[topic] = []
        self.statements[topic].append(statement)

    def find_contradictions(self, topic: str, new_statement: str) -> list[str]:
        """Check if a new statement contradicts previous statements on the same topic.

        Returns list of contradictory previous statements.
        """
        if topic not in self.statements:
            return []

        contradictions = []
        prev_statements = self.statements[topic]

        # Use the general contradiction detector
        for prev in prev_statements:
            if _statements_contradict(prev, new_statement):
                contradictions.append(prev)

        return contradictions


BLANKET_DENIALS = (
    "haven't seen anything", "havent seen anything", "didn't see anything",
    "didnt see anything", "saw nothing", "don't know anything", "dont know anything",
    "know nothing", "wasn't there", "wasnt there", "nothing to report",
    "i was sleeping", "i was asleep", "fell asleep", "no comment",
)


def _relevant_knowledge(question: str, knowledge: dict[str, str]) -> list[tuple[str, str]]:
    """Knowledge items this question plausibly reaches.

    The topic string cannot be relied on to name a knowledge key: it is authored by
    the agent, which has no legitimate way to learn the engine's key names. So
    relevance is judged from the question text and the key words, and a question
    broad enough to cover the whole period reaches everything the character knows.
    """
    q = question.lower()
    stop = {"the","and","you","did","was","were","what","when","where","who","how",
            "your","that","this","with","for","are","any","see","saw","about","between"}
    q_words = {w.strip("?,.'\"") for w in q.split() if len(w) > 3} - stop

    # A sweeping question ("what did you observe between 03:00 and 03:40") reaches
    # everything the character witnessed, so it is checked BEFORE per-item matching:
    # narrowing it to the one item whose wording happens to overlap would understate
    # how much a flat denial is actually denying.
    if any(w in q for w in ("observe", "notice", "happen", "anything", "unusual",
                            "everything", "night", "period", "window",
                            "what did you see", "did you see anything",
                            "tell me what", "walk me through")):
        return list(knowledge.items())

    hits = []
    for key, val in knowledge.items():
        key_words = {w for w in key.lower().split("_") if len(w) > 3}
        val_words = {w.strip(",.'\"") for w in val.lower().split() if len(w) > 3} - stop
        if key_words & q_words or len(val_words & q_words) >= 2:
            hits.append((key, val))
    return hits


def check_statement_consistency(
    statement: str,
    character: dict[str, Any],
    topic: str,
    previous_statements: Optional[dict[str, list[str]]] = None,
    question: str = "",
) -> dict[str, Any]:
    """Check if a human character's statement is consistent with their knowledge.

    Returns analysis of consistency, contradiction detection, and coaching notes.
    """
    knowledge = character.get("knowledge", {})
    secrets = character.get("secrets", {})

    # Check if statement aligns with character's knowledge
    consistency_score = 0.5  # neutral
    consistency_notes = []
    said = statement.lower()

    # Exact topic match is the precise case, but it is the rare one: keep it first,
    # then fall back to matching on the question and on blanket denials.
    if topic in knowledge:
        known_fact = knowledge[topic]
        if _statement_aligns_with_knowledge(statement, known_fact):
            consistency_score = 0.9  # truthful
            consistency_notes.append(f"Statement aligns with knowledge of: {topic}")
        else:
            consistency_score = 0.3  # suspicious
            consistency_notes.append(f"Statement diverges from known fact about: {topic}")
    else:
        relevant = _relevant_knowledge(question or topic.replace("_", " "), knowledge)
        denied = any(d in said for d in BLANKET_DENIALS)
        if relevant and denied:
            # Claiming to know nothing about something they demonstrably witnessed.
            consistency_score = 0.15
            consistency_notes.append(
                "Denies knowledge of " + ", ".join(k for k, _ in relevant[:3])
                + " — all of which this character witnessed."
            )
        elif relevant and any(_statement_aligns_with_knowledge(statement, v)
                              for _, v in relevant):
            consistency_score = 0.9
            consistency_notes.append(
                "Statement matches what this character witnessed."
            )
        elif relevant:
            consistency_score = 0.4
            consistency_notes.append(
                "Neither confirms nor matches the "
                f"{len(relevant)} relevant thing(s) this character witnessed."
            )
        elif any(w in said for w in ["not sure", "remember", "recall",
                                     "quite late", "don't recall"]):
            consistency_score = 0.55  # evasive about something they do not know

    # Contradictions are checked against EVERY prior answer from this character, not
    # only those filed under the same topic. The topic is the agent's own label and it
    # naturally invents a fresh one per question, so a same-topic-only comparison never
    # fires — a story can shift freely as long as each answer is filed separately.
    contradictions = []
    if previous_statements:
        for prior_topic, prior in previous_statements.items():
            for prev_statement in prior:
                if _statements_contradict(prev_statement, statement):
                    contradictions.append(prev_statement)
                    consistency_score *= 0.5  # lower score for contradictions

    # Secret-based heuristic: if character is lying about something, there's motivation
    lying_about_secret = False
    for secret_key, secret_value in secrets.items():
        if secret_key.lower() in topic.lower():
            lying_about_secret = True
            consistency_notes.append(
                f"⚠️ This topic ({topic}) relates to a secret the character has."
            )

    return {
        "consistency_score": round(consistency_score, 2),  # 0.0–1.0
        "is_truthful_estimate": consistency_score >= 0.7,
        "likely_lying": consistency_score < 0.4,
        "contradictions": contradictions,
        "consistency_notes": consistency_notes,
        "possibly_hiding_secret": lying_about_secret,
    }


# Which parts of an analysis may cross back to the agent.
#
# The split is a hidden-truth boundary, not a formatting choice. Contradictions
# are derived from the character's OWN prior statements — the agent heard those,
# so noticing they conflict is honest detective work. Everything else is derived
# from `character["knowledge"]`, which is ground truth the agent has no legitimate
# access to; handing it `likely_lying` would be the engine solving the case and
# calling it deduction.
AGENT_VISIBLE_FIELDS = ("contradictions", "contradiction_count")
ENGINE_ONLY_FIELDS = (
    "consistency_score",
    "is_truthful_estimate",
    "likely_lying",
    "possibly_hiding_secret",
    "consistency_notes",
)


def split_analysis(analysis: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a consistency analysis into (agent_visible, engine_only).

    The engine and evaluator get the full picture. The agent gets only what it
    could have worked out from the interview transcript.
    """
    contradictions = list(analysis.get("contradictions", []))
    agent_visible = {
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
    }
    engine_only = {k: analysis[k] for k in ENGINE_ONLY_FIELDS if k in analysis}
    return agent_visible, engine_only


def _statement_aligns_with_knowledge(statement: str, known_fact: str) -> bool:
    """Check if a statement aligns with a known fact using flexible matching.

    Rather than exact substring matching, extracts key entities and compares.
    """
    statement_lower = statement.lower()
    known_lower = known_fact.lower()

    # Exact match (after lowercasing)
    if known_lower in statement_lower or statement_lower in known_lower:
        return True

    # Extract key entities (words except common pronouns and articles)
    stop_words = {"i", "you", "the", "a", "an", "was", "were", "is", "are", "at", "in"}
    statement_words = set(w for w in statement_lower.split() if len(w) > 2 and w not in stop_words)
    known_words = set(w for w in known_lower.split() if len(w) > 2 and w not in stop_words)

    # If most key words match, consider it aligned
    if known_words and statement_words:
        overlap = len(statement_words & known_words)
        required = max(3, len(known_words) // 2)  # Require at least 50% match or 3 words
        if overlap >= required:
            return True

    return False


def _statements_contradict(statement_a: str, statement_b: str) -> bool:
    """Simple heuristic to detect contradictions between two statements.

    Real implementation would use semantic similarity or NLP.
    """
    a_lower = statement_a.lower()
    b_lower = statement_b.lower()

    # Direct negation patterns
    # Check if one is a negation of the other
    a_without_negation = a_lower.replace("not ", "").replace("didn't ", "").replace("wasn't ", "")
    b_without_negation = b_lower.replace("not ", "").replace("didn't ", "").replace("wasn't ", "")

    # If the core content is the same but one is negated, they contradict
    if a_without_negation == b_without_negation:
        if ("not" in a_lower or "didn't" in a_lower or "wasn't" in a_lower) != \
           ("not" in b_lower or "didn't" in b_lower or "wasn't" in b_lower):
            return True

    # Check for "in the X" patterns with negation
    if ("in the" in a_lower and "not in the" in b_lower) or \
       ("not in the" in a_lower and "in the" in b_lower):
        # Extract location and check if same
        loc_a = a_lower.split("in the")[-1].split(" ")[0] if "in the" in a_lower else ""
        loc_b = b_lower.split("in the")[-1].split(" ")[0] if "in the" in b_lower else ""
        if loc_a and loc_b and loc_a == loc_b:
            return True

    # Check for "stayed in X" vs "left X" contradiction
    if ("all night" in a_lower or "all evening" in a_lower) and "left" in b_lower:
        # "stayed in kitchen all night" vs "left the kitchen" = contradiction
        return True
    if ("all night" in b_lower or "all evening" in b_lower) and "left" in a_lower:
        return True

    # Time contradictions, compared by meaning rather than by matching characters:
    # '11:30' and 'half past eleven' are the same claim, 'quarter past three' and
    # 'after four' are not, and neither pair shares a substring.
    if _time_claims_conflict(parse_time_claims(a_lower), parse_time_claims(b_lower)):
        return True

    # Bare before/after with no parseable hour ('before midnight' vs 'after midnight').
    if ("before" in a_lower and "after" in b_lower) or \
       ("after" in a_lower and "before" in b_lower):
        return True

    return False


def extract_time_markers(text: str) -> list[str]:
    """Extract raw time references from text (11:30, before, after, etc.).

    Kept for callers that just want to know whether a statement mentions time at
    all. Comparisons should use `parse_time_claims`, which understands what the
    words actually mean.
    """
    markers = []
    import re

    # Time pattern HH:MM
    times = re.findall(r"\d{1,2}:\d{2}", text)
    markers.extend(times)

    # Before/after
    if "before" in text:
        markers.append("before")
    if "after" in text:
        markers.append("after")

    return markers


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "midnight": 12, "noon": 12,
}

# A claim is (bound, minutes) where bound is "at" | "before" | "after".
# Minutes are measured on a 12-hour dial, because people drop am/pm constantly
# and within one interview the half of the day is rarely what's in dispute.
_TimeClaim = tuple[str, int]


def parse_time_claims(text: str) -> list[_TimeClaim]:
    """Read the time claims out of a sentence.

    Handles the forms people actually say: '11:30pm', 'half past ten',
    'quarter past three', 'after four', 'before eleven', 'three o'clock'.
    """
    import re

    t = text.lower()
    claims: list[_TimeClaim] = []

    def dial(hour: int, minute: int = 0) -> int:
        return (hour % 12) * 60 + minute

    def bound_before(idx: int) -> str:
        """Which qualifier governs the time at position `idx`."""
        window = t[max(0, idx - 24):idx]
        if "before" in window or "earlier than" in window or "by " in window:
            return "before"
        if "after" in window or "later than" in window or "past " in window and "quarter" not in window \
                and "half" not in window:
            return "after"
        return "at"

    # 11:30, 11:30pm
    for m in re.finditer(r"(\d{1,2}):(\d{2})\s*(am|pm)?", t):
        hour, minute = int(m.group(1)), int(m.group(2))
        claims.append((bound_before(m.start()), dial(hour, minute)))

    # quarter/half past ten, quarter to eleven
    for m in re.finditer(r"(quarter|half)\s+(past|to)\s+(\w+)", t):
        frac, direction, word = m.group(1), m.group(2), m.group(3)
        hour = _WORD_NUMBERS.get(word) or (int(word) if word.isdigit() else None)
        if hour is None:
            continue
        offset = 15 if frac == "quarter" else 30
        claims.append(("at", dial(hour, offset) if direction == "past"
                       else dial(hour - 1, 60 - offset)))

    # bare hours: 'after four', 'before eleven', 'three o'clock', '4pm'
    for m in re.finditer(r"\b(\w+)\s*(?:o'clock)?\s*(am|pm)?\b", t):
        word = m.group(1)
        hour = _WORD_NUMBERS.get(word)
        if hour is None and word.isdigit() and len(word) <= 2:
            hour = int(word)
        if hour is None or not (1 <= hour <= 12):
            continue
        # Skip if this hour was already captured as part of an HH:MM or a
        # quarter/half phrase.
        if any(abs(c[1] - dial(hour)) < 60 and c[0] == "at" for c in claims):
            continue
        claims.append((bound_before(m.start()), dial(hour)))

    return claims


def _time_claims_conflict(a: list[_TimeClaim], b: list[_TimeClaim]) -> bool:
    """True when no single moment could satisfy both sets of claims."""
    if not a or not b:
        return False

    # Allow a little slack so 'around eleven' and 'at 11:05' are not a gotcha.
    SLACK = 10

    for bound_a, min_a in a:
        for bound_b, min_b in b:
            if bound_a == "at" and bound_b == "at":
                if abs(min_a - min_b) > SLACK:
                    return True
            elif bound_a == "at" and bound_b == "after":
                if min_a < min_b - SLACK:
                    return True
            elif bound_a == "at" and bound_b == "before":
                if min_a > min_b + SLACK:
                    return True
            elif bound_a == "after" and bound_b == "at":
                if min_b < min_a - SLACK:
                    return True
            elif bound_a == "before" and bound_b == "at":
                if min_b > min_a + SLACK:
                    return True
            elif bound_a == "before" and bound_b == "after":
                if min_b >= min_a:
                    return True
            elif bound_a == "after" and bound_b == "before":
                if min_a >= min_b:
                    return True
    return False


def simulate_human_response(
    character: dict[str, Any],
    question: str,
    strategy: str = "truthful",
) -> str:
    """Simulate a human character's response to a question.

    strategy: "truthful" | "lying" | "evasive"
    (Used for testing; in real gameplay, human types the response)
    """
    knowledge = character.get("knowledge", {})
    question_lower = question.lower()

    # Try to find matching knowledge by topic
    matching_topics = []
    for key, value in knowledge.items():
        if key.lower() in question_lower or \
           any(word in key.lower() for word in question_lower.split()):
            matching_topics.append((key, value))

    if strategy == "truthful":
        # Return something from knowledge
        if matching_topics:
            key, value = matching_topics[0]
            return f"Yes, {value}"
        return "I don't recall anything about that."

    elif strategy == "lying":
        # Return something false
        if matching_topics:
            key, value = matching_topics[0]
            # Negate or reverse the fact
            if "saw" in key.lower() or "saw" in question_lower:
                return "No, I didn't see that."
            elif "heard" in key.lower() or "heard" in question_lower:
                return "No, I didn't hear anything."
            else:
                return "No, that's not right. The opposite happened."
        return "I wasn't there."

    elif strategy == "evasive":
        return "I'm not sure I remember that clearly. It was quite late."

    return "I don't know."


if __name__ == "__main__":
    # Example usage
    from human_characters import EXAMPLE_CHARACTERS

    butler = EXAMPLE_CHARACTERS["butler_example"]
    print(f"Interviewing: {butler['name']}")
    print("=" * 60)

    # Test truthful answer
    statement = butler["knowledge"]["saw_alice_leave"]
    analysis = check_statement_consistency(statement, butler, "saw_alice_leave")
    print(f"\nTruthful statement: '{statement}'")
    print(f"Analysis: {analysis}")

    # Test lying answer
    lying_statement = "No, I didn't see Alice leave. She was in her room all night."
    analysis = check_statement_consistency(lying_statement, butler, "saw_alice_leave")
    print(f"\nLying statement: '{lying_statement}'")
    print(f"Analysis: {analysis}")
