"""How a human-played character actually answers the agent.

The agent asks a question. A REAL PERSON answers it. This module owns that
handoff, and it exists because the agent must never be able to author the
answer itself — if it could, the "interview" would just be the model talking
to itself and every consistency check downstream would be measuring nothing.

Three implementations of the same protocol:

  * ScriptedResponder  — answers come from a list. Tests only; keeps offline
                         runs hermetic and deterministic.
  * CLIResponder       — blocks on input(). The person sitting at the terminal
                         types the answer.
  * PendingResponder   — cannot block (a web request is in flight), so it raises
                         HumanInputRequired to suspend the run. The session
                         parks, the UI collects the answer, and the run resumes.

The pause protocol (PendingResponder) is two-phase on purpose. First call for a
given question raises. The session stores the pending question, the UI shows it
alongside the character's role card, the person types, and `submit()` buffers
the answer. The session then REPLAYS the same tool call — this time `ask()`
finds a buffered answer and returns it, so the tool executes exactly once from
the engine's point of view.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class HumanInputRequired(Exception):
    """Raised to suspend a run until a person answers.

    Carries everything the UI needs to render the prompt: who is being asked,
    what was asked, and the role card telling the player what their character
    knows and is hiding.
    """

    def __init__(self, suspect: str, question: str, role_card: str = "",
                 character_name: str = ""):
        self.suspect = suspect
        self.question = question
        self.role_card = role_card
        self.character_name = character_name
        super().__init__(f"Awaiting human answer from '{suspect}': {question}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "suspect": self.suspect,
            "character_name": self.character_name,
            "question": self.question,
            "role_card": self.role_card,
        }


class HumanResponder(Protocol):
    """Supplies an answer authored by a person, never by the agent."""

    def ask(self, suspect: str, question: str, character: dict[str, Any]) -> str:
        """Return the person's answer to `question`, asked of `suspect`.

        May raise HumanInputRequired to suspend the run instead of blocking.
        """
        ...


class ScriptedResponder:
    """Answers from a pre-written script. FOR TESTS ONLY.

    Accepts either a flat list (consumed in order) or a dict keyed by suspect
    id (each suspect gets its own ordered list). Running out of script is an
    error rather than a silent empty answer, so a test that under-specifies
    fails loudly instead of quietly asserting on "".
    """

    def __init__(self, answers: list[str] | dict[str, list[str]]):
        self._answers = answers
        self._flat_idx = 0
        self._per_suspect_idx: dict[str, int] = {}
        self.asked: list[tuple[str, str]] = []  # (suspect, question) audit for tests

    def ask(self, suspect: str, question: str, character: dict[str, Any]) -> str:
        self.asked.append((suspect, question))

        if isinstance(self._answers, dict):
            queue = self._answers.get(suspect)
            if not queue:
                raise IndexError(
                    f"ScriptedResponder has no answers for suspect '{suspect}'."
                )
            i = self._per_suspect_idx.get(suspect, 0)
            if i >= len(queue):
                raise IndexError(
                    f"ScriptedResponder exhausted for '{suspect}' "
                    f"(script had {len(queue)} answer(s))."
                )
            self._per_suspect_idx[suspect] = i + 1
            return queue[i]

        if self._flat_idx >= len(self._answers):
            raise IndexError(
                f"ScriptedResponder exhausted (script had "
                f"{len(self._answers)} answer(s))."
            )
        answer = self._answers[self._flat_idx]
        self._flat_idx += 1
        return answer


class CLIResponder:
    """Blocks on stdin. The person at the terminal plays the character.

    The role card prints once per character — re-printing it on every question
    would bury the actual interview in repeated boilerplate.
    """

    def __init__(self, input_fn=input, output_fn=print):
        self._input = input_fn
        self._output = output_fn
        self._briefed: set[str] = set()

    def ask(self, suspect: str, question: str, character: dict[str, Any]) -> str:
        from .human_characters import build_character_role_card

        if suspect not in self._briefed:
            self._briefed.add(suspect)
            card = build_character_role_card(character)
            if card:
                self._output("\n" + "=" * 62)
                self._output(card)
                self._output("=" * 62)

        name = character.get("name", suspect)
        self._output(f"\nDETECTIVE asks {name}: {question}")
        answer = self._input(f"{name} > ").strip()
        while not answer:
            answer = self._input(
                f"{name} > (say something — silence is not an option) "
            ).strip()
        return answer


class PendingResponder:
    """Suspends the run instead of blocking, for request/response frontends.

    Answers are buffered per (suspect, question) rather than in a flat queue.
    A flat queue would misroute if anything replayed out of order; keying on
    the exact question makes the handoff unambiguous.
    """

    def __init__(self):
        self._buffered: dict[tuple[str, str], str] = {}

    def ask(self, suspect: str, question: str, character: dict[str, Any]) -> str:
        key = (suspect, question)
        if key in self._buffered:
            # Second pass: the person has answered and the session is replaying.
            return self._buffered.pop(key)

        from .human_characters import build_character_role_card

        raise HumanInputRequired(
            suspect=suspect,
            question=question,
            role_card=build_character_role_card(character),
            character_name=character.get("name", suspect),
        )

    def submit(self, suspect: str, question: str, answer: str) -> None:
        """Buffer a person's answer so the replayed tool call can consume it."""
        if not answer or not answer.strip():
            raise ValueError("A human answer cannot be empty.")
        self._buffered[(suspect, question)] = answer.strip()


def default_responder() -> HumanResponder:
    """Responder used when a caller wires none.

    Deliberately a PendingResponder: pausing is the safe failure mode. A default
    that fabricated answers would resurrect exactly the bug this module exists
    to prevent.
    """
    return PendingResponder()
