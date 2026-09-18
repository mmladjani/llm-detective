"""The tool layer.

World tools expose case observations after the public briefing. Each scripted tool:
  * validates its arguments,
  * looks up a deterministic response from the case definition,
  * returns structured JSON (observation + evidence),
  * records a cost of 1,
  * never returns the hidden truth, required-evidence list, or evaluator metadata.

Scripted observations are fixed for a given case and target. `interview_human` is
the exception: a person supplies each answer, and the session can pause to await it.
In model mode, answers carry no automatic truthfulness or contradiction labels.
"""

from __future__ import annotations

from typing import Any

from .cases import Case
from .state import EvidenceItem
from .human_interview import (
    InterviewState,
    check_statement_consistency,
    split_analysis,
)
from .human_responder import HumanResponder, default_responder

# Tools that take a single-location argument.
LOCATION_TOOLS = {"check_access_log", "review_camera", "inspect_location"}

TOOL_NAMES = [
    "read_incident_report",
    "interview_character",
    "interview_human",
    "inspect_location",
    "analyze_object",
    "check_access_log",
    "review_camera",
    "verify_alibi",
    "compare_testimonies",
]


class ToolResult(dict):
    """Plain dict result: {tool, arguments, ok, observation, evidence, error, cost}."""


class ToolBox:
    def __init__(self, case: Case, responder: HumanResponder | None = None):
        self._case = case
        self._tools = case.tools
        self._cat = case.catalog()
        # "authored" -> the case labels who each item implicates and how strongly.
        # "model"    -> that labelling is stripped here and the agent must supply it
        #               via assess_evidence. Enforced at the tool boundary rather than
        #               trusted to the case file, so a half-converted case cannot leak
        #               an authored attribution into the agent's context.
        self._inference_mode = case.inference_mode
        # Who answers for human-played characters. Never the agent.
        self.responder: HumanResponder = responder or default_responder()
        # Track interview state for human-played characters (suspect_id -> InterviewState)
        self._human_interview_states: dict[str, InterviewState] = {}
        # Engine-only record of how truthful each answer was. Kept OFF the tool
        # result so it cannot reach the agent; the evaluator reads it from here.
        self.human_interview_audit: list[dict[str, Any]] = []
        # FULL character records (knowledge, secrets, motivation). Engine-side: these
        # feed the responder's role card and the consistency check, never the agent —
        # `self._cat` holds the redacted view that goes out in tool results.
        self._case_suspects = {c.get("id", c.get("name", "").lower()): c
                               for c in case.characters_full()}

    def human_played_suspects(self) -> list[str]:
        return [sid for sid, c in self._case_suspects.items() if c.get("human_played")]

    # ---- capability surface (navigational, non-secret) ------------------------
    def valid_targets(self, tool: str) -> list[str]:
        if tool == "read_incident_report":
            return [""]
        if tool == "interview_human":
            return self.human_played_suspects()
        return list(self._cat["tool_targets"].get(tool, []))

    # ---- argument helpers -----------------------------------------------------
    def arg_key(self, tool: str, arguments: dict[str, Any]) -> str | None:
        """Public, stable key identifying a (tool, args) target — used to track
        which actions have already been attempted."""
        return self._arg_key(tool, arguments)

    @staticmethod
    def _arg_key(tool: str, arguments: dict[str, Any]) -> str | None:
        if tool == "read_incident_report":
            return "_report"
        if tool in LOCATION_TOOLS:
            return arguments.get("location")
        if tool == "analyze_object":
            return arguments.get("object")
        if tool in ("interview_character", "interview_human", "verify_alibi"):
            return arguments.get("suspect")
        if tool == "compare_testimonies":
            a = arguments.get("a")
            b = arguments.get("b")
            if a and b:
                return "|".join(sorted([a, b]))
            return arguments.get("pair")
        return None

    # ---- the call -------------------------------------------------------------
    def call(self, tool: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        arguments = arguments or {}
        base = ToolResult(tool=tool, arguments=arguments, ok=True, observation="",
                          evidence=[], error=None, cost=1)

        if tool not in TOOL_NAMES:
            base.update(ok=False, error=f"Unknown tool '{tool}'.")
            return base

        # read_incident_report is a special, argument-free tool that also returns
        # the navigational catalog.
        if tool == "read_incident_report":
            node = self._tools.get("read_incident_report", {})
            raw_ev = node.get("evidence", [])
            base["observation"] = node.get("observation", self._case.incident_report)
            base["evidence"] = self._materialize(raw_ev, tool)
            base["reveals"] = self._collect_reveals(raw_ev)
            base["raises"] = self._collect(raw_ev, "raises_question")
            base["resolves"] = self._collect(raw_ev, "resolves_question")
            base["catalog"] = self._cat
            return base

        # interview_human puts a real person on the other side of the question.
        # The agent supplies ONLY the question; the answer comes from the responder.
        # Note there is no 'response' argument and no way to add one — an agent that
        # could write the answer would be interviewing itself.
        if tool == "interview_human":
            suspect = str(arguments.get("suspect", "")).lower()
            question = str(arguments.get("question", "")).strip()
            topic = str(arguments.get("topic", "") or "interview").strip()

            if not suspect:
                base.update(ok=False, error="Missing argument: 'suspect'.")
                return base
            if not question:
                base.update(ok=False, error="Missing argument: 'question'.")
                return base

            character = self._case_suspects.get(suspect)
            if not character:
                base.update(ok=False, error=f"Character '{suspect}' not found in case.")
                return base
            if not character.get("human_played"):
                base.update(
                    ok=False,
                    error=(f"Character '{suspect}' is not human-played — "
                           f"use interview_character instead."),
                )
                return base

            # Ask the person. PendingResponder raises HumanInputRequired here and the
            # session parks; the loop catches it before any budget is charged.
            answer = self.responder.ask(suspect, question, character)

            if suspect not in self._human_interview_states:
                self._human_interview_states[suspect] = InterviewState()
            interview_state = self._human_interview_states[suspect]

            analysis = check_statement_consistency(
                answer, character, topic, interview_state.statements, question=question
            )
            interview_state.add_statement(topic, answer)

            agent_visible, engine_only = split_analysis(analysis)
            if self._inference_mode == "model":
                # The transcript is the evidence available to the detective. In
                # model mode it must notice inconsistencies itself, not receive
                # the legacy keyword checker's interpretation of an answer.
                agent_visible = {}

            # Engine-side record. Deliberately stored on the ToolBox, not on `base`,
            # so nothing here can ride back into the agent's context.
            self.human_interview_audit.append({
                "suspect": suspect,
                "topic": topic,
                "question": question,
                "answer": answer,
                **engine_only,
            })

            lines = [f'{character.get("name", suspect)} answers: "{answer}"']
            if agent_visible.get("contradictions"):
                lines.append("")
                lines.append(
                    f"NOTE: this conflicts with what {character.get('name', suspect)} "
                    f"told you earlier on '{topic}':"
                )
                for prior in agent_visible["contradictions"][:3]:
                    lines.append(f'  - earlier: "{prior}"')

            base["observation"] = "\n".join(lines)
            # Testimony is a claim, not evidence. Corroborate it with the record.
            base["evidence"] = []
            base["interview"] = {
                "suspect": suspect,
                "topic": topic,
                "question": question,
                "answer": answer,
                "statements_on_topic": len(interview_state.statements.get(topic, [])),
                **agent_visible,
            }
            return base

        key = self._arg_key(tool, arguments)
        if not key:
            base.update(ok=False, error=f"Missing/invalid arguments for '{tool}': {arguments}.")
            return base

        node = self._tools.get(tool, {})
        if key not in node:
            base.update(
                ok=False,
                error=f"No information available from '{tool}' for '{key}'.",
            )
            return base

        entry = node[key]
        base["observation"] = entry.get("observation", "")
        base["evidence"] = self._materialize(entry.get("evidence", []), tool)
        base["reveals"] = self._collect_reveals(entry.get("evidence", []))
        base["raises"] = self._collect(entry.get("evidence", []), "raises_question")
        base["resolves"] = self._collect(entry.get("evidence", []), "resolves_question")
        return base

    # ---- internals ------------------------------------------------------------
    def _materialize(self, raw: list[dict[str, Any]], tool: str) -> list[EvidenceItem]:
        """Turn raw case JSON into EvidenceItems, honouring the inference mode.

        In "model" mode `supports` and `weight` are dropped: the agent gets the item's
        id, description and source type, and has to work out the rest.

        `relates_to_incident` is dropped too, and that one is the whole point. It is
        the case author's verdict on whether a record bears on the incident — i.e. it
        is the red-herring answer, pre-filled. Leaving it in did more than give the
        answer away: because `is_support` requires it, the engine silently overrode
        any weight the agent put on a flagged item and then reported the result as
        "agent discounted (weight 0)" — telling the agent it had made a judgement it
        never made. Spotting a red herring is the most valuable inference in a
        detective case; here the agent gets to make it, and to get it wrong.

        Deliberately NOT dropped are `independent_type` / `source_group`: an access log
        genuinely is access-type evidence — that is a property of the source, not a
        judgement, and leaving it authored is what stops an agent from inventing two
        "independent" lines out of one record.
        """
        model_infers = self._inference_mode == "model"
        items = []
        for e in raw:
            items.append(
                EvidenceItem(
                    id=e["id"],
                    description=e["description"],
                    supports=None if model_infers else e.get("supports"),
                    weight=0.0 if model_infers else float(e.get("weight", 0.0)),
                    relates_to_incident=(
                        True if model_infers
                        else bool(e.get("relates_to_incident", True))
                    ),
                    independent_type=e.get("independent_type", "misc"),
                    source_group=e.get("source_group"),
                    source_tool=tool,
                    attribution="unassessed" if model_infers else "authored",
                )
            )
        return items

    def _collect_reveals(self, raw: list[dict[str, Any]]) -> dict[str, str]:
        if self._inference_mode == "model":
            return {}  # Facts must be inferred with establish_fact, never copied from the case.
        merged: dict[str, str] = {}
        for e in raw:
            merged.update(e.get("reveals", {}) or {})
        return merged

    @staticmethod
    def _collect(raw: list[dict[str, Any]], field: str) -> list[Any]:
        out = []
        for e in raw:
            if e.get(field):
                out.append(e[field])
        return out
