"""Investigators: the component that *decides the next action*.

Two implementations behind one tiny interface:

  * RuleBasedInvestigator -- deterministic, no network, drives the demo + tests.
  * LLMInvestigator       -- optional; asks a model for a schema-constrained decision.

Stopping is NOT decided here. The loop owns the stopping conditions (loop.check_stop)
so that behaviour is identical and safe regardless of which investigator is driving.
An investigator only ever answers: "given the visible state, what is the single next
action?"  It may return None to mean "I have no productive move left."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from . import affordances
from .deciders import AnthropicDecider
from .skills import SKILLS
from .state import InvestigationState
from .tools import ToolBox


@dataclass
class NextAction:
    skill: str
    question: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """An action plus its full provenance, so the loop can record honestly where each
    decision came from (rule-based, LLM, or a rule-based fallback after invalid LLM
    output). `action is None` means 'yield' — either an explicit stop or no move left."""

    action: Optional[NextAction]
    source: str = "rule_based"            # rule_based / llm / rule_based_fallback / llm_invalid
    llm_called: bool = False
    validation_status: str = "n/a"        # ok / failed / n/a
    failure_reason: Optional[str] = None
    fallback_used: bool = False
    should_stop: bool = False
    stop_reason: Optional[str] = None
    execution_error: bool = False         # evaluation-mode: invalid LLM output halts run


class Investigator:
    name = "base"

    def next_action(self, state: InvestigationState) -> Optional[NextAction]:
        raise NotImplementedError

    def decide(self, state: InvestigationState) -> Decision:
        """Default: wrap next_action() with rule-based provenance."""
        return Decision(action=self.next_action(state), source=self.name,
                        validation_status="n/a")


# ---------------------------------------------------------------------------
# Rule-based (deterministic)
# ---------------------------------------------------------------------------
class RuleBasedInvestigator(Investigator):
    name = "rule_based"

    def __init__(self, toolbox: ToolBox):
        self._tb = toolbox

    def _attempted(self, state: InvestigationState) -> set[tuple[str, Any]]:
        return {(t["tool"], t.get("arg_key")) for t in state.tools_used}

    def next_action(self, state: InvestigationState) -> Optional[NextAction]:
        attempted = self._attempted(state)
        idx = state.index

        # 1) Always start by reading the incident report.
        if ("read_incident_report", "_report") not in attempted:
            return NextAction(
                "timeline_reconstruction",
                "Read the incident report to learn the scope, suspects and time window.",
                "read_incident_report",
                {},
            )

        loc = idx.get("location")

        # 2) Who could physically reach the scene?
        if loc and ("check_access_log", loc) not in attempted:
            return NextAction(
                "access_path_analysis",
                f"Who could reach {loc} during the incident window?",
                "check_access_log",
                {"location": loc},
            )

        # 3) Corroborate with the camera.
        if loc and ("review_camera", loc) not in attempted:
            return NextAction(
                "physical_evidence_analysis",
                f"Does footage show who was at {loc} during the window?",
                "review_camera",
                {"location": loc},
            )

        # 4) Read the scene.
        if loc and ("inspect_location", loc) not in attempted:
            return NextAction(
                "physical_evidence_analysis",
                f"What does the scene at {loc} reveal about how and why it was done?",
                "inspect_location",
                {"location": loc},
            )

        # 5) Examine objects for the method.
        for obj in idx.get("objects", []):
            if ("analyze_object", obj) not in attempted:
                return NextAction(
                    "physical_evidence_analysis",
                    f"What does examining the {obj} reveal about the method?",
                    "analyze_object",
                    {"object": obj},
                )

        # 6) Resolve any outstanding raised questions before concluding. This is what
        #    stops us from ignoring — or over-reading — a witness who drew attention.
        for q in state.open_questions:
            s = q.suspect
            if s and ("verify_alibi", s) not in attempted:
                return NextAction(
                    "statement_validation",
                    f"An outstanding claim about {s} must be checked before concluding.",
                    "verify_alibi",
                    {"suspect": s},
                )

        # 7) Interview suspects to gather their accounts.
        for s in idx.get("suspects", []):
            if ("interview_character", s) not in attempted:
                return NextAction(
                    "statement_validation",
                    f"Interview {s} to obtain their account.",
                    "interview_character",
                    {"suspect": s},
                )

        # 8) Test the leading suspect's alibi.
        lead = state.leading()
        if lead and ("verify_alibi", lead.suspect) not in attempted:
            return NextAction(
                "statement_validation",
                f"Test the leading suspect {lead.suspect}'s alibi against the record.",
                "verify_alibi",
                {"suspect": lead.suspect},
            )

        # 9) Compare conflicting testimonies as a last resort.
        for pair in self._tb.valid_targets("compare_testimonies"):
            if ("compare_testimonies", pair) not in attempted:
                a, b = pair.split("|")
                return NextAction(
                    "contradiction_resolution",
                    "Compare conflicting testimonies to resolve the remaining doubt.",
                    "compare_testimonies",
                    {"a": a, "b": b},
                )

        return None  # nothing productive left; loop will finalize honestly.


# ---------------------------------------------------------------------------
# LLM-backed (optional)
# ---------------------------------------------------------------------------
DECISION_SCHEMA_HINT = {
    "selected_skill": "one of " + ", ".join(SKILLS.keys()),
    "next_question": "the single most important unanswered question",
    "tool_name": "one tool to call",
    "tool_arguments": {"<arg>": "<value>"},
    "should_stop": False,
    "stop_reason": None,
}


TOOL_LIST = [
    "read_incident_report", "interview_character", "inspect_location",
    "analyze_object", "check_access_log", "review_camera",
    "verify_alibi", "compare_testimonies",
]


class LLMInvestigator(Investigator):
    """Asks a model for a schema-constrained next action, via the Anthropic *decider*.

    The model is given ONLY the public briefing, the navigational catalog, the tool
    surface and the current visible state — never the case file or hidden truth. Every
    output is parsed and validated, and every validation failure is reported through the
    returned `Decision` so the loop can record it. Fallback is never silent.

    The decider is a plain callable(prompt: str) -> str — the smallest possible seam
    around the SDK, so validation, provenance and fallback policy stay client-agnostic.

    Two policies:
      * demo (default): invalid output falls back to the deterministic policy, VISIBLY.
      * evaluation: invalid output does NOT get silently replaced by a good rule-based
        decision — it halts the run with an `execution_error`, so a fallback-heavy run is
        never scored as a clean LLM success.

    `raw_decider` is an injectable seam (callable(prompt: str) -> str, optionally with a
    `.provider`/`.model` attribute) used by tests to exercise every path without a live
    key or network.
    """

    name = "llm"

    def __init__(self, toolbox: ToolBox, model: str | None = None,
                 briefing: dict[str, Any] | None = None, mode: str = "demo",
                 raw_decider=None):
        self._tb = toolbox
        self._briefing = briefing or {}
        self._fallback = RuleBasedInvestigator(toolbox)
        if mode not in ("demo", "evaluation"):
            raise ValueError("mode must be 'demo' or 'evaluation'")
        self.mode = mode

        if raw_decider is not None:
            self._raw = raw_decider  # test/custom seam; no API key required
            self.provider = getattr(raw_decider, "provider", "injected")
            self._model = getattr(raw_decider, "model", model or "injected")
            return

        # Build the real Anthropic decider (raises ProviderConfigError on missing key).
        decider = AnthropicDecider(model=model)
        self._raw = decider
        self.provider = decider.provider
        self._model = decider.model

    # ---- prompt surface -------------------------------------------------------
    def _tool_surface(self) -> dict[str, list[str]]:
        return {t: self._tb.valid_targets(t) for t in TOOL_LIST}

    def _prompt(self, state: InvestigationState) -> str:
        legal = [
            {"tool": la["tool"], "arguments": la["arguments"],
             "skill": la["skill"], "why": la["reason"]}
            for la in affordances.legal_actions(state)
        ]
        visible = {
            "briefing": self._briefing,
            "catalog_learned": state.index,
            "facts_revealed": state.facts_revealed,
            "known_facts": state.known_facts,
            "open_questions": [q.model_dump() for q in state.open_questions],
            "hypotheses": [
                {"suspect": h.suspect, "confidence": round(h.confidence, 2),
                 "independent_types": h.independent_types}
                for h in state.ranked()
            ],
            "tools_used": [t["tool"] for t in state.tools_used],
            "remaining_budget": state.remaining_budget,
            "legal_actions": legal,
        }
        return (
            "You are an investigator agent solving a fictional mystery. You lead the "
            "investigation. You may ONLY choose an action from `legal_actions` below — "
            "that set changes as you discover more. Pick the single most valuable next "
            "action, or set should_stop=true if you can justify a conclusion.\n\n"
            "Respond with ONLY a JSON object of this shape:\n"
            + json.dumps(DECISION_SCHEMA_HINT, indent=2)
            + "\n\nCurrent visible state:\n"
            + json.dumps(visible, indent=2)
        )

    # ---- the decision ---------------------------------------------------------
    def decide(self, state: InvestigationState) -> Decision:
        try:
            text = self._raw(self._prompt(state))
        except Exception as exc:  # API exception / timeout
            return self._on_failure(state, f"api_exception: {exc}")

        if text is None or not str(text).strip():
            return self._on_failure(state, "empty_response")

        parsed = self._parse(text)
        if parsed is None:
            return self._on_failure(state, "invalid_json")

        err = self._validate(parsed, state)
        if err:
            return self._on_failure(state, err)

        if parsed.get("should_stop"):
            # A valid, explicit stop request is an LLM decision (not a failure).
            return Decision(action=None, source="llm", llm_called=True,
                            validation_status="ok", should_stop=True,
                            stop_reason=parsed.get("stop_reason") or "llm_requested_stop")

        action = NextAction(parsed["selected_skill"],
                            parsed.get("next_question", ""),
                            parsed["tool_name"],
                            parsed.get("tool_arguments") or {})
        return Decision(action=action, source="llm", llm_called=True,
                        validation_status="ok")

    def next_action(self, state: InvestigationState) -> Optional[NextAction]:
        # Back-compat: return just the action (provenance discarded).
        return self.decide(state).action

    def _on_failure(self, state: InvestigationState, reason: str) -> Decision:
        if self.mode == "evaluation":
            # Do NOT paper over invalid output with a good rule-based move.
            return Decision(action=None, source="llm_invalid", llm_called=True,
                            validation_status="failed", failure_reason=reason,
                            execution_error=True)
        # demo: fall back, but make it explicit in the returned provenance.
        return Decision(action=self._fallback.next_action(state),
                        source="rule_based_fallback", llm_called=True,
                        validation_status="failed", failure_reason=reason,
                        fallback_used=True)

    def _validate(self, d: dict[str, Any], state: InvestigationState) -> Optional[str]:
        """Return None if valid, else a short failure reason. Enforces both structural
        validity AND legality in the current state (dynamic tool gating)."""
        if not isinstance(d, dict):
            return "invalid_schema"
        if d.get("should_stop"):
            return None  # nothing else required for a stop
        skill = d.get("selected_skill")
        if skill not in SKILLS:
            return f"unknown_skill: {skill!r}"
        tool = d.get("tool_name")
        if tool not in TOOL_LIST:
            return f"unknown_tool: {tool!r}"
        args = {} if tool == "read_incident_report" else d.get("tool_arguments", {})
        if not isinstance(args, dict):
            return "invalid_arguments: not an object"
        if tool != "read_incident_report":
            arg_key = self._tb.arg_key(tool, args)
            if arg_key is None:
                return f"invalid_arguments: {args!r} for {tool}"
            if arg_key not in self._tb.valid_targets(tool):
                return f"unavailable_tool_target: {arg_key!r} for {tool}"
        # Dynamic gating: the chosen action must be in the current legal set.
        if not affordances.is_legal(state, tool, args):
            return f"illegal_action: {tool}({affordances.action_key(tool, args)}) not permitted now"
        return None

    @staticmethod
    def _parse(text: str) -> Optional[dict[str, Any]]:
        text = str(text).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
