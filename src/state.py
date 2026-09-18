"""Pydantic models for the investigation state, hypotheses, evidence, audit trace,
and the final report.

Everything the *investigator* is allowed to see lives here. The hidden truth of a
case is deliberately NOT part of any of these models — see engine/cases.py for the
boundary between the case file (which contains the truth) and what the investigator
receives at runtime.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

BASE_PRIOR = 0.20        # starting confidence for every fresh suspect
CONFIDENCE_CAP = 0.98    # we never claim absolute certainty


class EvidenceItem(BaseModel):
    """A single piece of evidence returned by a tool and stored in state."""

    id: str
    description: str
    supports: Optional[str] = None          # suspect id this points at (or None)
    weight: float = 0.0                     # >0 incriminating, <0 exculpatory
    relates_to_incident: bool = True        # False = interesting but off-topic
    independent_type: str = "misc"          # access / camera / object / alibi ...
    source_group: Optional[str] = None      # underlying source; guards independence
    source_tool: str = ""

    # WHOSE inference the supports/weight above represent. The audit must never be
    # ambiguous about this, because it is the difference between the case author
    # having solved the case and the agent having solved it:
    #   "authored"   -> the case file labelled it (inference_mode == "authored")
    #   "unassessed" -> labelling stripped, the agent has not weighed it yet
    #   "model"      -> the agent assigned supports/weight itself via assess_evidence
    attribution: str = "authored"
    assessment_rationale: str = ""

    @property
    def is_support(self) -> bool:
        return self.weight > 0 and self.relates_to_incident

    @property
    def is_contradiction(self) -> bool:
        return self.weight < 0 and self.relates_to_incident

    @property
    def independence_key(self) -> str:
        """Two evidence items are independent only if their keys differ. A shared
        underlying source (source_group) collapses to one, so the same fact surfaced
        through two tools is not double-counted as two independent lines."""
        return self.source_group or self.independent_type


class Hypothesis(BaseModel):
    id: str
    claim: str
    suspect: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: float = BASE_PRIOR
    independent_types: list[str] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    suspect: Optional[str] = None
    note: str


class InvestigationState(BaseModel):
    """The one explicit mutable state object the loop mutates each iteration.

    NOTE: `index` and `facts_revealed` are populated *from tool results*, so they are
    legitimately investigator-visible. They never contain the hidden solution.
    """

    case_id: str
    inference_mode: str = "authored"
    iteration: int = 0
    known_facts: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    evidence_collected: list[EvidenceItem] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    remaining_budget: int = 8
    # Set once the evidence clears the conclusion bar while a driver that states its
    # own verdict is in the seat. It is a SIGNAL, not a status: the run stays
    # "investigating" until the agent actually concludes (see
    # GameSession.defer_solved_to_agent).
    conclusion_ready: bool = False
    # "investigating" | "awaiting_human" | a terminal status (solved, unresolved, ...)
    status: str = "investigating"
    final_report: Optional["FinalReport"] = None

    # Set only while status == "awaiting_human": the question a person still has to
    # answer, plus the role card the UI shows them. Cleared on resume.
    pending_human: Optional[dict[str, Any]] = None

    # Illegal calls cost no budget (a naming slip must not be lethal) but are
    # counted, so an agent stuck in a loop is still bounded.
    illegal_actions: int = 0
    # True once the loop has granted the one free closing turn at budget 0.
    final_turn_used: bool = False

    # investigator-visible working memory (learned, not hidden)
    index: dict[str, Any] = Field(default_factory=dict)
    facts_revealed: dict[str, str] = Field(default_factory=dict)
    fact_assertions: dict[str, dict[str, str]] = Field(default_factory=dict)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    resolved_questions: list[str] = Field(default_factory=list)

    # ---- convenience accessors -------------------------------------------------
    def hypothesis_for(self, suspect: str) -> Optional[Hypothesis]:
        for h in self.hypotheses:
            if h.suspect == suspect:
                return h
        return None

    def leading(self) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.confidence)

    def ranked(self) -> list[Hypothesis]:
        return sorted(self.hypotheses, key=lambda h: h.confidence, reverse=True)


class ConfidenceDelta(BaseModel):
    """A single, fully attributable change to a hypothesis's confidence."""

    hypothesis_id: str
    suspect: str
    evidence_id: str
    reason: str            # support / contradiction / unrelated_discounted
    weight: float
    source_group: Optional[str] = None
    previous: float
    new: float


class StepVerification(BaseModel):
    """The agent's self-check on the step it just took: did it actually help?"""

    verdict: str          # progress / no_progress / wasted
    useful: bool
    reason: str


class AuditEntry(BaseModel):
    """One fully inspectable iteration of the loop."""

    iteration: int
    skill: str
    decision: str
    tool: Optional[str] = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_ok: bool = True
    observation: str = ""
    state_changes: list[str] = Field(default_factory=list)
    confidence_deltas: list[ConfidenceDelta] = Field(default_factory=list)
    hypotheses_snapshot: list[dict[str, Any]] = Field(default_factory=list)
    remaining_budget: int = 0
    stop_decision: bool = False
    continue_investigation: bool = True

    # --- agentic slice: what was allowed, and was the step worthwhile ----------
    legal_actions: list[str] = Field(default_factory=list)
    verification: Optional[StepVerification] = None

    # --- decision provenance (LLM honesty) -------------------------------------
    decision_source: str = "rule_based"   # rule_based / llm / rule_based_fallback / llm_invalid
    llm_called: bool = False
    validation_status: str = "n/a"        # ok / failed / n/a
    failure_reason: Optional[str] = None
    fallback_used: bool = False


class FinalReport(BaseModel):
    status: str
    culprit: Optional[str] = None
    action: Optional[str] = None
    method: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    motive: Optional[str] = None
    confidence: Optional[float] = None
    key_evidence: list[str] = Field(default_factory=list)
    rejected_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    contradictions_resolved: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    # The engine's own factual recap, assembled from state. Always present, so a run
    # with no agent narrative (rule-based, or a halted run) still reports something.
    investigation_summary: str = ""
    # The AGENT's narrative, verbatim from submit_conclusion. Kept separate rather
    # than overwriting the engine's: the evaluator scores them on different things
    # (the engine's is true by construction, the agent's is a claim to check), and
    # collapsing them would hide which one the reader is looking at.
    agent_summary: str = ""
    agent_reasoning: str = ""


InvestigationState.model_rebuild()
