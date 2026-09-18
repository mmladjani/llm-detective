"""The investigation loop: the heart of the demo.

One `GameSession` owns the hidden case, the mutable investigation state, the tool box,
an investigator, and the audit trace. The loop is responsible for:

  observe -> ask investigator for the next action -> call the tool -> interpret the
  result into state -> recompute hypotheses -> check stopping conditions -> continue
  or synthesize a final report.

Stopping conditions live here (check_stop) so they are enforced identically no matter
which investigator drives the run. World actions cannot exceed the budget. Native
LLM conclusions require judge acceptance, but semantic review can still accept an
unsupported interpretation.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from . import affordances
from .cases import Case
from .investigator import Decision, Investigator, NextAction, RuleBasedInvestigator
from .state import (
    BASE_PRIOR,
    CONFIDENCE_CAP,
    AuditEntry,
    ConfidenceDelta,
    EvidenceItem,
    FinalReport,
    Hypothesis,
    InvestigationState,
    OpenQuestion,
    StepVerification,
)
from .tools import ToolBox
from .human_responder import HumanInputRequired, HumanResponder
from . import evaluator
from . import skill_performance
from .judge import _norm

from .thresholds import (  # noqa: F401  (re-exported: callers import these from loop)
    CONFIDENCE_THRESHOLD,
    LEAD_MARGIN,
    MIN_INDEPENDENT,
    WEAK_LEAD_FLOOR,
)

MAX_ILLEGAL_ACTIONS = 3  # illegal calls are free but not unlimited


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# Maps a unit of evidence weight onto log-odds so that the LINEAR crossing point is
# preserved exactly: a weight sum of (CONFIDENCE_THRESHOLD - BASE_PRIOR) still lands
# on CONFIDENCE_THRESHOLD. Derived from the constants rather than hardcoded, so
# moving a threshold moves this with it.
_LOG_ODDS_SCALE = (
    (_logit(CONFIDENCE_THRESHOLD) - _logit(BASE_PRIOR))
    / (CONFIDENCE_THRESHOLD - BASE_PRIOR)
)


def _combine_log_odds(weights: list[float]) -> float:
    """Accumulate evidence in log-odds space (model-inference mode only).

    The linear form (`BASE_PRIOR + sum(weights)`, clamped at CONFIDENCE_CAP) saturates:
    with the per-item ceiling at 0.35, three assessments reach the 0.98 clamp and every
    later inference becomes a silent no-op. Measured on live case-011 runs, 5 of 8 (and
    3 of 6) assessments moved nothing at all — including the sharpest reasoning in the
    trace, because the agent happened to weigh it last. An agent cannot learn from a
    number that stops responding, and a demo about reasoning should not reward whoever
    front-loads their three heaviest weights.

    A sigmoid is also the honest shape for accumulating independent evidence: each new
    item still moves the posterior, with diminishing returns, and the curve approaches
    1.0 asymptotically without ever reaching it — so "never claim absolute certainty"
    is preserved by construction rather than by a clamp that flattens everything above
    it. Authored cases keep the linear form: their weights are the case author's and
    the corpus (plus its published baseline scores) is calibrated against it.
    """
    total = _logit(BASE_PRIOR) + sum(w * _LOG_ODDS_SCALE for w in weights)
    return round(1.0 / (1.0 + math.exp(-total)), 4)

# Bounds on a weight the MODEL assigns in model-inference mode (see Case.inference_mode).
# A single piece of evidence may not be declared decisive on its own: the ceiling is
# below CONFIDENCE_THRESHOLD - BASE_PRIOR, so reaching the bar always takes more than
# one assessment. This is the one place the engine constrains the model's inference,
# and it constrains only the SCALE, never the direction.
MAX_MODEL_WEIGHT = 0.35


class InvestigationComplete(Exception):
    """Raised when a caller tries to advance a finished investigation."""


class GameSession:
    def __init__(self, case: Case, investigator: Optional[Investigator] = None,
                 requested_mode: Optional[str] = None,
                 responder: Optional[HumanResponder] = None):
        self.case = case
        self.toolbox = ToolBox(case, responder=responder)
        # Set while a run is parked on a human answer: the decision to replay once
        # the person has typed. Holding the decision (not a partial result) is what
        # makes resume exact — the tool runs once, after the answer exists.
        self._parked: Optional[tuple[Decision, list[str]]] = None
        self.state = InvestigationState(
            case_id=case.case_id, remaining_budget=case.budget,
            inference_mode=case.inference_mode,
        )
        self.investigator = investigator or RuleBasedInvestigator(self.toolbox)
        self.requested_mode = requested_mode or getattr(self.investigator, "name", "rule_based")
        # Does this driver state its own verdict? A driver that can call
        # submit_conclusion (the tool_use agent) must be the one to conclude: if the
        # engine declares "solved" the moment the thresholds are met, the run ends
        # mid-flight and the judge never sees a conclusion at all. Measured on
        # case-011 before this existed: 6/6 runs ended that way, 0 conclusions
        # accepted. The rule-based investigator has no such tool, so for it the
        # engine still closes the case itself.
        self.defer_solved_to_agent = bool(
            getattr(self.investigator, "concludes_explicitly", False)
        )
        # Hard iteration ceiling independent of budget, so a misbehaving investigator
        # (e.g. an LLM emitting endless unavailable calls) can never loop forever.
        # Model cases audit cognition too (assessments, revisions and fact assertions).
        self.max_iterations = max(case.budget * 6, 60) if case.inference_mode == "model" else max(case.budget * 3, 30)
        self.audit: list[AuditEntry] = []
        self.events: list[dict[str, Any]] = []
        # The agent's own words from submit_conclusion, set via record_agent_conclusion
        # just before the run stops. _synthesize carries them into the report so the
        # evaluator can score the narrative the agent actually wrote; previously the
        # model's reasoning was discarded and only the engine's f-string survived.
        self.agent_conclusion: Optional[dict[str, Any]] = None
        self.conclusion_review: Optional[dict[str, Any]] = None

    # ---- public API -----------------------------------------------------------
    def record_event(self, event: dict[str, Any]) -> None:
        self.events.append({**event, "seq": len(self.events) + 1})

    def _record_action_event(self, entry: AuditEntry) -> None:
        self.record_event({"type": "tool", "name": entry.tool,
                           "input": entry.tool_arguments, "ok": entry.tool_ok,
                           "skill": entry.skill, "audit": entry.model_dump(),
                           "result": {"observation": entry.observation,
                                      "state_changes": entry.state_changes,
                                      "hypotheses": entry.hypotheses_snapshot,
                                      "remaining_budget": entry.remaining_budget}})

    def step(self) -> AuditEntry:
        st = self.state
        if st.status == "awaiting_human":
            raise InvestigationComplete(
                "Investigation is parked awaiting a human answer — call "
                "provide_human_answer() before stepping again."
            )
        if st.status != "investigating":
            raise InvestigationComplete(
                f"Investigation already finished with status '{st.status}'."
            )

        # Hard iteration ceiling — cannot be exceeded regardless of investigator.
        if st.iteration >= self.max_iterations:
            return self._finalize(
                "execution_error", action=None,
                decision=f"Hard iteration limit ({self.max_iterations}) reached.",
                provenance=Decision(action=None, source="loop_guard"),
            )

        # Snapshot the legal action set the agent had to choose from this turn.
        legal = affordances.legal_actions(st)
        legal_labels = [f"{la['tool']}({la['key']})" for la in legal]

        decision: Decision = self.investigator.decide(st)
        return self.execute_decision(decision, legal_labels)

    def execute_decision(self, decision: Decision,
                         legal_labels: Optional[list[str]] = None) -> AuditEntry:
        """Execute one already-made decision (from any driver: rule-based, schema LLM,
        or the native tool_use agent). Owns legality, budget, state application, audit
        and stopping — the decision-maker can never bypass these."""
        st = self.state
        if st.status == "awaiting_human":
            raise InvestigationComplete(
                "Investigation is parked awaiting a human answer — call "
                "provide_human_answer() before stepping again."
            )
        if st.status != "investigating":
            raise InvestigationComplete(
                f"Investigation already finished with status '{st.status}'."
            )
        if legal_labels is None:
            legal_labels = [f"{la['tool']}({la['key']})"
                            for la in affordances.legal_actions(st)]

        # Evaluation-mode invalid LLM output: halt honestly, do not fake success.
        if decision.execution_error:
            return self._finalize(
                "execution_error", action=None,
                decision=f"LLM output rejected ({decision.failure_reason}); "
                         "run halted under evaluation policy.",
                provenance=decision,
            )

        action = decision.action

        if action is None:
            if self.defer_solved_to_agent:
                # Only an explicitly accepted proposal may produce a solved/unresolved verdict.
                # Recomputing the numeric stop here used to turn rejected accusations into solves.
                review = self.conclusion_review or {}
                if self.agent_conclusion and review.get("ok"):
                    term = self.agent_conclusion["verdict"]
                else:
                    term = "conclusion_rejected" if self.agent_conclusion else "execution_error"
                return self._finalize(term, action=None,
                                      decision=decision.stop_reason or "No accepted conclusion.",
                                      provenance=decision)
            # Either an explicit (valid) LLM stop, or no productive move remains.
            term = self._check_stop(st) or (
                "budget_exhausted" if st.remaining_budget <= 0 else "unresolved"
            )
            reason = decision.stop_reason or "No productive action remains."
            return self._finalize(term, action=None, decision=reason, provenance=decision)

        # On the free closing turn no world action remains — only submit_conclusion.
        closing_turn = st.remaining_budget <= 0
        illegal = not affordances.is_legal(st, action.tool, action.arguments)

        if closing_turn:
            result = {"tool": action.tool, "arguments": action.arguments, "ok": False,
                      "observation": "",
                      "error": ("budget_exhausted: no world actions remain. This turn is "
                                "free but it is your last — call submit_conclusion with "
                                "the verdict the evidence supports."),
                      "evidence": []}
        # Guardrail: the engine also enforces legality at execution time. A validated
        # LLM choice or a rule-based choice is always legal; this only fires defensively.
        elif illegal:
            # FEEDBACK, not just rejection: tell the model what IS legal now, so it can
            # correct on the next move. An illegal move does NOT spend budget — a
            # mistake in a NAME must not be a death sentence — but it is counted
            # (MAX_ILLEGAL_ACTIONS), so an agent stuck in a loop is still bounded.
            st.illegal_actions += 1
            left = MAX_ILLEGAL_ACTIONS - st.illegal_actions
            legal_now = [f"{la['tool']}({la['key']})"
                         for la in affordances.legal_actions(st)][:12]
            result = {"tool": action.tool, "arguments": action.arguments, "ok": False,
                      "observation": "",
                      "error": (f"illegal_action: {action.tool}"
                                f"({affordances.action_key(action.tool, action.arguments)}) "
                                f"is not permitted in the current state. Legal actions "
                                f"now: {', '.join(legal_now) or '(none)'}. Use exact "
                                f"snake_case ids from earlier results. This cost you no "
                                f"budget; {left} illegal call(s) remain before the run "
                                f"is halted."),
                      "evidence": []}
        else:
            # Execute the tool — never let a tool exception corrupt state or crash the loop.
            try:
                result = self.toolbox.call(action.tool, action.arguments)
            except HumanInputRequired as pending:
                # A person has to answer before this action means anything. Park the
                # run WITHOUT charging budget or writing an audit entry: nothing has
                # happened in the world yet, and the replay would double-count it.
                return self._park_for_human(pending, decision, legal_labels)
            except Exception as exc:  # controlled execution result
                result = {"tool": action.tool, "arguments": action.arguments, "ok": False,
                          "observation": "", "error": f"tool_exception: {exc}", "evidence": []}

        arg_key = self.toolbox.arg_key(action.tool, action.arguments)
        st.iteration += 1
        st.tools_used.append(
            {
                "tool": action.tool,
                "arguments": action.arguments,
                "arg_key": arg_key,
                "ok": result["ok"],
                "iteration": st.iteration,
            }
        )
        # An EXECUTED action costs one budget unit — success or not. Illegal calls and
        # the free closing turn cost nothing: they touched no part of the world, and
        # charging for them turns a naming slip into a lost case (feedback beats a
        # silent penalty the agent cannot learn from).
        if not illegal and not closing_turn:
            st.remaining_budget = max(0, st.remaining_budget - 1)

        before = (len(st.evidence_collected), len(st.known_facts),
                  len(st.open_questions), len(st.resolved_questions))
        changes, deltas = self._apply_result(st, result)
        after = (len(st.evidence_collected), len(st.known_facts),
                 len(st.open_questions), len(st.resolved_questions))
        verification = self._verify_step(result, before, after, deltas)

        term = self._check_stop(st)
        if self.defer_solved_to_agent:
            # The evidence now supports a conclusion — but reaching the bar is not the
            # same act as stating what the evidence means, and only the agent can do
            # the second one. Hand it the signal and let the run continue; the budget
            # and the iteration ceiling still bound it, so an agent that never
            # concludes still terminates (budget_exhausted), just not as a "solve" it
            # never articulated.
            st.conclusion_ready = term == "solved"
            # World exhaustion is not a verdict. Let the agent assess its last findings
            # and conclude; an attempted world action on the closing turn still halts.
            term = "budget_exhausted" if closing_turn else None
            if st.remaining_budget <= 0:
                st.final_turn_used = True
        if term is None and st.illegal_actions >= MAX_ILLEGAL_ACTIONS:
            term = "illegal_action_limit"
        stop = term is not None

        entry = AuditEntry(
            iteration=st.iteration,
            skill=action.skill,
            decision=action.question,
            tool=action.tool,
            tool_arguments=action.arguments,
            tool_ok=result["ok"],
            observation=result.get("observation", "") or (result.get("error") or ""),
            state_changes=changes,
            confidence_deltas=deltas,
            hypotheses_snapshot=self._hyp_snapshot(st),
            remaining_budget=st.remaining_budget,
            stop_decision=stop,
            continue_investigation=not stop,
            decision_source=decision.source,
            llm_called=decision.llm_called,
            validation_status=decision.validation_status,
            failure_reason=decision.failure_reason,
            fallback_used=decision.fallback_used,
            legal_actions=legal_labels,
            verification=verification,
        )
        self.audit.append(entry)
        self._record_action_event(entry)
        if stop:
            self._set_final(term)
            # Close the trace with a synthesis marker. It is NOT itself a decision (the
            # decision was the action entry above), so tag it "closing" to keep the
            # run-metadata decision counts honest.
            self.audit.append(self._synthesis_entry(
                f"Stopping condition met: {term}.",
                provenance=Decision(action=None, source="closing"),
            ))
            return self.audit[-1]
        return entry

    def run(self, max_steps: Optional[int] = None) -> list[AuditEntry]:
        cap = max_steps if max_steps is not None else self.max_iterations + 2
        out = []
        for _ in range(cap):
            # 'awaiting_human' stops the loop too: a run cannot drive itself past a
            # question only a person can answer.
            if self.state.status != "investigating":
                break
            entry = self.step()
            if entry is not None:
                out.append(entry)
        return out

    # ---- human-in-the-loop pause / resume --------------------------------------
    def _park_for_human(self, pending: HumanInputRequired, decision: Decision,
                        legal_labels: list[str]) -> None:
        """Suspend the run until a person answers. Returns None — no action occurred."""
        self._parked = (decision, legal_labels)
        self.state.status = "awaiting_human"
        self.state.pending_human = pending.to_dict()
        return None

    @property
    def awaiting_human(self) -> bool:
        return self.state.status == "awaiting_human"

    def pending_question(self) -> Optional[dict[str, Any]]:
        """What the UI needs to render: suspect, question, and the role card."""
        return self.state.pending_human if self.awaiting_human else None

    def provide_human_answer(self, answer: str) -> AuditEntry:
        """Resume a parked run with a person's answer.

        Buffers the answer, then replays the exact decision that parked. The tool
        call re-runs, this time finding the answer already there, so the action
        executes once and is audited once.
        """
        if not self.awaiting_human:
            raise InvestigationComplete(
                f"Not awaiting a human answer (status '{self.state.status}')."
            )
        pending = self.state.pending_human or {}
        submit = getattr(self.toolbox.responder, "submit", None)
        if submit is None:
            raise TypeError(
                "The configured responder cannot accept a submitted answer; "
                "a parked session requires a PendingResponder."
            )
        submit(pending.get("suspect", ""), pending.get("question", ""), answer)

        decision, legal_labels = self._parked
        self._parked = None
        self.state.pending_human = None
        self.state.status = "investigating"
        return self.execute_decision(decision, legal_labels)

    def record_agent_conclusion(self, proposal: dict[str, Any],
                                review: Optional[dict[str, Any]] = None) -> None:
        """Keep the proposal and its review together; only approval authorizes a verdict."""
        self.agent_conclusion = dict(proposal or {})
        self.conclusion_review = dict(review) if review is not None else None

    # ---- model-supplied inference (inference_mode == "model") -----------------
    @property
    def inference_mode(self) -> str:
        return self.case.inference_mode

    def unassessed_evidence(self) -> list[dict[str, Any]]:
        """Collected evidence the agent has not yet weighed. Empty in authored mode."""
        if self.inference_mode != "model":
            return []
        return [
            {"id": e.id, "description": e.description,
             "independent_type": e.independent_type}
            for e in self.state.evidence_collected
            if e.attribution == "unassessed"
        ]

    def apply_assessment(self, spec: dict[str, Any],
                         provenance: Decision) -> AuditEntry:
        """Record the AGENT's own reading of one piece of evidence.

        This is the counterpart to authored `supports`/`weight`: in model-inference
        mode the engine supplies no interpretation, and this is where the agent's
        interpretation enters state. The engine still owns the bookkeeping — it
        validates the reference, bounds the magnitude, recomputes confidence and
        writes an attributable delta — but the direction and the strength of the
        inference are the model's.

        Costs no budget: it consumes no part of the world. It does NOT run the
        stopping conditions either, and that is the important part — if re-weighting
        its own evidence could trigger the engine's "solved" branch, an agent could
        finish a case by declaring numbers instead of by finding facts. Concluding
        stays a deliberate act: `submit_conclusion`, through the judge.
        """
        st = self.state
        if st.status != "investigating":
            raise InvestigationComplete(
                f"Investigation already finished with status '{st.status}'."
            )

        changes: list[str] = []
        deltas: list[ConfidenceDelta] = []
        error = self._validate_assessment(st, spec)

        if error is None:
            ev = next(e for e in st.evidence_collected if e.id == spec["evidence_id"])
            suspect = str(spec["supports"]).strip().lower()
            weight = round(float(spec["weight"]), 4)
            rationale = str(spec.get("rationale", "")).strip()

            # Re-assessment is legal (new facts change what an old record means), so
            # detach first: without this, re-attributing an item would leave it
            # counted on the previous suspect as well and inflate two hypotheses
            # from one piece of evidence.
            self._detach_evidence(st, ev.id, changes)

            ev.supports = suspect
            ev.weight = weight
            ev.attribution = "model"
            ev.assessment_rationale = rationale

            h = st.hypothesis_for(suspect)
            if h is None:
                h = Hypothesis(
                    id=f"H{len(st.hypotheses) + 1}",
                    claim=f"{suspect} is responsible for the incident.",
                    suspect=suspect,
                )
                st.hypotheses.append(h)

            prev = h.confidence
            if ev.is_support:
                h.supporting_evidence.append(ev.id)
                reason = "model_support"
                changes.append(
                    f"[{suspect}] agent weighed {ev.id} as +{weight} support"
                    + (f": {rationale}" if rationale else "")
                )
            elif ev.is_contradiction:
                h.contradicting_evidence.append(ev.id)
                reason = "model_contradiction"
                changes.append(
                    f"[{suspect}] agent weighed {ev.id} as {weight} exculpatory"
                    + (f": {rationale}" if rationale else "")
                )
            else:
                reason = "model_discounted"
                changes.append(
                    f"[{suspect}] agent discounted {ev.id} (weight 0)"
                    + (f": {rationale}" if rationale else "")
                )

            self._recompute(st, h)
            deltas.append(ConfidenceDelta(
                hypothesis_id=h.id, suspect=suspect, evidence_id=ev.id,
                reason=reason, weight=weight,
                source_group=ev.independence_key,
                previous=round(prev, 4), new=round(h.confidence, 4),
            ))
            observation = (f"Assessment recorded: {ev.id} -> {suspect} at weight "
                           f"{weight}. {suspect} is now at {round(h.confidence, 4)} "
                           f"on {len(h.independent_types)} independent line(s).")
            verification = StepVerification(
                verdict="progress", useful=True,
                reason="the agent's own attribution was applied to the hypotheses",
            )
        else:
            observation = error
            verification = StepVerification(
                verdict="wasted", useful=False, reason=error,
            )

        st.iteration += 1
        entry = AuditEntry(
            iteration=st.iteration,
            skill=getattr(self.investigator, "active_skill", None) or "unspecified",
            decision=str(spec.get("rationale", "")) or "assess evidence",
            tool="assess_evidence",
            tool_arguments=dict(spec),
            tool_ok=error is None,
            observation=observation,
            state_changes=changes,
            confidence_deltas=deltas,
            hypotheses_snapshot=self._hyp_snapshot(st),
            remaining_budget=st.remaining_budget,
            stop_decision=False,
            continue_investigation=True,
            decision_source=provenance.source,
            llm_called=provenance.llm_called,
            validation_status=provenance.validation_status,
            legal_actions=[],
            verification=verification,
        )
        self.audit.append(entry)
        self._record_action_event(entry)
        return entry

    def establish_fact(self, spec: dict[str, Any], provenance: Decision) -> AuditEntry:
        """Apply a revisable, evidence-cited inference. Candidate membership is not truth."""
        st = self.state
        if st.status != "investigating":
            raise InvestigationComplete(f"Investigation already finished: {st.status}")
        field = spec.get("field")
        value = spec.get("value")
        eid = spec.get("evidence_id")
        rationale = spec.get("rationale")
        rationale = rationale.strip() if isinstance(rationale, str) else ""
        error = None
        if self.inference_mode != "model":
            error = "establish_fact is only available in model inference mode."
        elif not isinstance(field, str) or field not in self.case.fact_candidates:
            error = "Choose a field from fact_candidates in the public briefing."
        elif value not in self.case.fact_candidates[field]:
            error = f"Choose a value from fact_candidates[{field!r}]."
        elif not any(e.id == eid for e in st.evidence_collected):
            error = "evidence_id must cite a collected record."
        elif not rationale:
            error = "Missing argument: 'rationale'. Explain how the cited record supports the fact."
        changes = []
        if error is None:
            old = st.fact_assertions.get(field)
            if old:
                prior = f"Agent inferred {field} = {old['value']}."
                st.known_facts = [f for f in st.known_facts if f != prior]
                changes.append(f"Earlier {field} inference withdrawn: {old['value']}")
            st.facts_revealed[field] = value
            st.fact_assertions[field] = {"value": value, "evidence_id": eid,
                                         "rationale": rationale, "attribution": "model"}
            st.known_facts.append(f"Agent inferred {field} = {value}.")
            changes.append(f"Agent inferred {field} = {value} from {eid}: {rationale}")
        st.iteration += 1
        entry = AuditEntry(
            iteration=st.iteration, skill=getattr(self.investigator, "active_skill", None) or "unspecified",
            decision=rationale, tool="establish_fact", tool_arguments=dict(spec),
            tool_ok=error is None, observation=error or changes[-1], state_changes=changes,
            hypotheses_snapshot=self._hyp_snapshot(st), remaining_budget=st.remaining_budget,
            decision_source=provenance.source, llm_called=provenance.llm_called,
            validation_status=provenance.validation_status,
        )
        self.audit.append(entry)
        self._record_action_event(entry)
        return entry

    def _validate_assessment(self, st: InvestigationState,
                             spec: dict[str, Any]) -> Optional[str]:
        """Return an error string, or None when the assessment is applicable."""
        if self.inference_mode != "model":
            return ("assess_evidence is only available on cases where the evidence "
                    "arrives unlabelled; this case already carries authored weights.")
        ev_id = str(spec.get("evidence_id", "")).strip()
        if not ev_id:
            return "Missing argument: 'evidence_id'."
        ev = next((e for e in st.evidence_collected if e.id == ev_id), None)
        if ev is None:
            have = ", ".join(e.id for e in st.evidence_collected) or "(none yet)"
            return (f"No evidence with id '{ev_id}' has been collected. You can only "
                    f"weigh what a tool has actually returned. Collected: {have}.")
        suspect = str(spec.get("supports", "")).strip().lower()
        if not suspect:
            return "Missing argument: 'supports' (the suspect id this points at)."
        known = [str(s).lower() for s in st.index.get("suspects", [])]
        if known and suspect not in known:
            return (f"'{suspect}' is not a suspect in this case. Known suspects: "
                    f"{', '.join(known)}.")
        raw_weight = spec.get("weight")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            return f"'weight' must be a number, got {raw_weight!r}."
        if weight != weight or abs(weight) == float("inf"):  # NaN / inf
            return "'weight' must be a finite number."
        if abs(weight) > MAX_MODEL_WEIGHT:
            return (f"weight {weight} exceeds the ±{MAX_MODEL_WEIGHT} limit for a "
                    f"single piece of evidence. No one record is decisive on its own — "
                    f"corroborate it with an independent source instead.")
        # The schema marks `rationale` required, but "required" only means the key is
        # present — an empty string satisfied it, and the reason an assessment was made
        # is the one part of it a reader cannot reconstruct from the numbers.
        #
        # Deliberately no minimum length: any threshold would be arbitrary, and this
        # repo's own cases show legitimately terse reasons ("her card, inside the
        # window, nobody else's"). Emptiness is the hole worth closing; judging whether
        # a stated reason is a GOOD one is the reviewer's job, not a character count's.
        if not isinstance(spec.get("rationale"), str) or not spec["rationale"].strip():
            return ("Missing argument: 'rationale'. Say why this record carries the "
                    "weight you gave it — the number alone is not an inference, and "
                    "the audit records your reason next to it.")
        return None

    def _detach_evidence(self, st: InvestigationState, ev_id: str,
                         changes: list[str]) -> None:
        """Remove ev_id from every hypothesis and recompute the ones it touched."""
        for h in st.hypotheses:
            if ev_id in h.supporting_evidence or ev_id in h.contradicting_evidence:
                h.supporting_evidence = [x for x in h.supporting_evidence if x != ev_id]
                h.contradicting_evidence = [x for x in h.contradicting_evidence
                                            if x != ev_id]
                self._recompute(st, h)
                changes.append(f"[{h.suspect}] earlier assessment of {ev_id} withdrawn")

    # ---- run provenance (LLM honesty) -----------------------------------------
    def run_metadata(self) -> dict[str, Any]:
        # Every audit entry is a decision except the pure "closing" synthesis marker.
        decisions_list = [a for a in self.audit if a.decision_source != "closing"]
        llm = sum(1 for a in decisions_list if a.decision_source == "llm")
        fallback = sum(1 for a in decisions_list if a.fallback_used)
        invalid = sum(1 for a in decisions_list if a.decision_source == "llm_invalid")
        rule = sum(1 for a in decisions_list if a.decision_source == "rule_based")
        fully_llm = (
            self.requested_mode == "llm" and fallback == 0 and invalid == 0 and llm > 0
        )
        return {
            "requested_mode": self.requested_mode,
            "provider": getattr(self.investigator, "provider", None),
            "model": getattr(self.investigator, "_model", None),
            "decisions": len(decisions_list),
            "llm_decisions": llm,
            "fallback_decisions": fallback,
            "invalid_decisions": invalid,
            "rule_based_decisions": rule,
            "fully_llm_driven": fully_llm,
            # Whose inference produced the hypothesis ranking. In "model" mode the
            # ranking the judge measures the conclusion against was built by the
            # agent's own assess_evidence calls, not by authored weights — without
            # this the two modes are indistinguishable in a saved run.
            "inference_mode": self.inference_mode,
            "assessments": sum(1 for a in self.audit if a.tool == "assess_evidence"),
            "facts_inferred": len(self.state.fact_assertions),
            "conclusion_accepted": (self.conclusion_review or {}).get("ok"),
        }

    # ---- interpreting a tool result into state --------------------------------
    def _apply_result(self, st: InvestigationState, result: dict[str, Any]):
        changes: list[str] = []
        deltas: list[ConfidenceDelta] = []
        if result.get("ok"):
            st.observations.append({
                "tool": result.get("tool"), "arguments": result.get("arguments", {}),
                "observation": result.get("observation", ""),
                "evidence_ids": [e.id for e in result.get("evidence", [])],
            })

        # read_incident_report seeds the board.
        catalog = result.get("catalog")
        if catalog and not st.index:
            st.index = catalog
            st.known_facts.append(
                f"Incident at {catalog.get('location')} during "
                f"{catalog.get('time_window', {}).get('from')}-"
                f"{catalog.get('time_window', {}).get('to')}."
            )
            for i, suspect in enumerate(catalog.get("suspects", []), start=1):
                st.hypotheses.append(
                    Hypothesis(
                        id=f"H{i}",
                        claim=f"{suspect} is responsible for the incident.",
                        suspect=suspect,
                        confidence=BASE_PRIOR,
                    )
                )
            changes.append(
                "Seeded hypotheses for: " + ", ".join(catalog.get("suspects", []))
            )

        # Evidence items.
        for ev in result.get("evidence", []):
            self._store_evidence(st, ev, changes, deltas)

        # Revealed facts (action/method/time/motive/location) — evidence-backed.
        for k, v in (result.get("reveals") or {}).items():
            if st.facts_revealed.get(k) != v:
                st.facts_revealed[k] = v
                st.known_facts.append(f"Established {k} = {v}.")
                changes.append(f"Fact established: {k} = {v}.")

        # Newly raised questions.
        for q in result.get("raises") or []:
            note = q.get("note", "")
            suspect = q.get("suspect")
            if not any(x.note == note for x in st.open_questions):
                st.open_questions.append(OpenQuestion(suspect=suspect, note=note))
                st.unverified_claims.append(note)
                changes.append(f"Open question raised: {note}")

        # Resolved questions.
        for suspect in result.get("resolves") or []:
            before = len(st.open_questions)
            st.open_questions = [q for q in st.open_questions if q.suspect != suspect]
            if len(st.open_questions) < before:
                st.resolved_questions.append(suspect)
                changes.append(f"Open question about {suspect} resolved.")

        return changes, deltas

    def _store_evidence(self, st: InvestigationState, ev: EvidenceItem,
                        changes: list[str], deltas: list[ConfidenceDelta]):
        if not any(e.id == ev.id for e in st.evidence_collected):
            st.evidence_collected.append(ev)
            if ev.attribution == "unassessed":
                # New information is progress even before the model interprets it.
                # In particular, it restores the native driver's revision allowance.
                changes.append(f"Collected unassessed record: {ev.id}.")

        if not ev.supports:
            return

        h = st.hypothesis_for(ev.supports)
        if h is None:
            h = Hypothesis(
                id=f"H{len(st.hypotheses) + 1}",
                claim=f"{ev.supports} is responsible for the incident.",
                suspect=ev.supports,
            )
            st.hypotheses.append(h)

        prev = h.confidence
        reason = None
        if ev.is_support:
            if ev.id not in h.supporting_evidence:  # dedup: never count an id twice
                h.supporting_evidence.append(ev.id)
                changes.append(f"[{h.suspect}] +support: {ev.description}")
                reason = "support"
        elif ev.is_contradiction:
            if ev.id not in h.contradicting_evidence:
                h.contradicting_evidence.append(ev.id)
                note = f"{ev.description}"
                if note not in st.contradictions:
                    st.contradictions.append(note)
                changes.append(f"[{h.suspect}] -contradiction: {ev.description}")
                reason = "contradiction"
        else:
            # Interesting but not tied to the incident — must NOT raise guilt.
            note = f"{ev.description} (noted, but not treated as evidence of guilt)."
            if note not in st.unverified_claims:
                st.unverified_claims.append(note)
            changes.append(f"[{h.suspect}] discounted (unrelated): {ev.description}")
            reason = "unrelated_discounted"

        self._recompute(st, h)

        # Record a fully attributable confidence delta (even a zero-change one for a
        # discounted item, so the reasoning is auditable). Duplicate ids produce no
        # reason and no delta.
        if reason is not None:
            deltas.append(ConfidenceDelta(
                hypothesis_id=h.id, suspect=h.suspect, evidence_id=ev.id,
                reason=reason, weight=ev.weight, source_group=ev.independence_key,
                previous=round(prev, 4), new=round(h.confidence, 4),
            ))

    def _recompute(self, st: InvestigationState, h: Hypothesis):
        by_id = {e.id: e for e in st.evidence_collected}
        weights: list[float] = []
        groups: list[str] = []
        for eid in h.supporting_evidence:
            e = by_id.get(eid)
            if e:
                weights.append(e.weight)
                # Independence is by underlying source group, not merely by tool/type,
                # so the same source seen twice is not two independent lines.
                if e.independence_key not in groups:
                    groups.append(e.independence_key)
        for eid in h.contradicting_evidence:
            e = by_id.get(eid)
            if e:
                weights.append(e.weight)  # weight is negative

        if self.inference_mode == "model":
            h.confidence = _combine_log_odds(weights)
        else:
            h.confidence = max(0.0, min(CONFIDENCE_CAP, round(BASE_PRIOR + sum(weights), 4)))
        h.independent_types = groups

    # ---- stopping conditions --------------------------------------------------
    def _check_stop(self, st: InvestigationState) -> Optional[str]:
        if not st.hypotheses:
            return None

        ranked = st.ranked()
        lead = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = lead.confidence - (second.confidence if second else 0.0)
        action_known = "action" in st.facts_revealed
        method_known = "method" in st.facts_revealed

        # 1) Solved: a well-supported, dominant, complete conclusion with nothing
        #    left dangling.
        if (
            lead.confidence >= CONFIDENCE_THRESHOLD
            and len(lead.independent_types) >= MIN_INDEPENDENT
            and margin >= LEAD_MARGIN
            and action_known
            and method_known
            and not st.open_questions
        ):
            return "solved"

        # 2) Budget exhausted — but grant ONE free closing turn first. Without it, an
        #    investigation whose last action lands the final piece of evidence is cut
        #    off before it can state the conclusion it just earned, and reports null.
        #    The turn buys no world action (execute_decision refuses those), only the
        #    chance to submit_conclusion — which the judge still has to accept.
        if st.remaining_budget <= 0:
            if not st.final_turn_used:
                st.final_turn_used = True
                return None
            if lead.confidence >= WEAK_LEAD_FLOOR:
                return "budget_exhausted"
            return "unresolved"

        # 3) Honest insufficiency: we have exhausted the physical trail and heard
        #    everyone, yet no suspect is both strong and independently supported.
        if self._physical_exhausted(st) and self._interviews_done(st):
            if lead.confidence < CONFIDENCE_THRESHOLD or len(lead.independent_types) < MIN_INDEPENDENT:
                return "unresolved"

        return None

    @staticmethod
    def _verify_step(result: dict[str, Any], before: tuple, after: tuple,
                     deltas: list[ConfidenceDelta]) -> StepVerification:
        """Self-check on the step just taken: did it move the investigation forward?
        This is the loop's built-in verifier — a wasted or unproductive action is flagged
        honestly, so the agent (and the audit) can see good vs bad decisions."""
        if not result.get("ok"):
            return StepVerification(verdict="wasted", useful=False,
                                    reason=result.get("error") or "tool call failed")
        moved = after != before or bool(deltas)
        if moved:
            return StepVerification(verdict="progress", useful=True,
                                    reason="added evidence / facts / questions or updated confidence")
        return StepVerification(verdict="no_progress", useful=False,
                                reason="the call succeeded but yielded no new information")

    def _attempted(self, st: InvestigationState) -> set:
        return {(t["tool"], t.get("arg_key")) for t in st.tools_used}

    def _physical_exhausted(self, st: InvestigationState) -> bool:
        attempted = self._attempted(st)
        loc = st.index.get("location")
        core = [
            ("read_incident_report", "_report"),
            ("check_access_log", loc),
            ("review_camera", loc),
            ("inspect_location", loc),
        ]
        core += [("analyze_object", o) for o in st.index.get("objects", [])]
        return all(c in attempted for c in core)

    def _interviews_done(self, st: InvestigationState) -> bool:
        attempted = self._attempted(st)
        suspects = st.index.get("suspects", [])
        return bool(suspects) and all(
            ("interview_character", s) in attempted for s in suspects
        )

    # ---- finalization ---------------------------------------------------------
    def _finalize(self, status: str, action: Optional[NextAction], decision: str,
                  provenance: Optional[Decision] = None) -> AuditEntry:
        self._set_final(status)
        entry = self._synthesis_entry(decision, provenance)
        self.audit.append(entry)
        return entry

    def _synthesis_entry(self, decision: str,
                         provenance: Optional[Decision] = None) -> AuditEntry:
        st = self.state
        p = provenance
        return AuditEntry(
            iteration=st.iteration,
            skill="final_case_synthesis",
            decision=decision,
            tool=None,
            tool_ok=True,
            observation=st.final_report.investigation_summary if st.final_report else "",
            state_changes=[f"Investigation finished: {st.status}."],
            hypotheses_snapshot=self._hyp_snapshot(st),
            remaining_budget=st.remaining_budget,
            stop_decision=True,
            continue_investigation=False,
            decision_source=p.source if p else "rule_based",
            llm_called=p.llm_called if p else False,
            validation_status=p.validation_status if p else "n/a",
            failure_reason=p.failure_reason if p else None,
            fallback_used=p.fallback_used if p else False,
        )

    def _set_final(self, status: str):
        st = self.state
        st.status = status
        st.final_report = self._synthesize(status)
        self._record_skill_performance()

    def _synthesize(self, status: str) -> FinalReport:
        st = self.state
        ranked = st.ranked()
        lead = ranked[0] if ranked else None
        by_id = {e.id: e for e in st.evidence_collected}

        key_ev = []
        if lead:
            key_ev = [
                by_id[eid].description
                for eid in lead.supporting_evidence
                if eid in by_id
            ]
        rejected = [
            {
                "suspect": h.suspect,
                "confidence": round(h.confidence, 2),
                "reason": self._rejection_reason(h),
            }
            for h in ranked[1:]
        ]
        unresolved_q = [q.note for q in st.open_questions]
        tools_used = [t["tool"] for t in st.tools_used if t.get("ok")]

        proposal = self.agent_conclusion or {}
        agent_summary = str(proposal.get("summary") or "").strip()
        agent_reasoning = str(proposal.get("reasoning") or "").strip()

        # G: the report must name whoever the agent actually accused, not just
        # whoever the arithmetic ranked highest. `agent_conclusion` is only set (by
        # record_agent_conclusion, in AgentInvestigator.decide's submit_conclusion
        # handler) when the tool_use agent concluded; the rule-based investigator has
        # no such tool, so authored-mode runs always leave `report_hyp` as `lead`,
        # which is what keeps that corpus bit-identical. Canonicalize the accused
        # name the same way judge.py::check_conclusion does before matching a
        # hypothesis, and fall back to `lead` if nothing matches (no conclusion, an
        # empty/unrecognized name, or a verdict with no culprit at all).
        report_hyp = lead
        if self.agent_conclusion:
            accused = _norm(proposal.get("culprit"))
            if accused:
                matched = next((h for h in ranked if _norm(h.suspect) == accused), None)
                if matched is not None:
                    report_hyp = matched

        if status == "solved" and lead and report_hyp:
            rejected = [{"suspect": h.suspect, "confidence": round(h.confidence, 2),
                         "reason": self._rejection_reason(h)}
                        for h in ranked if h.suspect != report_hyp.suspect]
            report_key_ev = [
                by_id[eid].description
                for eid in report_hyp.supporting_evidence
                if eid in by_id
            ]
            summary = (
                f"{report_hyp.suspect} {st.facts_revealed.get('action')} at "
                f"{st.index.get('location')} around {st.facts_revealed.get('time')}; "
                f"method: {st.facts_revealed.get('method')}; "
                f"motive: {st.facts_revealed.get('motive')}. "
                f"Confidence {round(report_hyp.confidence, 2)} from "
                f"{len(report_hyp.independent_types)} independent lines of evidence "
                f"({', '.join(report_hyp.independent_types)})."
            )
            return FinalReport(
                status="solved",
                culprit=report_hyp.suspect,
                action=st.facts_revealed.get("action"),
                method=st.facts_revealed.get("method"),
                location=st.index.get("location"),
                time=st.facts_revealed.get("time"),
                motive=st.facts_revealed.get("motive"),
                confidence=round(report_hyp.confidence, 2),
                key_evidence=report_key_ev,
                rejected_hypotheses=rejected,
                contradictions_resolved=st.contradictions[:],
                unresolved_questions=unresolved_q,
                tools_used=tools_used,
                investigation_summary=summary,
                agent_summary=agent_summary,
                agent_reasoning=agent_reasoning,
            )

        # Honest non-conclusion: leave the truth fields null rather than fabricate.
        reason = {
            "conclusion_rejected": "The judge did not accept the agent's conclusion; no accusation is issued.",
            "unresolved": "The available evidence was not strong enough to name a culprit with confidence.",
            "budget_exhausted": "The investigation budget ran out before a conclusion could be firmly established.",
            "illegal_action_limit": (
                f"The investigation was halted after {MAX_ILLEGAL_ACTIONS} illegal tool "
                "calls: the agent kept naming targets that do not exist in this case."
            ),
        }.get(status, f"Investigation ended with status '{status}'.")
        summary = (
            reason
            + (f" Strongest lead was {lead.suspect} at confidence "
               f"{round(lead.confidence, 2)}; that heuristic is not an accepted verdict."
               if lead else "")
        )
        return FinalReport(
            status=status,
            culprit=None,
            action=None,
            method=None,
            location=None,
            time=None,
            motive=None,
            confidence=round(lead.confidence, 2) if lead else None,
            key_evidence=key_ev,
            rejected_hypotheses=rejected,
            contradictions_resolved=st.contradictions[:],
            unresolved_questions=unresolved_q,
            tools_used=tools_used,
            investigation_summary=summary,
            agent_summary=agent_summary,
            agent_reasoning=agent_reasoning,
        )

    def _record_skill_performance(self):
        """Record the skills used and their effectiveness for learning."""
        st = self.state
        inv = self.investigator

        # Get skills used - prefer from investigator's own tracking
        skills_used = []
        if hasattr(inv, 'skills_used'):
            skills_used = list(inv.skills_used)
        elif hasattr(inv, 'event_log'):
            # Fallback: extract from event log
            for event in inv.event_log:
                if event.get('type') == 'tool' and event.get('name') == 'get_skill':
                    skill_name = event.get('input', {}).get('name')
                    if skill_name and skill_name not in skills_used:
                        skills_used.append(skill_name)

        # Evaluate the case outcome
        eval_result = evaluator.evaluate(self.case, st)
        final_score = eval_result.get('total_score', 0)
        was_successful = eval_result.get('status_match', False)
        actions_used = len([t for t in st.tools_used if t.get('ok')])

        # Record this outcome to the skill performance tracker
        if skills_used:
            skill_performance.record_case_outcome(
                case_id=st.case_id,
                skills_used=skills_used,
                final_score=final_score,
                was_successful=was_successful,
                budget=self.case.budget,
                actions_used=actions_used
            )

    @staticmethod
    def _rejection_reason(h: Hypothesis) -> str:
        if h.contradicting_evidence:
            return "Contradicted by the physical record."
        if not h.supporting_evidence:
            return "No incident-relevant evidence of guilt."
        return "Weaker than the leading hypothesis."

    @staticmethod
    def _hyp_snapshot(st: InvestigationState) -> list[dict[str, Any]]:
        return [
            {
                "id": h.id,
                "suspect": h.suspect,
                "confidence": round(h.confidence, 2),
                "independent_types": h.independent_types,
                "support": len(h.supporting_evidence),
                "contradictions": len(h.contradicting_evidence),
            }
            for h in st.ranked()
        ]
