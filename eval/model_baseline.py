"""An intentionally naive, reproducible comparator on the SAME unlabelled world.

Fixed tool policy, uniform positive weights for the first named suspect, lexical
candidate matching for facts. It never reads hidden_truth or authored evidence labels.
This is a baseline, not a model run and not evidence of LLM reasoning quality.
"""
import re

from src.cases import Case
from src.investigator import Decision
from src.loop import GameSession


def run_naive_model_baseline(case: Case) -> GameSession:
    session = GameSession(case, requested_mode="rule_based")
    provenance = Decision(action=None, source="rule_based")
    while session.state.status == "investigating":
        session.step()
        if session.state.status != "investigating":
            break
        suspects = session.state.index.get("suspects", [])
        for item in session.unassessed_evidence():
            named = [s for s in suspects if re.search(rf"\b{re.escape(s)}\b", item["description"], re.I)]
            session.apply_assessment({
                "evidence_id": item["id"], "supports": (named or suspects)[0],
                "weight": 0.2 if named else 0,
                "rationale": "Baseline: first named suspect, uniform weight; no semantic analysis.",
            }, provenance)
        records = session.state.evidence_collected
        if not records:
            continue
        for field, candidates in case.fact_candidates.items():
            match = next(((value, record) for record in records for value in candidates
                          if all(word in record.description.lower()
                                 for word in value.replace('_', ' ').split())), None)
            value, record = match or (candidates[0], records[0])
            if session.state.facts_revealed.get(field) != value:
                session.establish_fact({"field": field, "value": value,
                                        "evidence_id": record.id,
                                        "rationale": "Baseline: literal token match, otherwise first candidate."},
                                       provenance)
    return session
