"""Structural integrity of every hand-written case in cases/.

These checks exist because a case can be perfectly valid JSON and still be
unplayable: a required_evidence id that no tool ever emits makes the case
unwinnable, a compare_testimonies key in the wrong order can never be looked
up, and a suspect with no interview entry is a dead end the agent will spend
budget discovering. Each failure below has actually happened.
"""

from __future__ import annotations

import json

import pytest

from src.cases import list_case_files
from src.tools import LOCATION_TOOLS, ToolBox
from src.cases import Case

CASES = [json.loads(p.read_text()) for p in list_case_files()]
IDS = [c["case_id"] for c in CASES]


def _evidence_ids(data: dict) -> set[str]:
    """Every evidence id the tool tables can emit, anywhere in the case."""
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for item in node.get("evidence", []) or []:
                if isinstance(item, dict) and "id" in item:
                    found.add(item["id"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data.get("tools", {}))
    return found


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_required_evidence_exists_somewhere(data):
    """A required id that no tool emits at all makes the case impossible to score fully."""
    missing = sorted(set(data.get("required_evidence", [])) - _evidence_ids(data))
    assert not missing, f"required_evidence not emitted by any tool: {missing}"


def _offerable_evidence_ids(data: dict) -> set[str]:
    """Evidence the agent can actually obtain.

    Stricter than _evidence_ids: affordances only ever offers the three location
    tools for index.location, so anything filed under a secondary location is
    unreachable no matter how well the agent plays. This is the check that matters
    for required_evidence.
    """
    idx, tools = data["index"], data["tools"]
    reachable: list[dict] = []

    def collect(node):
        if isinstance(node, dict):
            reachable.extend(node.get("evidence", []) or [])

    collect(tools.get("read_incident_report", {}))
    for tool in LOCATION_TOOLS:
        collect(tools.get(tool, {}).get(idx.get("location"), {}))
    for obj in idx.get("objects", []):
        collect(tools.get("analyze_object", {}).get(obj, {}))
    for suspect in idx.get("suspects", []):
        collect(tools.get("interview_character", {}).get(suspect, {}))
        collect(tools.get("verify_alibi", {}).get(suspect, {}))
    for key, node in tools.get("compare_testimonies", {}).items():
        if all(part in idx.get("suspects", []) for part in key.split("|")):
            collect(node)

    return {e["id"] for e in reachable if isinstance(e, dict) and "id" in e}


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_required_evidence_is_actually_obtainable(data):
    """Required evidence must sit behind an action the engine will offer."""
    unreachable = sorted(set(data.get("required_evidence", []))
                         - _offerable_evidence_ids(data))
    assert not unreachable, (
        "required_evidence is filed behind targets affordances never offers "
        f"(usually a secondary location): {unreachable}"
    )


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_report_fields_are_pinnable_from_obtainable_evidence(data):
    """In a solvable case, every hidden_truth field the report must fill has to be
    revealed by evidence the agent can reach — otherwise the case caps below full
    marks however well the agent plays.

    Cases marked `solvable: false` are exempt: withholding the facts is the whole
    point of them, and the honest verdict there is `unresolved`.
    """
    if not data.get("evaluation", {}).get("solvable", True):
        pytest.skip("case is deliberately unsolvable; missing facts are by design")
    obtainable = _offerable_evidence_ids(data)
    if data.get("inference") == "model":
        candidates = data["fact_candidates"]
        for field in ("action", "method", "time", "motive"):
            assert data["hidden_truth"][field] in candidates[field]
            assert len(set(candidates[field])) >= 3, "Include genuine alternative values."
        assert obtainable
        return  # Reachable inference paths are exercised in test_llm_flow.py.
    revealed: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for item in node.get("evidence", []) or []:
                if isinstance(item, dict) and item.get("id") in obtainable:
                    revealed.update(item.get("reveals", {}) or {})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data.get("tools", {}))
    # culprit and location come from the hypothesis and the index, not from `reveals`.
    needed = set(data.get("hidden_truth", {})) - {"culprit", "location"}
    assert not needed - revealed, (
        f"no obtainable evidence reveals: {sorted(needed - revealed)}"
    )


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_evidence_ids_are_unique(data):
    """Two evidence items sharing an id would silently collide in state."""
    seen, dupes = set(), set()

    def walk(node):
        if isinstance(node, dict):
            for item in node.get("evidence", []) or []:
                if isinstance(item, dict) and "id" in item:
                    if item["id"] in seen:
                        dupes.add(item["id"])
                    seen.add(item["id"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data.get("tools", {}))
    assert not dupes, f"duplicate evidence ids: {sorted(dupes)}"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_compare_testimonies_keys_are_sorted(data):
    """ToolBox builds this key as '|'.join(sorted([a, b])); an unsorted key is unreachable."""
    bad = [k for k in data["tools"].get("compare_testimonies", {})
           if k != "|".join(sorted(k.split("|")))]
    assert not bad, f"compare_testimonies keys must be alphabetically ordered: {bad}"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_every_suspect_is_a_character(data):
    """A suspect with no character record gets a hypothesis nobody can investigate."""
    chars = {c.get("id") for c in data["index"].get("characters", [])}
    orphans = [s for s in data["index"].get("suspects", []) if s not in chars]
    assert not orphans, f"suspects with no character record: {orphans}"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_every_suspect_is_interviewable(data):
    """affordances offers interview_character for every suspect from the first turn,
    so a suspect with no entry is a guaranteed wasted call.

    verify_alibi is deliberately NOT checked: it only unlocks once a suspect has a
    claim to test, and several cases legitimately leave it out for suspects whose
    account nothing can be checked against.
    """
    gaps = [s for s in data["index"].get("suspects", [])
            if s not in data["tools"].get("interview_character", {})]
    assert not gaps, f"interview_character has no entry for: {gaps}"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_every_listed_object_can_be_analyzed(data):
    """An object in the index with no analyze_object entry only wastes a call."""
    gaps = [o for o in data["index"].get("objects", [])
            if o not in data["tools"].get("analyze_object", {})]
    assert not gaps, f"analyze_object has no entry for: {gaps}"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_primary_location_answers_every_location_tool(data):
    """affordances offers the three location tools for index.location only, so that
    one location must answer all three. Other entries in index.locations are scenery
    for the board view and are never offered as targets."""
    primary = data["index"].get("location")
    assert primary, "index.location is required"
    gaps = [tool for tool in sorted(LOCATION_TOOLS)
            if primary not in data["tools"].get(tool, {})]
    assert not gaps, f"'{primary}' has no entry for: {gaps}"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_hidden_truth_fields_are_scoreable(data):
    """The evaluator scores a fixed field set; extra keys are silently ignored."""
    from src.evaluator import FIELD_WEIGHTS
    allowed = set(FIELD_WEIGHTS) | {"culprit"}
    extra = sorted(set(data.get("hidden_truth", {})) - allowed)
    assert not extra, f"hidden_truth keys the evaluator never reads: {extra}"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_human_played_characters_are_askable(data):
    """A human-played character needs an id (interview_human addresses it) and a role card."""
    for char in data["index"].get("characters", []):
        if not char.get("human_played"):
            continue
        assert char.get("id"), f"human-played character without an id: {char.get('name')}"
        assert char.get("knowledge"), f"{char['id']} has nothing to disclose"
        assert char.get("motivation"), f"{char['id']} has no motivation for the role card"


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_reveals_only_targets_report_fields(data):
    """`reveals` pins fields of the final report; an unknown key pins nothing."""
    fields = {"time", "action", "method", "motive", "location"}

    def walk(node):
        if isinstance(node, dict):
            for item in node.get("evidence", []) or []:
                if isinstance(item, dict):
                    extra = set(item.get("reveals", {}) or {}) - fields
                    assert not extra, f"{item.get('id')} reveals unknown field(s): {sorted(extra)}"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data.get("tools", {}))


@pytest.mark.parametrize("data", CASES, ids=IDS)
def test_human_played_characters_are_reachable_by_the_tool(data):
    """interview_human must accept the human-played ids the case declares."""
    case = Case(data)
    box = ToolBox(case)
    declared = [c["id"] for c in data["index"].get("characters", [])
                if c.get("human_played")]
    assert sorted(box.human_played_suspects()) == sorted(declared)


HUMAN_CASES = [c for c in CASES
               if any(ch.get("human_played") for ch in c["index"].get("characters", []))
               and c.get("evaluation", {}).get("solvable", True)]
HUMAN_IDS = [c["case_id"] for c in HUMAN_CASES]


@pytest.mark.parametrize("data", HUMAN_CASES, ids=HUMAN_IDS)
def test_human_cases_require_model_interpretation(data):
    """Collection alone must no longer solve a human case through preset weights.

    Complete reference routes, with and without a refusing human, are exercised
    through the native agent in test_llm_flow.py. This pins the data boundary.
    """
    from src.investigator import Decision, NextAction
    from src.loop import GameSession
    from src.state import BASE_PRIOR

    case = Case(data)
    assert case.inference_mode == "model"
    session = GameSession(case, requested_mode="llm")
    for tool, args in [("read_incident_report", {}),
                       ("check_access_log", {"location": case.index["location"]})]:
        session.execute_decision(Decision(
            action=NextAction("unspecified", "Collect a record", tool, args),
            source="llm", llm_called=True, validation_status="ok"))
    assert session.unassessed_evidence()
    assert {h.confidence for h in session.state.hypotheses} == {BASE_PRIOR}
    assert not session.state.fact_assertions
    assert not ({"action", "method", "time", "motive"} & session.state.facts_revealed.keys())
