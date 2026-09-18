"""Model-inference mode: the agent, not the case file, decides what evidence means.

In the authored corpus each evidence item ships `supports` + `weight`, so the engine
sums numbers a human wrote and the model's only real freedom is which question to ask
next. These tests pin the alternative down: on an "inference": "model" case the
attribution is stripped at the tool boundary, hypotheses stay flat until the agent
calls assess_evidence, and the agent's own numbers are what the judge then measures
its conclusion against.

The point is not that the agent scores better this way. It is that a correct verdict
here is attributable to the agent's inference rather than to the case author's.
"""

from __future__ import annotations

import json

import pytest

from src.agent import AgentInvestigator, build_schemas
from src.cases import list_case_files, load_case
from src.investigator import Decision, NextAction
from src.loop import MAX_MODEL_WEIGHT, GameSession
from src.state import BASE_PRIOR
from src.thresholds import CONFIDENCE_THRESHOLD
from src.tools import ToolBox

from tests.test_agent_native import FakeClient, FakeResponse, TextBlock, ToolUseBlock


def expected_confidence(*weights: float) -> float:
    """What the engine should report for these agent-assigned weights.

    Model-inference mode accumulates in LOG-ODDS, not linearly, so this is not
    `BASE_PRIOR + sum(weights)`. The scale is pinned by requiring the linear crossing
    point to survive: weights summing to `CONFIDENCE_THRESHOLD - BASE_PRIOR` still
    land exactly on `CONFIDENCE_THRESHOLD` — see test_crossing_point_is_preserved.

    Spelled out here rather than imported from `loop` on purpose: importing the
    engine's own combiner would turn every assertion below into a tautology.
    """
    import math

    def logit(p: float) -> float:
        return math.log(p / (1.0 - p))

    scale = ((logit(CONFIDENCE_THRESHOLD) - logit(BASE_PRIOR))
             / (CONFIDENCE_THRESHOLD - BASE_PRIOR))
    # Rounded to 4 like the engine reports it: confidence is a displayed quantity,
    # and the contract is the rounded value, not the float that produced it.
    return round(1.0 / (1.0 + math.exp(-(logit(BASE_PRIOR) + sum(weights) * scale))), 4)


def _llm() -> Decision:
    return Decision(action=None, source="llm", llm_called=True, validation_status="ok")


def _session(case_id: str = "case-011") -> GameSession:
    return GameSession(load_case(case_id), requested_mode="llm")


# --- the boundary: authored attribution never reaches the agent ----------------
def test_unlabelled_case_declares_its_mode():
    assert load_case("case-011").inference_mode == "model"
    assert load_case("case-001").inference_mode == "authored"
    # The agent has to be told, or it cannot know weighing is its job.
    assert load_case("case-011").public_briefing()["inference"] == "model"


def test_case_file_carries_no_authored_inference():
    """The file itself is clean — not merely masked at runtime."""
    path = next(p for p in list_case_files()
                if json.loads(p.read_text())["case_id"] == "case-011")
    data = json.loads(path.read_text())

    def walk(node):
        if isinstance(node, dict):
            for item in node.get("evidence", []) or []:
                if isinstance(item, dict):
                    assert "supports" not in item, item
                    assert "weight" not in item, item
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data["tools"])


def test_toolbox_strips_attribution_even_if_a_case_still_has_it():
    """Defense in depth: the stripping is enforced at the tool boundary.

    A half-converted case file must not be able to leak an authored attribution,
    so this drives the check with case-001's fully labelled tables read through a
    case whose declared mode is "model".
    """
    labelled = load_case("case-001")
    labelled._data = dict(labelled._data, inference="model")
    tb = ToolBox(labelled)
    result = tb.call("check_access_log", {"location": "archive_room"})
    assert result["evidence"], "expected the access log to yield evidence"
    for ev in result["evidence"]:
        assert ev.supports is None
        assert ev.weight == 0.0
        assert ev.attribution == "unassessed"
        # What is NOT stripped: the source type. An access log is access-type
        # evidence as a matter of fact, and keeping it authored is what prevents
        # an agent from manufacturing two "independent" lines out of one record.
        assert ev.independent_type == "access"


def test_authored_mode_is_untouched():
    tb = ToolBox(load_case("case-001"))
    evidence = tb.call("check_access_log", {"location": "archive_room"})["evidence"]
    assert any(e.supports == "mara" and e.weight > 0 for e in evidence)
    assert all(e.attribution == "authored" for e in evidence)


def _llm_action(tool: str, args: dict) -> Decision:
    return Decision(action=NextAction("unspecified", "", tool, args),
                    source="llm", llm_called=True, validation_status="ok")


# --- without an assessment, nothing moves --------------------------------------
def test_collecting_evidence_alone_does_not_move_a_hypothesis():
    s = _session()
    s.execute_decision(_llm_action("read_incident_report", {}))
    s.execute_decision(_llm_action("check_access_log", {"location": "archive_room"}))

    assert s.state.evidence_collected, "evidence should still be collected"
    # ...but every suspect is still sitting on the prior. This is the whole point:
    # in this mode the engine contributes no interpretation.
    assert {h.confidence for h in s.state.hypotheses} == {BASE_PRIOR}
    assert [e["id"] for e in s.unassessed_evidence()]


# --- the assessment applies the agent's own numbers ----------------------------
def _collect(s: GameSession) -> None:
    s.execute_decision(_llm_action("read_incident_report", {}))
    s.execute_decision(_llm_action("check_access_log", {"location": "archive_room"}))


def test_assessment_moves_the_hypothesis_by_the_agents_weight():
    s = _session()
    _collect(s)
    entry = s.apply_assessment(
        {"evidence_id": "al_mara_entry", "supports": "mara", "weight": 0.3,
         "rationale": "Her card, inside the window, nobody else's."}, _llm())

    assert entry.tool_ok
    assert entry.tool == "assess_evidence"
    h = s.state.hypothesis_for("mara")
    assert h.confidence == pytest.approx(expected_confidence(0.3))
    assert h.independent_types == ["access"]
    # Attributable, and attributed to the MODEL.
    delta = entry.confidence_deltas[0]
    assert delta.reason == "model_support"
    assert delta.previous == pytest.approx(BASE_PRIOR)
    ev = next(e for e in s.state.evidence_collected if e.id == "al_mara_entry")
    assert ev.attribution == "model"


def test_assessment_costs_no_budget():
    s = _session()
    _collect(s)
    before = s.state.remaining_budget
    s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "mara",
                        "weight": 0.2, "rationale": "x"}, _llm())
    assert s.state.remaining_budget == before


def test_assessment_never_ends_the_run():
    """Cognition must not be able to finish a case.

    Weights alone can satisfy the engine's "solved" test, so if apply_assessment ran
    the stopping conditions an agent could close a case by declaring numbers rather
    than by finding facts. Concluding stays a deliberate submit_conclusion.
    """
    s = _session()
    _collect(s)
    for eid in [e.id for e in s.state.evidence_collected]:
        s.apply_assessment({"evidence_id": eid, "supports": "mara",
                            "weight": MAX_MODEL_WEIGHT, "rationale": "all of it"},
                           _llm())
    assert s.state.status == "investigating"
    assert s.state.final_report is None


def test_zero_weight_discounts_a_red_herring():
    s = _session()
    _collect(s)
    entry = s.apply_assessment(
        {"evidence_id": "al_idris_absent", "supports": "idris", "weight": 0.0,
         "rationale": "Real, but says nothing about the swap."}, _llm())
    assert entry.tool_ok
    assert s.state.hypothesis_for("idris").confidence == pytest.approx(BASE_PRIOR)
    assert entry.confidence_deltas[0].reason == "model_discounted"


def test_negative_weight_is_exculpatory():
    s = _session()
    _collect(s)
    s.apply_assessment({"evidence_id": "al_idris_absent", "supports": "idris",
                        "weight": -0.15, "rationale": "Places him elsewhere."}, _llm())
    assert s.state.hypothesis_for("idris").confidence < BASE_PRIOR


def test_reassessment_withdraws_the_earlier_reading_instead_of_stacking():
    """Re-weighing is legal; double-counting is not."""
    s = _session()
    _collect(s)
    spec = {"evidence_id": "al_mara_entry", "supports": "mara", "weight": 0.3,
            "rationale": "first read"}
    s.apply_assessment(spec, _llm())
    s.apply_assessment({**spec, "weight": 0.1, "rationale": "second read"}, _llm())

    h = s.state.hypothesis_for("mara")
    assert h.confidence == pytest.approx(expected_confidence(0.1))
    assert h.supporting_evidence.count("al_mara_entry") == 1


def test_reattribution_does_not_leave_the_item_on_the_old_suspect():
    s = _session()
    _collect(s)
    s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "mara",
                        "weight": 0.3, "rationale": "her card"}, _llm())
    s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "idris",
                        "weight": 0.3, "rationale": "he cloned her card"}, _llm())

    assert s.state.hypothesis_for("mara").confidence == pytest.approx(BASE_PRIOR)
    assert "al_mara_entry" not in s.state.hypothesis_for("mara").supporting_evidence
    assert s.state.hypothesis_for("idris").confidence == pytest.approx(expected_confidence(0.3))


# --- the engine still bounds the model's inference -----------------------------
@pytest.mark.parametrize("weight", [0.5, -0.9, 1.0])
def test_a_single_item_cannot_be_declared_decisive(weight):
    s = _session()
    _collect(s)
    entry = s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "mara",
                                "weight": weight, "rationale": "case closed"}, _llm())
    assert not entry.tool_ok
    assert str(MAX_MODEL_WEIGHT) in entry.observation
    assert s.state.hypothesis_for("mara").confidence == pytest.approx(BASE_PRIOR)


def test_cannot_weigh_evidence_that_was_never_collected():
    s = _session()
    _collect(s)
    entry = s.apply_assessment({"evidence_id": "invented_id", "supports": "mara",
                                "weight": 0.2, "rationale": "trust me"}, _llm())
    assert not entry.tool_ok
    assert "invented_id" in entry.observation


def test_cannot_weigh_against_a_suspect_who_is_not_in_the_case():
    s = _session()
    _collect(s)
    entry = s.apply_assessment({"evidence_id": "al_mara_entry",
                                "supports": "the_butler", "weight": 0.2,
                                "rationale": "always is"}, _llm())
    assert not entry.tool_ok
    assert "not a suspect" in entry.observation


@pytest.mark.parametrize("weight", ["heavy", None, float("nan")])
def test_non_numeric_weight_is_rejected(weight):
    s = _session()
    _collect(s)
    entry = s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "mara",
                                "weight": weight, "rationale": "x"}, _llm())
    assert not entry.tool_ok


def test_assessment_is_refused_on_an_authored_case():
    """Authored cases already carry weights; letting the agent add its own on top
    would mix two inferences into one number and make the score unattributable."""
    s = _session("case-001")
    _collect(s)
    entry = s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "mara",
                                "weight": 0.2, "rationale": "x"}, _llm())
    assert not entry.tool_ok
    assert "authored weights" in entry.observation


# --- provenance: a saved run must say whose inference it was -------------------
def test_run_metadata_records_the_inference_mode_and_assessment_count():
    s = _session()
    _collect(s)
    s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "mara",
                        "weight": 0.2, "rationale": "x"}, _llm())
    md = s.run_metadata()
    assert md["inference_mode"] == "model"
    assert md["assessments"] == 1
    assert _session("case-001").run_metadata()["inference_mode"] == "authored"


# --- the tool surface the agent actually sees ----------------------------------
def test_assess_evidence_is_offered_to_the_model():
    names = [s["name"] for s in build_schemas()]
    assert "assess_evidence" in names
    schema = next(s for s in build_schemas() if s["name"] == "assess_evidence")
    assert set(schema["input_schema"]["required"]) == {
        "evidence_id", "supports", "weight", "rationale"}


def test_outstanding_ids_travel_with_every_tool_result():
    """The agent must not have to remember ids across a long transcript."""
    case = load_case("case-011")
    inv = AgentInvestigator(case, client=FakeClient([]))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    _collect(s)
    payload = json.loads(inv._last_result_payload())
    assert payload["unassessed_evidence"]
    assert "assess_evidence" in payload["note"]


def test_authored_mode_sends_no_assessment_prompting():
    case = load_case("case-001")
    inv = AgentInvestigator(case, client=FakeClient([]))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    _collect(s)
    payload = json.loads(inv._last_result_payload())
    assert "unassessed_evidence" not in payload


def test_agent_drives_assessment_through_the_engine_end_to_end():
    """The model calls assess_evidence; the engine applies it; the audit shows it."""
    case = load_case("case-011")
    script = [
        FakeResponse([TextBlock("Report first."),
                      ToolUseBlock("read_incident_report", {})]),
        FakeResponse([TextBlock("Who got in?"),
                      ToolUseBlock("check_access_log", {"location": "archive_room"})]),
        # Display-case names must survive normalization to ids.
        FakeResponse([ToolUseBlock("assess_evidence",
                                   {"evidence_id": "al_mara_entry",
                                    "supports": "Mara", "weight": 0.25,
                                    "rationale": "Her card, in the window."})]),
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved",
                                    "reasoning": "One line of evidence is not two.",
                                    "summary": "Only mara's access at 21:17 is "
                                               "established; not enough to accuse."})]),
    ]
    inv = AgentInvestigator(case, client=FakeClient(script))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    while s.state.status == "investigating":
        s.step()

    assessments = [a for a in s.audit if a.tool == "assess_evidence"]
    assert len(assessments) == 1
    assert assessments[0].tool_ok
    assert assessments[0].tool_arguments["supports"] == "mara"  # canonicalized
    assert s.state.hypothesis_for("mara").confidence == pytest.approx(expected_confidence(0.25))
    assert s.run_metadata()["assessments"] == 1


# --- evidence accumulation: no saturation, crossing point intact ---------------
def test_crossing_point_is_preserved():
    """Log-odds replaced the linear sum, but the bar sits in the same place.

    Weights summing to exactly CONFIDENCE_THRESHOLD - BASE_PRIOR must still land on
    CONFIDENCE_THRESHOLD, or every number in thresholds.py quietly means something
    different in model mode than it does in an authored case.
    """
    s = _session()
    _collect(s)
    s.execute_decision(_llm_action("review_camera", {"location": "archive_room"}))
    need = CONFIDENCE_THRESHOLD - BASE_PRIOR
    for eid in ("al_mara_entry", "cam_mara"):
        s.apply_assessment({"evidence_id": eid, "supports": "mara",
                            "weight": need / 2,
                            "rationale": "half of what the bar requires"}, _llm())

    assert s.state.hypothesis_for("mara").confidence == pytest.approx(
        CONFIDENCE_THRESHOLD, abs=1e-4)


def test_evidence_never_stops_moving_the_hypothesis():
    """The saturation bug.

    Under the linear sum, three max-weight items hit the 0.98 clamp and every
    assessment after that was a silent no-op — the agent kept reasoning into a number
    that had stopped responding. Measured on live runs: 5 of 8 assessments moved
    nothing, including the sharpest inference in the trace, purely because it came
    last. Every new item must still move the hypothesis.
    """
    s = _session()
    for tool, args in (("read_incident_report", {}),
                       ("check_access_log", {"location": "archive_room"}),
                       ("review_camera", {"location": "archive_room"}),
                       ("inspect_location", {"location": "archive_room"}),
                       ("analyze_object", {"object": "access_card"}),
                       ("analyze_object", {"object": "manuscript_case"})):
        s.execute_decision(_llm_action(tool, args))

    seen = [BASE_PRIOR]
    for ev in list(s.state.evidence_collected):
        s.apply_assessment({"evidence_id": ev.id, "supports": "mara",
                            "weight": MAX_MODEL_WEIGHT,
                            "rationale": "the maximum the engine allows"}, _llm())
        seen.append(s.state.hypothesis_for("mara").confidence)

    assert len(seen) >= 6, "need more items than the old clamp allowed"
    for previous, current in zip(seen, seen[1:]):
        assert current > previous, (
            f"confidence stalled at {previous}: a later inference changed nothing")
    assert seen[-1] < 1.0, "must never claim absolute certainty"


# --- C: the red herring is the agent's call, not the case author's -------------
def test_agent_may_weigh_an_authored_red_herring_and_be_wrong():
    """`relates_to_incident` used to survive into model mode and hard-gate is_support,
    so the engine silently zeroed whatever weight the agent put on a flagged item —
    and then reported it back as "agent discounted", crediting the agent with a
    judgement it never made. Spotting the red herring is the inference under test, so
    the agent has to own the wrong answer as well as the right one."""
    s = _session()
    s.execute_decision(_llm_action("read_incident_report", {}))
    s.execute_decision(_llm_action("inspect_location", {"location": "archive_room"}))

    herring = next(e for e in s.state.evidence_collected if e.id == "loc_lena_slip")
    assert herring.relates_to_incident is True, "the authored flag must not reach here"

    s.apply_assessment({"evidence_id": "loc_lena_slip", "supports": "lena",
                        "weight": 0.3, "rationale": "agent misreads a stale slip"},
                       _llm())
    assert s.state.hypothesis_for("lena").confidence > BASE_PRIOR, (
        "the agent's reading must be applied, including when it is wrong")

    # ...and weight 0 is still how an agent says "red herring".
    s.apply_assessment({"evidence_id": "loc_lena_slip", "supports": "lena",
                        "weight": 0.0, "rationale": "dated the previous day"}, _llm())
    assert s.state.hypothesis_for("lena").confidence == pytest.approx(BASE_PRIOR)


# --- A: reaching the bar is not the same act as stating a verdict --------------
class _ConcludingDriver:
    """Stand-in for any driver that owns submit_conclusion (the tool_use agent)."""

    name = "llm"
    concludes_explicitly = True
    active_skill = None

    def decide(self, state):  # pragma: no cover - this test drives the engine directly
        raise AssertionError("this test calls execute_decision, not step()")


def _drive_to_the_bar(s: GameSession) -> None:
    """Collect what a conclusion needs, and weigh it to exactly the threshold."""
    for tool, args in (("read_incident_report", {}),
                       ("inspect_location", {"location": "archive_room"}),
                       ("analyze_object", {"object": "access_card"}),
                       ("check_access_log", {"location": "archive_room"}),
                       ("review_camera", {"location": "archive_room"})):
        s.execute_decision(_llm_action(tool, args))
    for eid in ("al_mara_entry", "cam_mara"):
        s.apply_assessment({"evidence_id": eid, "supports": "mara", "weight": 0.25,
                            "rationale": "weighed by the agent"}, _llm())
    for field, value, eid in (("action", "replaced_manuscript", "loc_forgery"),
                              ("method", "copied_access_card", "obj_card")):
        s.establish_fact({"field": field, "value": value, "evidence_id": eid,
                          "rationale": "Inferred from the collected record."}, _llm())


def test_engine_does_not_close_a_case_the_agent_never_concluded():
    """Measured before this existed: 6 of 6 live case-011 runs ended this way — the
    engine hit the thresholds mid-run, synthesized a report, and scored 88-90 while
    `submit_conclusion` was never accepted once and the judge often never ran at all.
    Clearing the bar is a property of the evidence; saying what it means is the
    agent's job, and only the agent can do it."""
    s = GameSession(load_case("case-011"), investigator=_ConcludingDriver(),
                    requested_mode="llm")
    assert s.defer_solved_to_agent is True
    _drive_to_the_bar(s)

    # One more world action: this is the step that used to trip the "solved" branch.
    s.execute_decision(_llm_action("analyze_object", {"object": "manuscript_case"}))

    assert s.state.status == "investigating", "the engine concluded on the agent's behalf"
    assert s.state.conclusion_ready is True, "the agent was not told the bar was met"
    assert s.agent_conclusion is None
    assert s.state.final_report is None


def test_a_driver_without_a_verdict_tool_still_gets_closed_by_the_engine():
    """The rule-based investigator cannot call submit_conclusion, so for it the engine
    must still close the case — otherwise the offline baseline never terminates."""
    s = GameSession(load_case("case-001"))
    assert s.defer_solved_to_agent is False
    s.run()
    assert s.state.status == "solved"


# --- B: judge checks the agent cannot satisfy by declaring numbers -------------
def test_reviewer_can_inspect_attribution_without_a_name_matching_veto():
    """The circularity this closes: every other test in the deterministic gate reads
    the hypothesis ranking, which in this mode the agent itself built. Measured before
    this existed — an agent that weighed two Mara-specific records onto Idris passed
    with `ok=True` and zero issues, and the run convicted the wrong man."""
    from src import judge as judge_mod

    s = _session()
    _collect(s)
    s.execute_decision(_llm_action("review_camera", {"location": "archive_room"}))
    for eid in ("al_mara_entry", "cam_mara"):
        s.apply_assessment({"evidence_id": eid, "supports": "idris", "weight": 0.3,
                            "rationale": "agent misreads the record"}, _llm())

    assert s.state.ranked()[0].suspect == "idris", "the agent did make idris the lead"
    verdict = judge_mod.check_conclusion(
        s.state, {"verdict": "solved", "culprit": "idris", "reasoning": "x"})

    assert verdict["ok"] is False
    digest = judge_mod._evidence_digest(s.state)
    disputed = next(e for e in digest["evidence_on_record"] if e["id"] == "al_mara_entry")
    assert disputed["points_at"] == "idris"
    assert "Mara" in disputed["description"]
    assert disputed["assessment_rationale"] == "agent misreads the record"
    assert not any("names mara" in i.lower() for i in verdict["issues"])


def test_judge_rejects_concluding_while_evidence_sits_unweighed():
    """An unweighed record moves nothing, so concluding over it means resting on a
    record the agent never finished reading."""
    from src import judge as judge_mod

    s = _session()
    _drive_to_the_bar(s)           # weighs 2 of the items it collected, not all
    assert s.unassessed_evidence(), "fixture should leave something unweighed"

    verdict = judge_mod.check_conclusion(
        s.state, {"verdict": "solved", "culprit": "mara", "reasoning": "x"})

    assert verdict["ok"] is False
    assert any("never weighed" in i for i in verdict["issues"]), verdict["issues"]


def test_judge_leaves_authored_cases_alone():
    """These checks are gated on model attribution: an authored run must not start
    failing because the engine, not the agent, assigned the weights."""
    from src import judge as judge_mod

    s = GameSession(load_case("case-001"))
    s.run()
    assert s.state.status == "solved"
    verdict = judge_mod.check_conclusion(
        s.state, {"verdict": "solved", "culprit": s.state.ranked()[0].suspect,
                  "reasoning": "x"})
    assert not any("never weighed" in i for i in verdict["issues"]), verdict["issues"]
