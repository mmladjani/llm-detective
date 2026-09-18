"""LLM-path regressions for conclusion review, inference and ordered events.

Scripted clients verify protocol and state, not live model reasoning quality.
"""
import copy
import json

import pytest

from src.agent import AgentInvestigator
from src.cases import Case, load_case
from src.evaluator import evaluate
from src.investigator import Decision
from src.judge import _evidence_digest, check_conclusion, make_llm_conclusion_judge
from src.loop import GameSession
from src.session_codec import dump_session, load_session
from tests.test_agent_native import FakeClient, FakeResponse, TextBlock, ToolUseBlock
from tests.test_model_inference import _drive_to_the_bar, _llm, _llm_action


def session(script=(), reviewer=None):
    case = load_case("case-011")
    inv = AgentInvestigator(case, client=FakeClient(script), conclusion_judge=reviewer)
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    return s, inv


def submit(culprit="mara", verdict="solved"):
    return FakeResponse([ToolUseBlock("submit_conclusion", {
        "verdict": verdict, "culprit": culprit, "reasoning": "Proposed inference.",
    })])


def ready(s):
    _drive_to_the_bar(s)
    for ev in s.unassessed_evidence():
        s.apply_assessment({"evidence_id": ev["id"], "supports": "mara", "weight": 0,
                            "rationale": "Not used to identify the culprit."}, _llm())


def test_rejected_verdict_can_never_become_solved_at_budget_zero():
    """The engine previously overrode the review and synthesized a solved accusation."""
    s, inv = session([submit() for _ in range(3)],
                     reviewer=lambda *_: {"ok": False, "issues": ["Unsupported inference."]})
    ready(s)
    s.state.remaining_budget = 0
    s.step()
    assert inv.final_proposal["judge"]["ok"] is False
    assert s.state.status == "conclusion_rejected"
    assert s.state.final_report.culprit is None
    assert evaluate(s.case, s.state)["total_score"] == 0


def test_prose_stop_without_a_review_is_not_a_solve():
    """Clearing numeric thresholds alone used to let an agent bypass submit_conclusion."""
    s, _ = session([FakeResponse([TextBlock("Finished.")]) for _ in range(2)])
    ready(s)
    s.step()
    assert s.agent_conclusion is None
    assert s.state.status == "execution_error"


def test_accepted_abstention_is_respected_even_with_a_high_numeric_lead():
    s, _ = session([submit(None, "unresolved")],
                   reviewer=lambda *_: {"ok": True, "issues": []})
    ready(s)
    s.step()
    assert s.state.status == "unresolved"
    assert s.state.final_report.culprit is None


def test_report_and_rejected_hypotheses_follow_the_accepted_suspect():
    """Fields, summary and rejected list must all describe the same verdict."""
    s, _ = session()
    ready(s)
    assert s.state.ranked()[0].suspect == "mara"
    s.record_agent_conclusion({"verdict": "solved", "culprit": "idris"}, {"ok": True})
    s.execute_decision(Decision(action=None, source="llm", should_stop=True))
    report = s.state.final_report
    assert report.culprit == "idris"
    assert report.confidence == 0.2
    assert report.key_evidence == []
    assert report.investigation_summary.startswith("idris ")
    assert "idris" not in {h["suspect"] for h in report.rejected_hypotheses}
    assert "mara" in {h["suspect"] for h in report.rejected_hypotheses}
    result = evaluate(s.case, s.state)
    assert result["culprit_correct"] is False
    assert result["status_match"] is False
    assert result["total_score"] <= 15


@pytest.mark.parametrize("rationale", ["", "   ", None])
def test_empty_assessment_rationale_is_rejected(rationale):
    """An empty rationale cannot masquerade as an explained inference."""
    s, _ = session()
    ready(s)
    entry = s.apply_assessment({"evidence_id": "cam_mara", "supports": "mara",
                               "weight": 0.25, "rationale": rationale}, _llm())
    assert entry.tool_ok is False
    assert "rationale" in entry.observation


def test_model_facts_are_inferred_and_revisable_not_revealed():
    """Gathering a record must not silently fill the report's answers."""
    s, _ = session()
    s.execute_decision(_llm_action("read_incident_report", {}))
    s.execute_decision(_llm_action("inspect_location", {"location": "archive_room"}))
    assert s.state.facts_revealed == {}
    budget = s.state.remaining_budget
    spec = {"field": "action", "value": "stole_manuscript", "evidence_id": "loc_forgery",
            "rationale": "First interpretation."}
    assert s.establish_fact(spec, _llm()).tool_ok
    assert s.establish_fact({**spec, "value": "replaced_manuscript",
                            "rationale": "The original was replaced, not merely removed."}, _llm()).tool_ok
    assert s.state.facts_revealed["action"] == "replaced_manuscript"
    assert s.state.fact_assertions["action"]["evidence_id"] == "loc_forgery"
    assert not any("stole_manuscript" in f for f in s.state.known_facts)
    assert s.state.remaining_budget == budget


@pytest.mark.parametrize("update", [{"value": "invented"}, {"field": "culprit"},
                                    {"evidence_id": "unseen"}, {"rationale": " "}])
def test_invalid_fact_does_not_change_state(update):
    s, _ = session()
    ready(s)
    before = dict(s.state.facts_revealed)
    entry = s.establish_fact({"field": "method", "value": "copied_access_card",
                              "evidence_id": "obj_card", "rationale": "Copy trace.", **update}, _llm())
    assert not entry.tool_ok
    assert s.state.facts_revealed == before


def test_reviewer_receives_all_observations_assertions_and_summary():
    """A true tool observation was called invented because the reviewer never saw it."""
    s, _ = session()
    ready(s)
    client = FakeClient([FakeResponse([TextBlock('{"findings": []}')])])
    judge = make_llm_conclusion_judge(client, "test")
    judge(s.state, {"verdict": "solved", "culprit": "mara", "summary": "Review this too."})
    payload = json.loads(client.messages.calls[0]["messages"][0]["content"])
    record = payload["evidence_record"]
    assert "two buildings five minutes apart" in json.dumps(record["observations"])
    assert record["agent_fact_assertions"]["method"]["evidence_id"] == "obj_card"
    assert payload["proposed_conclusion"]["summary"] == "Review this too."
    assert "hidden_truth" not in json.dumps(payload)


def test_web_default_and_rehydrated_reviewer(monkeypatch):
    """Web used to promise two layers but instantiate only the deterministic gate."""
    import app
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-placeholder")
    monkeypatch.setattr("anthropic.Anthropic", lambda: FakeClient([]))
    assert app._resolve_mode("") == "llm"
    s = app._build_session(load_case("case-011"), "llm", "demo")
    assert callable(s.investigator._conclusion_judge)
    ready(s)
    payload = dump_session(s, mode="llm", policy="demo", case_is_custom=False)
    restored, _ = load_session(payload, app._build_session)
    assert callable(restored.investigator._conclusion_judge)
    assert restored.state.observations == s.state.observations
    assert restored.state.fact_assertions == s.state.fact_assertions
    assert restored.events == s.events
    view = app._view({"session": restored})
    assert any(e.get("name") == "establish_fact" for e in view["events"])
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert app._resolve_mode("") == "llm", "No silent offline fallback."


def test_critique_audit_is_visible_and_survives_session_roundtrip():
    import app
    audit = [{"index": 0, "disposition": "withdraw", "reason": "Recorded mechanism.",
              "record_quote": "The archive entry at 21:17 used the duplicate",
              "original_finding": "No duplicate was recorded."}]
    s, inv = session([submit()], reviewer=lambda *_: {
        "ok": True, "issues": [], "finding_audit": audit,
        "advisories": ["Review correction: Recorded mechanism."]})
    ready(s)
    s.step()
    assert s.state.status == "solved"
    assert inv.final_proposal["judge"]["finding_audit"] == audit
    gate = next(e for e in app._view({"session": s})["events"]
                if e.get("phase") == "judge")
    assert gate["finding_audit"] == audit
    payload = json.loads(json.dumps(dump_session(s, mode="llm", policy="demo",
                                                case_is_custom=False)))
    restored, _ = load_session(payload, lambda *_: session()[0])
    assert restored.conclusion_review["finding_audit"] == audit
    assert restored.events == s.events


def test_free_inferences_appear_before_the_next_world_action():
    """A /step previously displayed only its final world action, hiding the reasoning tools."""
    import app
    s, inv = session()
    ready(s)
    inv._client = FakeClient([
        FakeResponse([ToolUseBlock("assess_evidence", {"evidence_id": "cam_mara",
             "supports": "mara", "weight": 0.3, "rationale": "Reconsidered camera."})]),
        FakeResponse([ToolUseBlock("establish_fact", {"field": "time", "value": "21:17",
             "evidence_id": "al_mara_entry", "rationale": "Entry record."})]),
        FakeResponse([ToolUseBlock("analyze_object", {"object": "manuscript_case"})]),
    ])
    offset = len(s.events)
    latest = s.step()
    view = app._view({"session": s}, latest=latest)
    assert [e["name"] for e in view["events"][offset:]] == [
        "assess_evidence", "establish_fact", "analyze_object"]


def test_naive_baseline_has_scores_and_does_not_read_truth():
    """Compare on unlabelled cases, with no oracle or fake LLM provenance."""
    from eval.model_baseline import run_naive_model_baseline
    from eval.run_all import run_once_baseline
    for cid in ("case-011", "case-012", "case-013"):
        result = run_once_baseline(cid)
        assert isinstance(result["score"], int)
        assert result["fully_llm_driven"] is False
        assert result["baseline_policy"] == "uniform_lexical"
    data = copy.deepcopy(load_case("case-013")._data)
    normal = run_naive_model_baseline(Case(data))
    data["hidden_truth"] = {key: "POISON" for key in data["hidden_truth"]}
    changed = run_naive_model_baseline(Case(data))
    assert normal.state.final_report == changed.state.final_report


@pytest.mark.parametrize("findings", [[{}], [{"type": "advisory", "detail": " "}], [None]])
def test_malformed_review_cannot_accept_a_conclusion(findings):
    s, _ = session()
    client = FakeClient([FakeResponse([TextBlock(json.dumps({"findings": findings}))])])
    result = make_llm_conclusion_judge(client, "test")(s.state, {})
    assert result["ok"] is False
    assert result["error"] == "unparseable_review"


def test_model_gate_does_not_require_the_numeric_leader():
    s, _ = session()
    ready(s)
    for eid in ("cam_mara", "obj_card"):
        s.apply_assessment({"evidence_id": eid, "supports": "idris", "weight": 0.01,
                            "rationale": "Attribution must be checked by the semantic reviewer."}, _llm())
    assert s.state.ranked()[0].suspect == "mara"
    gate = check_conclusion(s.state, {"verdict": "solved", "culprit": "idris"})
    assert gate["ok"], gate
    # Structural acceptance is NOT truth certification: the LLM reviewer still runs.


def test_last_world_result_can_be_assessed_before_an_explicit_verdict():
    s, _ = session([
        FakeResponse([ToolUseBlock("analyze_object", {"object": "manuscript_case"})]),
        FakeResponse([ToolUseBlock("assess_evidence", {"evidence_id": "obj_case",
            "supports": "mara", "weight": 0.2, "rationale": "Fibers corroborate camera identification."})]),
        submit(),
    ], reviewer=lambda *_: {"ok": True, "issues": []})
    ready(s)
    s.state.remaining_budget = 1
    s.step()
    assert s.state.remaining_budget == 0
    assert s.state.status == "investigating"
    s.step()
    assert s.state.status == "solved"
    assert s.conclusion_review["ok"]
    blob = dump_session(s, mode="llm", policy="demo", case_is_custom=False)
    restored, _ = load_session(blob, lambda *_: session()[0])
    assert restored.conclusion_review == s.conclusion_review
    assert restored.state.final_report == s.state.final_report
    assert restored.events == s.events


def test_new_unlabelled_record_restores_revision_allowance_at_zero_budget():
    s, inv = session([
        FakeResponse([ToolUseBlock("analyze_object", {"object": "manuscript_case"})]),
        submit(),  # Intentionally premature: obj_case still needs an assessment.
        FakeResponse([ToolUseBlock("assess_evidence", {"evidence_id": "obj_case",
            "supports": "mara", "weight": 0.2, "rationale": "Corroborates the camera."})]),
        submit(),
    ], reviewer=lambda *_: {"ok": True, "issues": []})
    ready(s)
    inv.revises_left = 0
    s.state.remaining_budget = 1
    entry = s.step()
    assert "Collected unassessed record: obj_case." in entry.state_changes
    assert s.state.remaining_budget == 0
    s.step()
    assert s.state.status == "solved"
    assert [e["passed"] for e in s.events if e.get("phase") == "judge"] == [False, True]


def test_exhausted_reviews_cannot_be_rerolled_without_new_evidence():
    calls = []

    def reviewer(state, proposal):
        calls.append(proposal)
        return {"ok": len(calls) > 1, "issues": ["Collect corroboration."] if len(calls) == 1 else []}

    s, inv = session([submit(), submit(),
        FakeResponse([ToolUseBlock("analyze_object", {"object": "manuscript_case"})])], reviewer)
    ready(s)
    inv.revises_left = 0  # This submission is the last permitted review.
    entry = s.step()
    assert entry.tool == "analyze_object", "The second submission bypassed the exhausted limit."
    assert len(calls) == 1
    assert inv.revises_left == -1
    blob = dump_session(s, mode="llm", policy="demo", case_is_custom=False)
    next_script = [FakeResponse([ToolUseBlock("assess_evidence", {
        "evidence_id": "obj_case", "supports": "mara", "weight": 0.2,
        "rationale": "New physical corroboration."})]), submit()]
    restored, _ = load_session(blob, lambda *_: session(next_script, reviewer)[0])
    assert restored.investigator.revises_left == -1
    restored.step()  # The pending new record restores the allowance after reload.
    assert len(calls) == 2
    assert restored.state.status == "solved"


def test_oracle_data_never_enters_native_or_reviewer_messages():
    """Poison engine-only fields: checking just the hidden_truth key misses value leaks."""
    data = copy.deepcopy(load_case("case-013")._data)
    data["hidden_truth"] = {k: "SECRET_ORACLE" for k in data["hidden_truth"]}
    data["required_evidence"] = ["SECRET_REQUIREMENT"]
    data["evaluation"] = {"expected_status": "SECRET_EXPECTATION"}
    case = Case(data)
    client = FakeClient([FakeResponse([ToolUseBlock("read_incident_report", {})]),
                         submit(None, "unresolved")])
    reviewer = FakeClient([FakeResponse([TextBlock('{"findings": []}')])])
    inv = AgentInvestigator(case, client=client,
                           conclusion_judge=make_llm_conclusion_judge(reviewer, "test"))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    s.run()
    assert s.state.status == "unresolved"
    messages = dump_session(s, mode="llm", policy="demo", case_is_custom=False)["agent"]["messages"]
    assert "SECRET_" not in json.dumps(messages)
    assert "SECRET_" not in json.dumps(reviewer.messages.calls)


@pytest.mark.parametrize("cid,witness,location,secret", [
    ("case-008", "pike", "dispensary", "told_to_forget"),
    ("case-009", "emma", "display_room", "you_do_not_want_this"),
    ("case-010", "sarah", "archive_room", "you_feel_responsible"),
])
def test_native_human_interview_survives_reload_and_only_sends_the_answer(cid, witness, location, secret):
    """The real UI parks between HTTP requests; the native transcript must resume too."""
    case = load_case(cid)
    client = FakeClient([
        FakeResponse([ToolUseBlock("read_incident_report", {})]),
        FakeResponse([ToolUseBlock("interview_human", {
            "suspect": witness, "question": "What did you see?", "topic": "observations"})]),
    ])
    inv = AgentInvestigator(case, client=client)
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    s.step()
    before = s.state.remaining_budget
    s.step()
    assert s.awaiting_human
    assert s.state.remaining_budget == before
    assert "YOUR SECRETS" in s.pending_question()["role_card"]
    blob = dump_session(s, mode="llm", policy="demo", case_is_custom=False)
    resumed_client = FakeClient([FakeResponse([ToolUseBlock(
        "check_access_log", {"location": location})])])

    def rebuild(case, mode, policy):
        driver = AgentInvestigator(case, client=resumed_client)
        restored = GameSession(case, investigator=driver, requested_mode=mode)
        driver.session = restored
        return restored

    restored, _ = load_session(blob, rebuild)
    restored.provide_human_answer("I do not want to answer that question.")
    assert restored.state.remaining_budget == before - 1
    assert len([a for a in restored.audit if a.tool == "interview_human"]) == 1
    restored.step()
    messages = json.dumps(resumed_client.messages.calls[0]["messages"], default=vars)
    assert "I do not want to answer that question." in messages
    assert "YOUR SECRETS" not in messages
    assert secret not in messages
    assert "consistency_score" not in messages
    assert restored.state.inference_mode == "model"


# Explicit reference routes, not an agent policy and not a live reasoning benchmark.
# They prove the world + tool contract can yield every scored fact inside the budget.
REFERENCE_CASES = [
    ("case-008", "rhodes", "dispensary", [
        ("analyze_object", {"object": "override_key"}),
        ("analyze_object", {"object": "drug_register"}),
    ], {"al_rhodes_override", "cam_rhodes", "obj_override", "loc_clinic_slip"}, {
        "action": ("removed_controlled_drugs", "obj_register"),
        "method": ("used_service_override", "obj_override"),
        "motive": ("supply_private_clinic", "loc_clinic_slip"),
        "time": ("03:20", "al_rhodes_override"),
    }),
    ("case-009", "davis", "display_room", [
        ("analyze_object", {"object": "case_key"}),
        ("analyze_object", {"object": "stock_record"}),
    ], {"al_davis_alone", "cam_davis_key", "obj_key_davis", "loc_lender_letter"}, {
        "action": ("removed_pendant", "obj_stock_gap"),
        "method": ("used_display_key", "obj_key_davis"),
        "motive": ("settle_gambling_debt", "loc_lender_letter"),
        "time": ("11:45", "cam_davis_key"),
    }),
    ("case-010", "carmen", "archive_room", [
        ("analyze_object", {"object": "damaged_records"}),
    ], {"log_carmen_key", "cam_carmen_boxes", "obj_fragments_name", "obj_damage_deliberate"}, {
        "action": ("destroyed_records", "obj_damage_deliberate"),
        "method": ("discarded_in_bins", "bin_lid_record"),
        "motive": ("hide_family_history", "obj_fragments_name"),
        "time": ("18:00", "bin_lid_record"),
    }),
    ("case-011", "mara", "archive_room", [
        ("analyze_object", {"object": "access_card"}),
        ("analyze_object", {"object": "manuscript_case"}),
    ], {"al_mara_entry", "cam_mara", "obj_card", "obj_case"}, {
        "action": ("replaced_manuscript", "loc_forgery"),
        "method": ("copied_access_card", "obj_card"),
        "motive": ("hide_forgery", "loc_forgery"),
        "time": ("21:17", "al_mara_entry"),
    }),
    ("case-012", "idris", "server_room", [
        ("verify_alibi", {"suspect": "tomas"}),
        ("analyze_object", {"object": "usb_drive"}),
    ], {"al_idris", "cam_idris", "obj_usb", "loc_broker"}, {
        "action": ("copied_database", "obj_usb"),
        "method": ("usb_drive", "obj_usb"),
        "motive": ("sell_data", "loc_broker"),
        "time": ("02:44", "obj_usb"),
    }),
    ("case-013", "blair", "lab", [
        ("interview_character", {"suspect": "alex"}),
        ("verify_alibi", {"suspect": "alex"}),
        ("analyze_object", {"object": "badge_reader"}),
        ("analyze_object", {"object": "delivery_crate"}),
    ], {"cam_courier", "reader_serial", "crate_receipt", "buyer_offer"}, {
        "action": ("stole_prototype", "empty_mount"),
        "method": ("stolen_badge", "reader_serial"),
        "motive": ("sell_prototype", "buyer_offer"),
        "time": ("18:22", "crate_receipt"),
    }),
]


@pytest.mark.parametrize("cid,culprit,location,extra,supports,facts,human_answer", [
    (*case, answer) for case in REFERENCE_CASES
    for answer in ([None, "I saw nothing at all."] if case[0] in {"case-008", "case-009", "case-010"} else [None])
])
def test_model_case_has_a_complete_legal_native_inference_path(cid, culprit, location, extra, supports, facts, human_answer):
    from src.tools import ToolBox
    from src.human_responder import ScriptedResponder
    case = load_case(cid)
    script = []

    def call(name, args):
        script.append(FakeResponse([ToolUseBlock(name, args)]))

    path = [("read_incident_report", {}), ("check_access_log", {"location": location}),
            ("review_camera", {"location": location}),
            ("inspect_location", {"location": location}), *extra]
    for tool, args in path:
        call(tool, args)
        if tool == "read_incident_report" and human_answer is not None:
            witness = next(c["id"] for c in case.characters_full() if c.get("human_played"))
            call("interview_human", {"suspect": witness, "question": "What did you see?"})
        for ev in ToolBox(case).call(tool, args)["evidence"]:
            accused = "alex" if ev.id == "badge_alex" else culprit
            call("assess_evidence", {"evidence_id": ev.id, "supports": accused,
                 "weight": 0.3 if ev.id == "badge_alex" else 0.2 if ev.id in supports else 0,
                 "rationale": ev.description})
        if cid == "case-013" and tool == "check_access_log":
            call("submit_conclusion", {"verdict": "solved", "culprit": "alex",
                                       "reasoning": "The badge belongs to Alex."})
    if cid == "case-013":
        call("assess_evidence", {"evidence_id": "badge_alex", "supports": "blair", "weight": 0.15,
             "rationale": "reader_serial identifies Blair as the user; Alex is only the badge owner."})
    for field, (value, eid) in facts.items():
        call("establish_fact", {"field": field, "value": value, "evidence_id": eid,
                                "rationale": f"Reference inference from {eid}."})
    call("submit_conclusion", {"verdict": "solved", "culprit": culprit,
         "reasoning": "The cited independent records support this reference solution.",
         "summary": f"The collected record supports {culprit}."})
    reviewer_client = FakeClient([FakeResponse([TextBlock('{"findings": []}')])])
    inv = AgentInvestigator(case, client=FakeClient(script),
                           conclusion_judge=make_llm_conclusion_judge(reviewer_client, "test"))
    responder = ScriptedResponder([human_answer] if human_answer else [])
    s = GameSession(case, investigator=inv, requested_mode="llm", responder=responder)
    inv.session = s
    s.run()
    assert s.state.status == "solved", [(e.tool, e.observation) for e in s.audit if not e.tool_ok]
    assert s.state.final_report.culprit == culprit
    assert not [e for e in s.audit if not e.tool_ok]
    assert s.state.remaining_budget >= 0
    result = evaluate(case, s.state, s.run_metadata())
    assert all(result["correct"].values()), result
    assert not result["missed_required_evidence"]
    assert reviewer_client.messages.calls
    assert len(responder.asked) == (1 if human_answer is not None else 0)
    if cid == "case-013":
        gates = [e for e in s.events if e.get("phase") == "judge"]
        assert [e["passed"] for e in gates] == [False, True]
        assert any(e.get("phase") == "revise" for e in s.events)
        assert "badge_alex" not in s.state.hypothesis_for("alex").supporting_evidence
        assert "badge_alex" in s.state.hypothesis_for("blair").supporting_evidence
    # Both model payloads and the judge only receive discovered records, never an oracle.
    assert "hidden_truth" not in json.dumps(reviewer_client.messages.calls)
