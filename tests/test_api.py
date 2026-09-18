"""API smoke tests: the endpoints the UI relies on, including the boundary that the
hidden truth is only available after completion."""

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_list_cases():
    r = client.get("/api/cases")
    assert r.status_code == 200
    # case-001..013: the hand-written corpus, including the three human-as-actor cases
    # (008/009/010) and unlabelled-inference cases (011–013). Generated cases are
    # deliberately excluded from this listing.
    cases = r.json()["cases"]
    assert len(cases) == 13
    # The listing must say which cases hand inference to the model, so the UI can
    # label them — a run of 011 is not comparable to a run of 001.
    by_id = {c["case_id"]: c for c in cases}
    assert by_id["case-001"]["inference"] == "authored"
    assert by_id["case-011"]["inference"] == "model"
    assert {c["case_id"] for c in cases if c["human_played"]} == {
        "case-008", "case-009", "case-010"}
    assert "knowledge" not in r.text
    assert "secrets" not in r.text


def test_full_flow_and_truth_boundary():
    start = client.post("/api/sessions", json={"case_id": "case-001", "mode": "rule_based"})
    assert start.status_code == 200
    sid = start.json()["session_id"]
    # No hidden solution leaks in the start payload.
    assert "hidden_truth" not in start.text.lower()

    # Evaluation is blocked while investigating.
    assert client.get(f"/api/sessions/{sid}/evaluation").status_code == 409

    # Step at least once, then run to completion.
    client.post(f"/api/sessions/{sid}/step")
    run = client.post(f"/api/sessions/{sid}/run")
    assert run.json()["state"]["status"] == "solved"

    # Now evaluation (and hidden truth) is available.
    ev = client.get(f"/api/sessions/{sid}/evaluation")
    assert ev.status_code == 200
    body = ev.json()
    assert body["hidden_truth"]["culprit"] == "mara"
    assert body["evaluation"]["total_score"] >= 80

    # Trace endpoint has an entry per iteration.
    trace = client.get(f"/api/sessions/{sid}/trace").json()["trace"]
    assert len(trace) >= 5


def test_reset_endpoint_clears_all_prior_data():
    sid = client.post("/api/sessions", json={"case_id": "case-002", "mode": "rule_based"}).json()["session_id"]
    client.post(f"/api/sessions/{sid}/run")
    reset = client.post(f"/api/sessions/{sid}/reset").json()
    assert reset["state"]["status"] == "investigating"
    assert reset["state"]["iteration"] == 0
    assert reset["run_metadata"]["decisions"] == 0
    # audit/evaluation gone: trace empty, evaluation blocked again
    assert client.get(f"/api/sessions/{sid}/trace").json()["trace"] == []
    assert client.get(f"/api/sessions/{sid}/evaluation").status_code == 409


def test_evaluation_includes_run_metadata():
    # Pin the mode explicitly: with an ANTHROPIC_API_KEY present in the environment the
    # default ("") auto-resolves to "llm", which would make this a slow, live LLM run
    # and break the rule_based assertion below. The test's intent is run_metadata shape.
    sid = client.post("/api/sessions",
                      json={"case_id": "case-001", "mode": "rule_based"}).json()["session_id"]
    client.post(f"/api/sessions/{sid}/run")
    ev = client.get(f"/api/sessions/{sid}/evaluation").json()
    assert ev["evaluation"]["run_metadata"]["requested_mode"] == "rule_based"
    assert ev["evaluation"]["llm_integrity_ok"] is True


@pytest.mark.parametrize("mode", ["llm", ""])
def test_llm_mode_without_key_errors_cleanly(monkeypatch, mode):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/api/sessions", json={"case_id": "case-011", "mode": mode})
    assert r.status_code == 400
    assert "ANTHROPIC_API_KEY" in r.text


# --- human-in-the-loop handoff -------------------------------------------------

def test_views_report_not_awaiting_when_no_human_is_involved():
    """Every view carries the flag, so the client never has to guess."""
    start = client.post("/api/sessions",
                        json={"case_id": "case-001", "mode": "rule_based"}).json()
    assert start["awaiting_human"] is False
    assert start["pending_question"] is None


def test_answer_is_refused_when_nothing_was_asked():
    sid = client.post("/api/sessions",
                      json={"case_id": "case-001", "mode": "rule_based"}).json()["session_id"]

    r = client.post(f"/api/sessions/{sid}/answer", json={"answer": "I did it"})

    assert r.status_code == 409
    assert "not awaiting" in r.text.lower()


def test_full_interview_round_trip_through_the_api(monkeypatch):
    """Park on a question, answer it from the client, and see the run resume.

    This is the whole feature end to end: the answer enters through /answer and
    nowhere else, and the same person's second answer is checked against the first.
    """
    import app as app_mod
    from src.investigator import Decision, Investigator, NextAction

    QUESTIONS = [("When did you last see Rhodes?", "timing"),
                 ("You're sure about that?", "timing")]

    class AsksPike(Investigator):
        """Scripted, and deliberately STATELESS.

        Sessions are serialized between requests, and the codec preserves the native
        agent's conversation but not a custom investigator's private attributes — a
        driver that popped from an instance list would silently restart its script on
        every request. Deriving the next move from InvestigationState is what makes a
        scripted driver survive a rehydration.
        """

        name = "scripted"

        def __init__(self, toolbox=None):
            pass

        def decide(self, state):
            if not state.index:
                return Decision(action=NextAction("t", "q", "read_incident_report", {}),
                                source="rule_based", validation_status="ok")
            asked = sum(1 for t in state.tools_used
                        if t["tool"] == "interview_human")
            if asked >= len(QUESTIONS):
                return Decision(action=None, source="rule_based", validation_status="ok",
                                should_stop=True, stop_reason="done")
            q, topic = QUESTIONS[asked]
            return Decision(
                action=NextAction("statement_validation", q, "interview_human",
                                  {"suspect": "pike", "question": q, "topic": topic}),
                source="rule_based", validation_status="ok")

    # Patch the builder, not one live object: the session is rebuilt from the store on
    # every request, so the driver has to be attached wherever that rebuild happens.
    real_build = app_mod._build_session

    def build_with_scripted(case, mode, policy):
        session = real_build(case, mode, policy)
        session.investigator = AsksPike()
        return session

    monkeypatch.setattr(app_mod, "_build_session", build_with_scripted)

    sid = client.post("/api/sessions",
                      json={"case_id": "case-008", "mode": "rule_based"}).json()["session_id"]

    client.post(f"/api/sessions/{sid}/step")            # read the report
    parked = client.post(f"/api/sessions/{sid}/step").json()

    assert parked["awaiting_human"] is True
    assert client.get(f"/api/sessions/{sid}/evaluation").status_code == 409
    q = parked["pending_question"]
    assert q["character_name"] == "Pike"
    assert q["question"] == "When did you last see Rhodes?"
    assert "WHAT YOU KNOW:" in q["role_card"]
    # The player is told their secrets; the agent is not.
    assert "told_to_forget" in q["role_card"]

    # Stepping is refused while parked — answering is the only way forward.
    assert client.post(f"/api/sessions/{sid}/step").status_code == 409

    resumed = client.post(f"/api/sessions/{sid}/answer",
                          json={"answer": "About quarter past three."}).json()
    assert resumed["awaiting_human"] is False
    assert "About quarter past three." in resumed["latest"]["observation"]

    # Ask again, then contradict the earlier answer.
    client.post(f"/api/sessions/{sid}/step")
    shifted = client.post(f"/api/sessions/{sid}/answer",
                          json={"answer": "Actually it was after four."}).json()

    obs = shifted["latest"]["observation"]
    assert "conflicts with what" in obs           # the agent is told the story moved
    assert "likely_lying" not in obs              # but never handed the verdict


def test_answer_requires_non_empty_text(monkeypatch):
    """An empty answer is rejected before it can be mistaken for testimony."""
    from src.human_responder import PendingResponder
    from src.loop import GameSession
    import app as app_mod

    sid = client.post("/api/sessions",
                      json={"case_id": "case-001", "mode": "rule_based"}).json()["session_id"]
    # Park the session by hand — driving a real LLM here would make the test live.
    # The mutation has to go back to the store, because the next request rebuilds
    # the session from there rather than from a live object.
    entry = app_mod._get(sid)
    session: GameSession = entry["session"]
    session.state.status = "awaiting_human"
    session.state.pending_human = {"suspect": "butler", "question": "Where were you?",
                                   "role_card": "ROLE: ...", "character_name": "Edmund"}
    app_mod._commit(entry)

    blank = client.post(f"/api/sessions/{sid}/answer", json={"answer": "   "})
    assert blank.status_code == 400

    # And the parked question is exposed to the UI meanwhile.
    view = client.post(f"/api/sessions/{sid}/step")
    assert view.status_code == 409
    assert "provide_human_answer" in view.text
