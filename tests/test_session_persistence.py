"""Sessions survive between requests, which is what makes serverless hosting possible.

The app used to hold live GameSession objects in a module dict. On Vercel a second
request may land on a different instance, so the investigation has to be serialized
after each turn and rebuilt before the next. These tests pin the two things that can
go wrong quietly:

  * the round trip loses something the engine, judge or evaluator reads later, so a
    resumed run scores differently from an uninterrupted one;
  * the serialized blob (which contains a custom case's hidden truth) escapes into a
    response.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app as app_mod
from src import session_codec, session_store
from src.agent import AgentInvestigator
from src.cases import load_case
from src.investigator import Decision, NextAction
from src.loop import GameSession
from src.state import BASE_PRIOR
from tests.test_model_inference import expected_confidence

from tests.test_agent_native import FakeClient, FakeResponse, TextBlock, ToolUseBlock

client = TestClient(app_mod.app)


def _offline_build(case, mode, policy):
    """Stand-in for app._build_session that needs no API key.

    Rehydrating an `llm` session constructs a fresh AgentInvestigator, and the real
    builder refuses to do that without ANTHROPIC_API_KEY — correct in production,
    useless in a hermetic test. Everything under test (the message list, the state,
    the audit) is independent of which client is attached.
    """
    if mode == "llm":
        inv = AgentInvestigator(case, client=FakeClient([]))
        session = GameSession(case, investigator=inv, requested_mode="llm")
        inv.session = session
        return session
    return app_mod._build_session(case, mode, policy)


def _roundtrip(session, mode="rule_based", policy="demo", custom=False):
    blob = session_codec.dump_session(session, mode=mode, policy=policy,
                                      case_is_custom=custom)
    # Must survive a real JSON encode: the store writes a string, so an object that
    # only *looks* serializable fails in production and not here.
    blob = json.loads(json.dumps(blob))
    restored, _ = session_codec.load_session(blob, _offline_build)
    return restored


# --- the store backends --------------------------------------------------------
def test_memory_store_roundtrip_and_delete():
    store = session_store.MemoryStore()
    assert store.get("missing") is None
    store.put("k", {"a": 1})
    assert store.get("k") == {"a": 1}
    store.delete("k")
    assert store.get("k") is None


def test_memory_store_expires_entries():
    store = session_store.MemoryStore()
    store.put("k", {"a": 1}, ttl=-1)     # already expired
    assert store.get("k") is None


def test_memory_store_index():
    store = session_store.MemoryStore()
    store.add_to_index("idx", "b")
    store.add_to_index("idx", "a")
    assert store.members("idx") == ["a", "b"]
    assert store.members("empty") == []


def test_build_store_picks_memory_without_redis_env(monkeypatch):
    for var in ("UPSTASH_REDIS_REST_URL", "KV_REST_API_URL",
                "UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert session_store.build_store().name == "memory"
    assert session_store.redis_configured() is False


@pytest.mark.parametrize("url_var,token_var", [
    ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"),
    ("KV_REST_API_URL", "KV_REST_API_TOKEN"),    # projects migrated off Vercel KV
])
def test_build_store_picks_redis_from_either_env_pair(monkeypatch, url_var, token_var):
    for var in ("UPSTASH_REDIS_REST_URL", "KV_REST_API_URL",
                "UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(url_var, "https://example.upstash.io")
    monkeypatch.setenv(token_var, "tok")
    assert session_store.redis_configured() is True
    assert session_store.build_store().name == "redis"


def test_redis_store_speaks_the_upstash_rest_protocol(monkeypatch):
    """Commands go as a JSON array body, not in the URL — a serialized session is
    far too large for a path segment."""
    calls = []

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"result": json.dumps({"hello": "world"})}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "body": json, "headers": headers})
        return FakeResponse()

    monkeypatch.setattr(session_store.httpx, "post", fake_post)
    store = session_store.RedisStore("https://example.upstash.io/", "tok")

    assert store.get("k") == {"hello": "world"}
    assert calls[0]["url"] == "https://example.upstash.io"
    assert calls[0]["body"] == ["GET", "k"]
    assert calls[0]["headers"]["Authorization"] == "Bearer tok"

    store.put("k", {"a": 1}, ttl=99)
    assert calls[1]["body"][:2] == ["SET", "k"]
    assert calls[1]["body"][3:] == ["EX", "99"]


def test_redis_read_failure_is_a_miss_not_a_crash(monkeypatch):
    """A lost read looks like an expired session, which the app already 404s."""
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(session_store.httpx, "post", boom)
    assert session_store.RedisStore("https://x.upstash.io", "t").get("k") is None


def test_redis_write_failure_raises(monkeypatch):
    """A silently dropped write would strand the caller mid-investigation with a
    session id that never resolves, so writes must not fail quietly."""
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(session_store.httpx, "post", boom)
    with pytest.raises(RuntimeError):
        session_store.RedisStore("https://x.upstash.io", "t").put("k", {"a": 1})


def test_corrupt_stored_value_is_a_miss(monkeypatch):
    class Bad:
        def raise_for_status(self): pass
        def json(self): return {"result": "{not json"}

    monkeypatch.setattr(session_store.httpx, "post", lambda *a, **k: Bad())
    assert session_store.RedisStore("https://x.upstash.io", "t").get("k") is None


# --- the codec -----------------------------------------------------------------
@pytest.mark.parametrize("cid", ["case-008", "case-009", "case-010"])
def test_pre_conversion_human_sessions_require_a_restart(cid):
    case = load_case(cid)
    s = GameSession(case, requested_mode="llm")
    blob = session_codec.dump_session(s, mode="llm", policy="demo", case_is_custom=False)
    blob["state"]["inference_mode"] = "authored"
    with pytest.raises(ValueError, match="evidence rules have changed"):
        session_codec.load_session(blob, _offline_build)


def test_engine_state_and_audit_survive():
    s = GameSession(load_case("case-001"), requested_mode="rule_based")
    s.run()
    restored = _roundtrip(s)

    assert restored.state.model_dump() == s.state.model_dump()
    assert [a.model_dump() for a in restored.audit] == [a.model_dump() for a in s.audit]
    assert restored.state.final_report.culprit == s.state.final_report.culprit
    # run_metadata is derived from the audit, so it must come out identical too.
    assert restored.run_metadata() == s.run_metadata()


def test_older_board_gets_public_capabilities_without_new_evidence():
    from src import affordances
    s = GameSession(load_case("case-011"))
    s.step()
    s.state.index.pop("tool_targets")
    s.state.index.pop("method_definitions")
    before = s.state.model_dump()
    restored = _roundtrip(s)
    assert restored.state.index["tool_targets"] == s.case.tool_targets()
    assert restored.state.index["method_definitions"] == s.case.method_definitions
    after = restored.state.model_dump()
    after["index"] = before["index"]
    assert after == before
    assert affordances.legal_actions(restored.state)
    fresh = _roundtrip(GameSession(load_case("case-011")))
    assert fresh.state.index == {}
    assert [a["tool"] for a in affordances.legal_actions(fresh.state)] == ["read_incident_report"]


def test_a_resumed_run_scores_the_same_as_an_uninterrupted_one():
    """The point of the whole exercise: serialization must not change the outcome."""
    from src.evaluator import evaluate

    case = load_case("case-001")
    straight = GameSession(case, requested_mode="rule_based")
    straight.run()
    direct = evaluate(case, straight.state, straight.run_metadata())

    stepped = GameSession(case, requested_mode="rule_based")
    while stepped.state.status == "investigating":
        stepped.step()
        stepped = _roundtrip(stepped)      # rehydrate after EVERY turn
    resumed = evaluate(case, stepped.state, stepped.run_metadata())

    assert resumed["total_score"] == direct["total_score"]
    assert resumed["correct"] == direct["correct"]


def test_agent_conversation_survives():
    """The message list IS the agent's memory; it cannot be regenerated."""
    case = load_case("case-001")
    script = [FakeResponse([TextBlock("Report first."),
                            ToolUseBlock("read_incident_report", {})])]
    inv = AgentInvestigator(case, client=FakeClient(script))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    s.step()
    inv.active_skill = "timeline_reconstruction"
    inv.skills_used = ["timeline_reconstruction"]
    inv.revises_left = 1
    inv.usage = {"llm_calls": 3, "input_tokens": 10, "output_tokens": 4}

    restored = _roundtrip(s, mode="llm")
    r_inv = restored.investigator
    assert len(r_inv._messages) == len(inv._messages)
    assert r_inv.active_skill == "timeline_reconstruction"
    assert r_inv.skills_used == ["timeline_reconstruction"]
    assert r_inv.revises_left == 1
    assert r_inv.usage["llm_calls"] == 3
    # The rehydrated investigator must drive the rehydrated session, not the old one.
    assert r_inv.session is restored


def test_thinking_blocks_keep_their_signature():
    """A thinking block stripped of its signature is rejected by the API on the next
    turn, which would break the run one step after the resume rather than at it."""
    class ThinkingBlock:
        type = "thinking"

        def __init__(self):
            self.thinking = "deliberating"
            self.signature = "sig-abc123"

        def model_dump(self):
            return {"type": "thinking", "thinking": self.thinking,
                    "signature": self.signature}

    case = load_case("case-001")
    inv = AgentInvestigator(case, client=FakeClient([]))
    s = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = s
    inv._messages.append({"role": "assistant", "content": [ThinkingBlock()]})

    restored = _roundtrip(s, mode="llm")
    block = restored.investigator._messages[-1]["content"][0]
    assert block["type"] == "thinking"
    assert block["signature"] == "sig-abc123"


def test_model_assessments_survive():
    case = load_case("case-011")
    s = GameSession(case, requested_mode="llm")
    s.execute_decision(Decision(action=NextAction("u", "", "read_incident_report", {}),
                                source="llm", llm_called=True, validation_status="ok"))
    s.execute_decision(Decision(
        action=NextAction("u", "", "check_access_log", {"location": "archive_room"}),
        source="llm", llm_called=True, validation_status="ok"))
    s.apply_assessment({"evidence_id": "al_mara_entry", "supports": "mara",
                        "weight": 0.3, "rationale": "her card"},
                       Decision(action=None, source="llm", llm_called=True,
                                validation_status="ok"))

    restored = _roundtrip(s, mode="llm")
    assert restored.state.hypothesis_for("mara").confidence == pytest.approx(
        expected_confidence(0.3))
    assert restored.run_metadata()["assessments"] == 1
    assert restored.run_metadata()["inference_mode"] == "model"
    ev = next(e for e in restored.state.evidence_collected if e.id == "al_mara_entry")
    assert ev.attribution == "model"


def test_interview_state_and_engine_side_audit_survive():
    """Contradiction detection compares against earlier answers, and the evaluator
    reads the truthfulness record — losing either changes a resumed run's result."""
    from src.human_interview import InterviewState

    s = GameSession(load_case("case-008"), requested_mode="rule_based")
    st = InterviewState()
    st.add_statement("timing", "Quarter past three.")
    s.toolbox._human_interview_states["pike"] = st
    s.toolbox.human_interview_audit.append({"suspect": "pike", "topic": "timing",
                                            "likely_lying": True})

    restored = _roundtrip(s)
    assert restored.toolbox._human_interview_states["pike"].statements == {
        "timing": ["Quarter past three."]}
    assert restored.toolbox.human_interview_audit[0]["likely_lying"] is True


def test_parked_decision_survives_so_a_human_answer_replays_the_right_action():
    s = GameSession(load_case("case-008"), requested_mode="rule_based")
    decision = Decision(
        action=NextAction("statement_validation", "Where were you?", "interview_human",
                          {"suspect": "pike", "question": "Where were you?",
                           "topic": "timing"}),
        source="rule_based", validation_status="ok")
    s._parked = (decision, ["interview_human(pike|where were you?)"])

    restored = _roundtrip(s)
    restored_decision, labels = restored._parked
    assert restored_decision.action.tool == "interview_human"
    assert restored_decision.action.arguments["suspect"] == "pike"
    assert restored_decision.source == "rule_based"
    assert labels == ["interview_human(pike|where were you?)"]


def test_agent_narrative_survives():
    s = GameSession(load_case("case-001"), requested_mode="rule_based")
    s.record_agent_conclusion({"verdict": "unresolved", "reasoning": "thin",
                               "summary": "Nothing established."})
    assert _roundtrip(s).agent_conclusion["summary"] == "Nothing established."


def test_schema_mismatch_is_refused_rather_than_half_loaded():
    """After a deploy changes the format, an old blob must fail loudly. Silently
    loading a partial session would corrupt an investigation mid-flight."""
    s = GameSession(load_case("case-001"), requested_mode="rule_based")
    blob = session_codec.dump_session(s, mode="rule_based", policy="demo",
                                      case_is_custom=False)
    blob["schema"] = 999
    with pytest.raises(ValueError, match="not supported"):
        session_codec.load_session(blob, app_mod._build_session)


# --- the boundary: a serialized session is engine-side data --------------------
def test_custom_case_body_is_stored_server_side_and_never_returned():
    # The recipe is fully defaulted, so an empty one is a valid minimal request.
    made = client.post("/api/custom_case", json={})
    assert made.status_code == 200, made.text[:300]

    case_id = made.json()["case_id"]
    body = made.text.lower()
    assert "hidden_truth" not in body
    assert "required_evidence" not in body

    # The truth lives in the store, which is why the case can be resumed at all.
    stored = app_mod.STORE.get(app_mod._case_key(case_id))
    assert stored is not None and "hidden_truth" in stored

    # ...and a session over it still refuses to leak.
    start = client.post("/api/sessions",
                        json={"case_id": case_id, "mode": "rule_based"})
    assert start.status_code == 200
    assert "hidden_truth" not in start.text.lower()


def test_unknown_session_id_is_a_clean_404():
    assert client.post("/api/sessions/deadbeef/step").status_code == 404
    assert client.get("/api/sessions/deadbeef/trace").status_code == 404


def test_stepping_a_session_twice_advances_it_across_requests():
    """The regression the store exists to prevent: losing the run between turns."""
    sid = client.post("/api/sessions",
                      json={"case_id": "case-001", "mode": "rule_based"}
                      ).json()["session_id"]
    first = client.post(f"/api/sessions/{sid}/step").json()
    second = client.post(f"/api/sessions/{sid}/step").json()
    assert first["state"]["iteration"] == 1
    assert second["state"]["iteration"] == 2
    assert len(client.get(f"/api/sessions/{sid}/trace").json()["trace"]) == 2
