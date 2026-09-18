"""The two protections a publicly reachable deployment needs.

They stop different things and are tested separately:

  * the access gate decides whether a stranger may talk to the app at all;
  * the spend limit decides how much the world may cost, which the gate does not
    address — an authenticated visitor can still start investigations in a loop.

Both must be inert when unconfigured, or every existing test and every local run
would need credentials.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

import app as app_mod
from src import access, session_store

client = TestClient(app_mod.app)


def _basic(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "let-me-in")
    monkeypatch.delenv("APP_ACCESS_USER", raising=False)
    return _basic(access.DEFAULT_USER, "let-me-in")


# --- off by default ------------------------------------------------------------
def test_no_password_configured_means_no_gate(monkeypatch):
    """Local development and the offline suite must not need credentials."""
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    assert access.gate_enabled() is False
    assert client.get("/api/cases").status_code == 200


def test_blank_password_does_not_enable_a_gate_that_nothing_can_pass(monkeypatch):
    """A variable set to empty (easy to do in a dashboard) must read as 'unset',
    not as a gate whose password is the empty string."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "   ")
    assert access.gate_enabled() is False


# --- the gate ------------------------------------------------------------------
def test_api_is_refused_without_credentials(gated):
    r = client.get("/api/cases")
    assert r.status_code == 401
    # Without this header the browser never prompts and the UI just looks broken.
    assert r.headers["WWW-Authenticate"].startswith("Basic realm=")


def test_correct_credentials_pass(gated):
    assert client.get("/api/cases", headers=gated).status_code == 200


@pytest.mark.parametrize("header", [
    {},                                               # nothing
    {"Authorization": "Basic"},                       # scheme only
    {"Authorization": "Bearer let-me-in"},            # wrong scheme
    {"Authorization": "Basic not-base64!!"},          # undecodable
    {"Authorization": "Basic " + base64.b64encode(b"nocolon").decode()},
])
def test_malformed_authorization_is_refused_not_crashed(gated, header):
    assert client.get("/api/cases", headers=header).status_code == 401


def test_wrong_password_and_wrong_user_are_both_refused(gated):
    assert client.get("/api/cases",
                      headers=_basic(access.DEFAULT_USER, "guess")).status_code == 401
    assert client.get("/api/cases",
                      headers=_basic("someone", "let-me-in")).status_code == 401


def test_custom_username_is_honoured(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "pw")
    monkeypatch.setenv("APP_ACCESS_USER", "milos")
    assert client.get("/api/cases", headers=_basic("milos", "pw")).status_code == 200
    assert client.get("/api/cases",
                      headers=_basic(access.DEFAULT_USER, "pw")).status_code == 401


def test_the_ui_itself_is_gated(gated):
    """Serving the page to an anonymous visitor whose API calls then 401 looks like a
    broken app rather than a protected one."""
    assert client.get("/").status_code == 401
    assert client.get("/board").status_code == 401
    assert client.get("/", headers=gated).status_code == 200


def test_health_check_stays_reachable(gated):
    """One curl has to be able to tell you whether a deployment is misconfigured."""
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["access_gate"] is True
    # It reports that the protections are on, never what they are.
    assert "let-me-in" not in r.text
    assert "password" not in body


def test_session_endpoints_are_gated(gated):
    assert client.post("/api/sessions",
                       json={"case_id": "case-001", "mode": "rule_based"}
                       ).status_code == 401
    ok = client.post("/api/sessions", json={"case_id": "case-001",
                                            "mode": "rule_based"}, headers=gated)
    assert ok.status_code == 200
    sid = ok.json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/step").status_code == 401
    assert client.post(f"/api/sessions/{sid}/step", headers=gated).status_code == 200


# --- the spend limit -----------------------------------------------------------
def test_limit_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MAX_LLM_SESSIONS_PER_HOUR", raising=False)
    assert access.llm_session_limit() == 0


@pytest.mark.parametrize("raw,expected", [("5", 5), ("0", 0), ("-3", 0), ("junk", 0)])
def test_limit_parsing_is_defensive(monkeypatch, raw, expected):
    """A typo in a dashboard variable must not crash every request."""
    monkeypatch.setenv("MAX_LLM_SESSIONS_PER_HOUR", raw)
    assert access.llm_session_limit() == expected


def test_rule_based_sessions_are_never_charged(monkeypatch):
    """The offline baseline calls no model, so rate-limiting it would only make the
    free path worse."""
    monkeypatch.setenv("MAX_LLM_SESSIONS_PER_HOUR", "1")
    monkeypatch.setattr(app_mod, "STORE", session_store.MemoryStore())
    for _ in range(5):
        r = client.post("/api/sessions",
                        json={"case_id": "case-001", "mode": "rule_based"})
        assert r.status_code == 200


def test_llm_sessions_are_refused_once_the_window_is_spent(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    monkeypatch.setenv("MAX_LLM_SESSIONS_PER_HOUR", "2")
    monkeypatch.setattr(app_mod, "STORE", session_store.MemoryStore())

    # The builder is stubbed: this test is about the limit, not about talking to a model.
    monkeypatch.setattr(app_mod, "_build_session",
                        lambda case, mode, policy: app_mod.GameSession(
                            case, requested_mode="rule_based"))

    body = {"case_id": "case-001", "mode": "llm"}
    assert client.post("/api/sessions", json=body).status_code == 200
    assert client.post("/api/sessions", json=body).status_code == 200
    third = client.post("/api/sessions", json=body)
    assert third.status_code == 429
    # The refusal has to point somewhere useful, not just say no.
    assert "rule_based" in third.text

    # ...and the free baseline is still available while the LLM path is capped.
    assert client.post("/api/sessions",
                       json={"case_id": "case-001", "mode": "rule_based"}
                       ).status_code == 200


def test_limit_refuses_rather_than_fails_open_when_the_store_is_down(monkeypatch):
    """A rate limiter that fails open is not a rate limiter."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    monkeypatch.setenv("MAX_LLM_SESSIONS_PER_HOUR", "5")

    class BrokenStore(session_store.MemoryStore):
        def incr(self, key, ttl):
            raise RuntimeError("store unreachable")

    monkeypatch.setattr(app_mod, "STORE", BrokenStore())
    r = client.post("/api/sessions", json={"case_id": "case-001", "mode": "llm"})
    assert r.status_code == 503


def test_config_advertises_the_limit(monkeypatch):
    monkeypatch.setenv("MAX_LLM_SESSIONS_PER_HOUR", "7")
    assert client.get("/api/config").json()["llm_sessions_per_window"] == 7


# --- the counter ---------------------------------------------------------------
def test_memory_counter_counts_and_expires():
    store = session_store.MemoryStore()
    assert store.incr("k", 3600) == 1
    assert store.incr("k", 3600) == 2
    assert store.incr("expired", -1) == 1
    assert store.incr("expired", -1) == 1     # window already gone each time


def test_redis_counter_sets_the_window_only_on_the_first_hit(monkeypatch):
    """EXPIRE on every increment would slide the window forward forever, so a busy
    deployment would never reset its allowance."""
    sent = []
    counter = {"n": 0}

    class Reply:
        def __init__(self, result):
            self._result = result

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": self._result}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(json)
        if json[0] == "INCR":
            counter["n"] += 1
            return Reply(counter["n"])
        return Reply(1)

    monkeypatch.setattr(session_store.httpx, "post", fake_post)
    store = session_store.RedisStore("https://x.upstash.io", "t")

    assert store.incr("k", 900) == 1
    assert ["EXPIRE", "k", "900"] in sent
    sent.clear()
    assert store.incr("k", 900) == 2
    assert not any(cmd[0] == "EXPIRE" for cmd in sent)
