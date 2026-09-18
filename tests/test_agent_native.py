"""Native tool_use agent (src/agent.py) with a scripted client — no network/key.

Covers: the agent loading a skill itself (get_skill), world actions going through the
engine (provenance = llm), the judge rejecting an unsupported conclusion and returning
a critique (revise), an honest `unresolved` being accepted, and the event stream
(input/think/tool/gate/final) coming out in the right order.
"""

from __future__ import annotations

import json

from src.agent import AgentInvestigator, iter_agent
from src.cases import load_case
from src.loop import GameSession


class Usage:
    input_tokens = 10
    output_tokens = 5


class TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    type = "tool_use"
    _n = 0

    def __init__(self, name, tool_input):
        self.name = name
        self.input = tool_input
        ToolUseBlock._n += 1
        self.id = f"tu_{ToolUseBlock._n}"


class FakeResponse:
    def __init__(self, blocks):
        self.content = blocks
        self.stop_reason = ("tool_use" if any(getattr(b, "type", "") == "tool_use"
                                              for b in blocks) else "end_turn")
        self.usage = Usage()


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self.script, "script exhausted while the agent still calls the model"
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def _script():
    return [
        FakeResponse([TextBlock("Report first."),
                      ToolUseBlock("read_incident_report", {})]),
        FakeResponse([ToolUseBlock("get_skill", {"name": "access_path_analysis"})]),
        FakeResponse([TextBlock("Who could have entered?"),
                      ToolUseBlock("check_access_log", {"location": "archive_room"})]),
        # Premature accusation — the judge MUST reject it (no 2 independent lines, etc.)
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "solved", "culprit": "mara",
                                    "reasoning": "She looks suspicious."})]),
        # After the critique: an honest unresolved — the judge accepts (evidence is thin).
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved",
                                    "reasoning": "Not enough independent evidence."})]),
    ]


def _run_events():
    case = load_case("case-001")
    client = FakeClient(_script())
    inv = AgentInvestigator(case, client=client)
    session = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = session
    events = []
    while session.state.status == "investigating":
        entry = session.step()
        events.append(entry)
    return session, inv, events


def test_judge_rejects_premature_accusation_then_accepts_unresolved():
    session, inv, _ = _run_events()
    gates = [e for e in inv.event_log if e.get("type") == "gate"]
    judged = [e for e in gates if e.get("phase") == "judge"]
    assert len(judged) == 2
    assert judged[0]["passed"] is False                 # accusation rejected
    assert any("independent" in i for i in judged[0]["issues"])
    assert judged[1]["passed"] is True                  # honest unresolved accepted
    assert any(e.get("phase") == "revise" for e in gates)
    assert session.state.status in ("unresolved", "budget_exhausted")
    assert inv.final_proposal["proposal"]["verdict"] == "unresolved"


def test_world_actions_have_llm_provenance_and_skill_attribution():
    session, inv, _ = _run_events()
    world = [a for a in session.audit if a.tool]
    assert all(a.decision_source == "llm" for a in world)
    md = session.run_metadata()
    assert md["fully_llm_driven"] is True and md["fallback_decisions"] == 0
    # A skill loaded via get_skill is attributed to the next world action.
    by_tool = {a.tool: a for a in world}
    assert by_tool["check_access_log"].skill == "access_path_analysis"


def test_iter_agent_event_stream(monkeypatch):
    import src.agent as agent_mod
    client = FakeClient(_script())
    events = list(iter_agent("case-001", client=client))
    types = [e["type"] for e in events]
    assert types[0] == "input" and types[-1] == "final"
    assert "think" in types and "tool" in types and "gate" in types
    final = events[-1]
    assert final["usage"]["llm_calls"] == 5
    assert final["structured"]["run_metadata"]["requested_mode"] == "llm"
    # hidden content must not appear in the input event
    assert "hidden" not in json.dumps(events[0]).lower()


def test_tool_result_feeds_state_back_to_model():
    session, inv, _ = _run_events()
    # The second decide() had to send a tool_result with observation/state_changes.
    results = [c for m in inv._messages if isinstance(m.get("content"), list)
               for c in m["content"]
               if isinstance(c, dict) and c.get("type") == "tool_result"]
    assert results
    assert any("remaining_budget" in str(c.get("content")) for c in results)


def test_judge_accepts_display_name_culprit():
    """The model naturally writes 'Mara' — the judge must normalize to the id 'mara'."""
    from src.judge import check_conclusion
    case = load_case("case-001")
    s = GameSession(case)
    s.run()  # skupi sve dokaze
    v = check_conclusion(s.state, {"verdict": "solved", "culprit": "Mara",
                                   "reasoning": "records + camera + card"})
    assert v["ok"], v["issues"]
    v2 = check_conclusion(s.state, {"verdict": "solved", "culprit": " MARA ",
                                    "reasoning": "..."})
    assert v2["ok"]


def test_world_tool_args_are_canonicalized():
    from src.agent import _canon_args
    out = _canon_args({"suspect": "Mara", "location": "Archive Room",
                       "reasoning": "Keep Text"})
    assert out["suspect"] == "mara" and out["location"] == "archive_room"
    assert out["reasoning"] == "Keep Text"   # Non-ID fields remain unchanged.


def test_full_run_with_display_form_args_solves_case():
    """Reproduction of the reported bug: the model sends 'Archive Room'/'Mara' (display
    names). Canonicalization turns them into ids and the case is SOLVED."""
    script = [
        FakeResponse([ToolUseBlock("read_incident_report", {})]),
        FakeResponse([ToolUseBlock("check_access_log", {"location": "Archive Room"})]),
        FakeResponse([ToolUseBlock("review_camera", {"location": "Archive Room"})]),
        FakeResponse([ToolUseBlock("inspect_location", {"location": "Archive Room"})]),
        FakeResponse([ToolUseBlock("analyze_object", {"object": "Access Card"})]),
        FakeResponse([ToolUseBlock("interview_character", {"suspect": "Mara"})]),
        FakeResponse([ToolUseBlock("verify_alibi", {"suspect": "Mara"})]),
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "solved", "culprit": "Mara",
                                    "reasoning": "The log, camera and card place her "
                                                 "at the scene; her alibi failed."})]),
    ]
    case = load_case("case-001")
    client = FakeClient(script)
    inv = AgentInvestigator(case, client=client)
    session = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = session
    while session.state.status == "investigating":
        session.step()

    # no move was "wasted" because of a name
    assert all(a.tool_ok for a in session.audit if a.tool), \
        [(a.tool, a.observation) for a in session.audit if not a.tool_ok]
    assert session.state.status == "solved"
    assert session.state.final_report.culprit == "mara"
    # The engine may auto-conclude as soon as the conditions are met (before submit); if the
    # model did manage to submit a conclusion, the judge had to accept it.
    assert inv.final_proposal is None or inv.final_proposal["judge"]["ok"]


def test_llm_reviewer_blocks_then_accepts_after_deterministic_gate_passes():
    """Layer 2 (LLM reviewer) must be able to reject BY ITSELF a conclusion that the deterministic
    gate let through — then accept it in the next round (revise). A deterministically-ok verdict
    here is a thin `unresolved` (no strong lead), and the fake reviewer first objects then relents."""
    calls = {"n": 0}

    def fake_reviewer(state, proposal):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ok": False, "issues": ["reasoning ignores the unexamined object"]}
        return {"ok": True, "issues": []}

    script = [
        FakeResponse([ToolUseBlock("read_incident_report", {})]),
        # Deterministically passes (thin state, no strong lead), but the reviewer rejects.
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved", "reasoning": "too early"})]),
        # After the reviewer critique: another honest unresolved — now accepted.
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved",
                                    "reasoning": "the trail is exhausted as far as the budget allows"})]),
    ]
    case = load_case("case-001")
    inv = AgentInvestigator(case, client=FakeClient(script), conclusion_judge=fake_reviewer)
    session = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = session
    while session.state.status == "investigating":
        session.step()

    judged = [e for e in inv.event_log if e.get("phase") == "judge"]
    assert len(judged) == 2
    # First: deterministically OK, but the reviewer rejected -> overall failed, with a [reviewer] prefix.
    assert judged[0]["deterministic_ok"] is True
    assert judged[0]["llm_ok"] is False
    assert judged[0]["passed"] is False
    assert any(i.startswith("[reviewer]") for i in judged[0]["issues"])
    # Second: both checks passed.
    assert judged[1]["deterministic_ok"] is True and judged[1]["llm_ok"] is True
    assert judged[1]["passed"] is True
    assert any(e.get("phase") == "revise" for e in inv.event_log)
    assert calls["n"] == 2


def test_unavailable_reviewer_cannot_approve_a_conclusion():
    """A failed review must be visible and retryable, never labelled accepted."""
    from src.judge import make_llm_conclusion_judge
    from src.cases import load_case as _lc

    class Boom:
        class messages:
            @staticmethod
            def create(**kw):
                raise RuntimeError("network down")

    class Garbage:
        class messages:
            @staticmethod
            def create(**kw):
                return FakeResponse([TextBlock("this is not json")])

    case = _lc("case-001")
    s = GameSession(case)
    s.run()
    proposal = {"verdict": "solved", "culprit": "mara", "reasoning": "log+camera+card"}
    for client in (Boom(), Garbage()):
        j = make_llm_conclusion_judge(client, "m")
        out = j(s.state, proposal)
        assert out["ok"] is False and out["issues"]
        assert out["error"] in ("review_unavailable", "unparseable_review")


def test_illegal_action_error_lists_legal_moves():
    """If the model does hit something illegal, the error must say what IS legal."""
    script = [
        FakeResponse([ToolUseBlock("read_incident_report", {})]),
        # verify_alibi pre ijednog intervjua/pitanja -> nelegalno
        FakeResponse([ToolUseBlock("verify_alibi", {"suspect": "mara"})]),
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved", "reasoning": "stop"})]),
    ]
    case = load_case("case-001")
    client = FakeClient(script)
    inv = AgentInvestigator(case, client=client)
    session = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = session
    while session.state.status == "investigating":
        session.step()
    bad = [a for a in session.audit if a.tool == "verify_alibi"][0]
    assert not bad.tool_ok
    assert "Legal actions now" in bad.observation
    assert "check_access_log(archive_room)" in bad.observation
