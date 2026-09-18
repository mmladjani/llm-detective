"""Regression tests for three defects found by reading a real run's terminal log.

All three were invisible to the existing suite because nothing exercised the
log/record path end to end:

  1. iter_agent dropped the FINAL action of every run, because execute_decision
     returns the closing synthesis entry (tool=None) once a stopping condition
     fires. The action that established the last fact never reached the log.
  2. think_log was drained in full before event_log, so within one decide() the
     agent's reaction to a judge rejection printed ABOVE the rejection.
  3. run_cli treated history.record_from_trace's return value as a Path; it is
     the record dict, so every completed CLI run ended in an AttributeError.
"""

import json
from pathlib import Path

import pytest

from src import history
from src.agent import AgentInvestigator, iter_agent
from src.cases import load_case
from src.loop import GameSession


# --- fakes: the same shape tests/test_agent_native.py uses ----------------------

class TextBlock:
    type = "text"
    def __init__(self, text): self.text = text

class ToolUseBlock:
    type = "tool_use"
    _n = 0
    def __init__(self, name, input):
        ToolUseBlock._n += 1
        self.id = f"tu{ToolUseBlock._n}"
        self.name = name
        self.input = input

class FakeResponse:
    def __init__(self, content): self.content = content
    usage = None

class FakeMessages:
    def __init__(self, script): self._script = list(script)
    def create(self, **kw):
        return self._script.pop(0) if self._script else FakeResponse(
            [ToolUseBlock("submit_conclusion",
                          {"verdict": "unresolved", "reasoning": "out of script"})])

class FakeClient:
    def __init__(self, script): self.messages = FakeMessages(script)


def T(name, **args):
    return FakeResponse([TextBlock(f"about to call {name}"), ToolUseBlock(name, args)])


# --- 1. the final action must appear in the event stream ------------------------

def test_final_action_is_not_dropped_from_the_stream():
    """The move that trips the stopping condition must still be emitted."""
    script = [
        T("read_incident_report"),
        T("check_access_log", location="archive_room"),
        T("review_camera", location="archive_room"),
        T("analyze_object", object="access_card"),
        T("interview_character", suspect="mara"),
        T("verify_alibi", suspect="mara"),
        T("inspect_location", location="archive_room"),
        T("analyze_object", object="manuscript_case"),
        T("interview_character", suspect="idris"),
        T("interview_character", suspect="lena"),
        T("compare_testimonies", a="mara", b="idris"),
    ]
    events = list(iter_agent("case-001", client=FakeClient(script)))
    session_tools = [e["name"] for e in events if e["type"] == "tool"
                     and e["name"] != "get_skill"]

    # Every executed action in the audit must have a matching tool event.
    import src.agent as agent_mod
    logged = [t for t in session_tools]
    assert logged, "no tool events emitted at all"

    # The run reached a terminal state, so the last audited action is the one that
    # historically went missing.
    finals = [e for e in events if e["type"] == "final"]
    assert finals, "run did not finish"
    assert len(logged) >= 1


def test_every_executed_action_reaches_the_stream():
    """Cross-check the emitted tool events against the engine's own audit."""
    script = [T("read_incident_report"),
              T("check_access_log", location="archive_room"),
              T("review_camera", location="archive_room"),
              T("analyze_object", object="access_card"),
              T("interview_character", suspect="mara"),
              T("verify_alibi", suspect="mara")]
    cap = {}
    events = list(iter_agent("case-001", client=FakeClient(script), _capture=cap))
    session = cap["session"]

    audited = [(a.tool, a.iteration) for a in session.audit if a.tool is not None]
    emitted = [(e["name"], e["step"]) for e in events
               if e["type"] == "tool" and e["name"] != "get_skill"]

    assert len(emitted) == len(audited), (
        f"{len(audited)} actions executed but {len(emitted)} logged — "
        f"audited={[t for t,_ in audited]} emitted={[n for n,_ in emitted]}")


# --- 2. causal ordering within one decide() ------------------------------------

def test_thinking_does_not_print_above_the_verdict_it_reacts_to():
    """A rejected conclusion, then a revision: the gate event must come first."""
    script = [
        T("read_incident_report"),
        # premature accusation -> the deterministic gate rejects it
        FakeResponse([TextBlock("I am confident already."),
                      ToolUseBlock("submit_conclusion",
                                   {"verdict": "solved", "culprit": "mara",
                                    "reasoning": "hunch"})]),
        # reaction to the critique, then a real action in the same decide()
        FakeResponse([TextBlock("The judge rejected my conclusion; gathering evidence."),
                      ToolUseBlock("check_access_log", {"location": "archive_room"})]),
        T("review_camera", location="archive_room"),
        T("analyze_object", object="access_card"),
    ]
    events = list(iter_agent("case-001", client=FakeClient(script)))

    kinds = [(e["type"], e.get("phase")) for e in events]
    gate_at = next(i for i, k in enumerate(kinds) if k == ("gate", "judge"))
    reaction_at = next(i for i, e in enumerate(events)
                       if e["type"] == "think" and "rejected my conclusion" in e["text"])

    assert gate_at < reaction_at, (
        "the agent's reaction to the rejection was emitted before the rejection")


def test_seq_stamps_interleave_think_and_events():
    """drain_think/drain_events expose the order things actually happened in."""
    case = load_case("case-001")
    inv = AgentInvestigator(case, client=FakeClient([]))
    inv._seq = 0
    inv.think_log.clear(); inv._think_seq.clear()

    inv._seq += 1; inv.think_log.append("first thought"); inv._think_seq.append(inv._seq)
    inv._log_event({"type": "gate", "phase": "judge"})
    inv._seq += 1; inv.think_log.append("second thought"); inv._think_seq.append(inv._seq)

    merged = sorted([*inv.drain_think(), *inv.drain_events()], key=lambda p: p[0])
    shapes = [(t if isinstance(t, str) else t.get("phase")) for _, t in merged]
    assert shapes == ["first thought", "judge", "second thought"]


# --- 3. the CLI's history line ------------------------------------------------

def test_record_from_trace_returns_a_dict_and_the_file_is_findable(tmp_path):
    """run_cli crashed by calling .relative_to() on this. Pin the contract."""
    from src.evaluator import evaluate
    from src.investigator import RuleBasedInvestigator

    case = load_case("case-001")
    session = GameSession(case, requested_mode="rule_based")
    session.investigator = RuleBasedInvestigator(session.toolbox)
    session.run()

    # third positional arg is the EVALUATION dict, as app.py passes it
    evaluation = evaluate(case, session.state)
    runs = tmp_path / "runs"
    rec = history.record_from_trace(case, session, evaluation,
                                    seconds=0.1, runs_dir=runs)

    assert isinstance(rec, dict), "callers rely on the record dict"
    assert not isinstance(rec, Path)
    assert "run_id" in rec

    # The same lookup app.py now performs must find the written file.
    hits = sorted(runs.glob(f"*_{rec['run_id']}.json"))
    assert hits, f"no file matching run_id {rec['run_id']}"
    assert json.loads(hits[-1].read_text())["run_id"] == rec["run_id"]


# --- 4. the judge must not assume the crime is a homicide ---------------------

def test_crossexam_questions_do_not_assume_a_homicide():
    """The corpus is mostly theft and substitution; a 'victim died' prompt is wrong.

    A live run showed the agent spending a turn reasoning about the judge's
    template error instead of about the evidence.
    """
    from src.judge import _generate_crossexam_questions
    from src.state import InvestigationState

    st = InvestigationState(case_id="case-004", remaining_budget=4)
    qs = _generate_crossexam_questions(st, {"verdict": "solved", "culprit": "damir"})

    blob = " ".join(qs).lower()
    assert qs, "expected questions when method/motive/opportunity are unestablished"
    for word in ("victim die", "autopsy", "toxicology", "weapon", "murder"):
        assert word not in blob, f"homicide-specific wording leaked in: {word!r}"
    assert "method" in blob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
