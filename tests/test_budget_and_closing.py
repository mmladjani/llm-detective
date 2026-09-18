"""Regression for a live run on 21 Jul (case-001, LLM mode) that failed even though the agent
rezonovao ispravno. Tri nezavisna uzroka, svaki ovde pokriven:

  1. an illegal move spent budget — a name error on an object ("conservation_case")
     ate 2 of 11 moves and brought the case down;
  2. the run shut down the moment the budget hit 0 — the agent had no chance to
     submit the conclusion it had just earned, so the report was null;
  3. the judge critique ("method is not established") did not say WHAT was available,
     unlike a tool error which cleanly lists the legal moves.

All offline — a scripted client, no network and no key.
"""

from __future__ import annotations

from src.agent import AgentInvestigator
from src.cases import load_case
from src.judge import check_conclusion
from src.loop import MAX_ILLEGAL_ACTIONS, GameSession
from tests.test_agent_native import FakeClient, FakeResponse, TextBlock, ToolUseBlock


def _session(script):
    case = load_case("case-001")
    inv = AgentInvestigator(case, client=FakeClient(script))
    session = GameSession(case, investigator=inv, requested_mode="llm")
    inv.session = session
    return session, inv


def _read_report():
    return FakeResponse([TextBlock("Report first."),
                         ToolUseBlock("read_incident_report", {})])


# --- 1. an illegal move is free, but limited -------------------------------------

def test_illegal_action_costs_no_budget():
    """The exact scenario from the live run: the model takes an object name from the prose."""
    session, _ = _session([
        _read_report(),
        FakeResponse([TextBlock("Analiziram kofer."),
                      ToolUseBlock("analyze_object", {"object": "conservation_case"})]),
        FakeResponse([ToolUseBlock("analyze_object", {"object": "access_card"})]),
    ])
    session.step()                                   # read_incident_report
    after_report = session.state.remaining_budget

    session.step()                                   # illegal move
    assert session.state.remaining_budget == after_report, \
        "an illegal move must NOT deduct budget"
    assert session.state.illegal_actions == 1
    assert session.state.status == "investigating"

    session.step()                                   # legal move
    assert session.state.remaining_budget == after_report - 1


def test_illegal_action_error_teaches_and_counts_down():
    session, _ = _session([
        _read_report(),
        FakeResponse([ToolUseBlock("analyze_object", {"object": "conservation_case"})]),
    ])
    session.step()
    entry = session.step()
    assert entry.tool_ok is False
    obs = entry.observation
    assert "illegal_action" in obs
    assert "analyze_object(access_card)" in obs        # what IS legal
    assert "cost you no budget" in obs
    assert f"{MAX_ILLEGAL_ACTIONS - 1} illegal call(s) remain" in obs


def test_repeated_illegal_actions_still_halt_the_run():
    """Free does not mean unlimited — an agent stuck in a loop stops."""
    bad = [FakeResponse([ToolUseBlock("analyze_object", {"object": "nema_ovoga"})])
           for _ in range(MAX_ILLEGAL_ACTIONS)]
    session, _ = _session([_read_report(), *bad])
    session.step()
    for _ in range(MAX_ILLEGAL_ACTIONS):
        if session.state.status == "investigating":
            session.step()
    assert session.state.illegal_actions == MAX_ILLEGAL_ACTIONS
    assert session.state.status == "illegal_action_limit"
    assert session.state.final_report is not None
    assert session.state.final_report.culprit is None   # we do not invent a culprit


# --- 2. a free final move ---------------------------------------------------------

def test_budget_zero_grants_one_free_closing_turn():
    """Before: budget 0 -> the run shuts down at once, the agent never states a verdict.
    Now: it gets one free move, and that move is recorded as its proposal."""
    session, inv = _session([
        _read_report(),
        FakeResponse([ToolUseBlock("check_access_log", {"location": "archive_room"})]),
        # This is the free final move:
        FakeResponse([TextBlock("Budget is up, stating what I have."),
                      ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved",
                                    "reasoning": "Not enough independent lines."})]),
    ])
    session.step()
    session.state.remaining_budget = 1
    session.step()                                    # budget drops to 0

    assert session.state.remaining_budget == 0
    assert session.state.final_turn_used is True
    assert session.state.status == "investigating", "does not shut down before the final move"

    session.step()                                    # agent concludes
    assert session.state.status in ("unresolved", "budget_exhausted")
    assert inv.final_proposal is not None, "the agent proposal must be recorded"
    assert inv.final_proposal["proposal"]["verdict"] == "unresolved"


def test_closing_turn_buys_no_world_action():
    """The free move is ONLY for concluding — not a hidden extra budget."""
    session, _ = _session([
        _read_report(),
        FakeResponse([ToolUseBlock("check_access_log", {"location": "archive_room"})]),
        FakeResponse([ToolUseBlock("review_camera", {"location": "archive_room"})]),
    ])
    session.step()
    session.state.remaining_budget = 1
    session.step()
    session.step()                                    # attempt a world action at 0

    # The last audit entry is the "closing" marker; the move is the one before it.
    attempt = [a for a in session.audit if a.tool == "review_camera"][-1]
    assert attempt.tool_ok is False
    assert "budget_exhausted" in attempt.observation
    assert "submit_conclusion" in attempt.observation
    assert session.state.remaining_budget == 0
    assert session.state.status != "investigating", "after a missed move the run stops"


def test_model_is_told_it_is_the_final_turn():
    """The signal must reach the model, otherwise the free move is worthless."""
    session, inv = _session([
        _read_report(),
        FakeResponse([ToolUseBlock("check_access_log", {"location": "archive_room"})]),
        FakeResponse([ToolUseBlock("submit_conclusion",
                                   {"verdict": "unresolved", "reasoning": "Tanko."})]),
    ])
    session.step()
    session.state.remaining_budget = 1
    session.step()
    session.step()

    sent = "".join(str(c["messages"]) for c in inv._client.messages.calls)
    assert "final_turn" in sent
    assert "submit_conclusion now" in sent


# --- 3. kritika sudije je akcionabilna --------------------------------------------

def test_judge_names_the_actions_that_could_establish_the_missing_fact():
    session, _ = _session([_read_report()])
    session.step()
    st = session.state

    verdict = check_conclusion(st, {"verdict": "solved", "culprit": "mara",
                                    "reasoning": "Ona je."})
    joined = " ".join(verdict["issues"])
    assert "method of the incident is not established" in joined
    assert "Actions still available" in joined
    assert "analyze_object(access_card)" in joined, \
        "the judge must say what is available, like a tool error does"


# --- 4. replay of the live run ----------------------------------------------------

def test_live_run_replay_now_solves_the_case():
    """A literal replay of the 21 Jul run: same moves, same object-name error.
    It used to end with budget_exhausted and culprit=None at confidence 0.98."""
    def T(name, **args):
        return FakeResponse([TextBlock("..."), ToolUseBlock(name, args)])

    session, _ = _session([
        T("read_incident_report"),
        T("check_access_log", location="archive_room"),
        T("review_camera", location="archive_room"),
        T("interview_character", suspect="mara"),
        T("verify_alibi", suspect="mara"),
        T("interview_character", suspect="idris"),
        T("interview_character", suspect="lena"),
        T("inspect_location", location="archive_room"),
        T("analyze_object", object="conservation_case"),   # the error from the live run
        T("analyze_object", object="access_card"),
        T("submit_conclusion", verdict="solved", culprit="mara",
          reasoning="Sve linije se poklapaju."),
    ])
    while session.state.status == "investigating":
        session.step()

    report = session.state.final_report
    assert session.state.status == "solved"
    assert report.culprit == "mara"
    assert report.method == "copied_access_card"
    assert session.state.illegal_actions == 1
    assert session.state.remaining_budget > 0, "a name error must not eat the budget"


# --- 5. an exhausted revise must not end an investigation that still has budget ---

def test_rejected_conclusion_sends_agent_back_to_work_while_budget_remains():
    """Live run #2 (a generated case, maternity_ward): the agent submitted the same
    conclusion three times, spent the revise budget and the run was cut off with 2 UNUSED moves —
    even though a move (`analyze_object`) existed that establishes the `method`."""
    def T(name, **args):
        return FakeResponse([TextBlock("..."), ToolUseBlock(name, args)])

    def submit():
        return T("submit_conclusion", verdict="solved", culprit="mara",
                 reasoning="Sve pokazuje na nju.")

    session, inv = _session([
        T("read_incident_report"),
        T("check_access_log", location="archive_room"),
        T("review_camera", location="archive_room"),
        T("interview_character", suspect="mara"),
        T("verify_alibi", suspect="mara"),
        T("inspect_location", location="archive_room"),
        submit(), submit(), submit(),          # the revise budget is spent here
        T("analyze_object", object="access_card"),   # tek sada method
        submit(),
    ])
    while session.state.status == "investigating":
        session.step()

    assert session.state.status == "solved", \
        "a rejected conclusion must not end the investigation while moves remain"
    assert session.state.final_report.method == "copied_access_card"
    assert any(a.tool == "analyze_object" for a in session.audit), \
        "the agent had to be sent back into the world, not cut off"


def test_new_evidence_restores_the_right_to_submit():
    """The revise budget exists to prevent repeating the SAME conclusion without new data.
    As soon as the agent finds something, the right to submit is restored."""
    def T(name, **args):
        return FakeResponse([TextBlock("..."), ToolUseBlock(name, args)])

    session, inv = _session([
        T("read_incident_report"),
        T("submit_conclusion", verdict="solved", culprit="mara", reasoning="a"),
        T("submit_conclusion", verdict="solved", culprit="mara", reasoning="b"),
        T("check_access_log", location="archive_room"),
        T("review_camera", location="archive_room"),
    ])
    session.step()
    assert inv.revises_left == inv._max_revises
    session.step()                      # dva odbijena slanja pa svet-akcija
    assert inv.revises_left == 0, "submissions without new evidence spend the revise budget"
    session.step()                      # new evidence reaches the model
    assert inv.revises_left == inv._max_revises, "nov dokaz obnavlja pravo na slanje"


# --- 6. a case must not name an object that does not exist ------------------------

def test_case_prose_does_not_invent_objects():
    """Root of the live bug: the camera described 'conservation case', but the world only knows
    'manuscript_case' — the model obeyed the evidence and was penalized."""
    case = load_case("case-001")
    known = set(case.index.get("objects", []))
    camera = case.tools["review_camera"]["archive_room"]["observation"]
    obj = case.tools["analyze_object"]["access_card"]["observation"]

    assert "conservation case" not in camera.lower()
    assert "conservation case" not in obj.lower()
    assert "manuscript_case" in known
