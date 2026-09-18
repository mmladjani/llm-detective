"""Turn a live GameSession into JSON and back, losslessly enough to resume a turn.

Needed because a serverless request cannot hand a Python object to the next request
(see session_store). The rule this module follows: everything the engine, the judge or
the evaluator will read later must round-trip; everything reconstructible from the
environment is rebuilt instead of stored.

Rebuilt, never stored:
  * the Anthropic client (from ANTHROPIC_API_KEY),
  * the case body for built-in cases (loaded from cases/*.json by id),
  * the tool tables and the catalog (derived from the case).

Stored:
  * `InvestigationState` and the audit trail — pydantic, so model_dump round-trips,
  * the agent's message list, which IS the conversation and cannot be regenerated,
  * the interview bookkeeping the contradiction check reads,
  * the parked decision, so a human-in-the-loop resume replays the exact action.

The hidden truth is part of a stored CUSTOM case body, so a serialized session is
engine-side data. It goes to the session store, never to the browser — the browser
keeps receiving `_view()` as before.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from .cases import Case, load_case
from .human_interview import InterviewState
from .investigator import Decision, NextAction
from .loop import GameSession
from .state import AuditEntry, InvestigationState

# Old model sessions automatically revealed facts and lack full reviewer observations.
# Do not resume them under the new inference contract; ask for a new investigation.
SCHEMA_VERSION = 2


def _jsonable(value: Any) -> Any:
    """Recursively convert SDK/pydantic objects to plain JSON types.

    The agent appends raw `response.content` to its message list, so those entries are
    Anthropic SDK block objects rather than dicts. They are pydantic models, so
    `model_dump()` gives back the exact wire shape the API accepts on the way in —
    which matters most for `thinking` blocks, whose `signature` must survive intact or
    the next request is rejected.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _jsonable(dump())
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return str(value)


def _decision_to_dict(decision: Decision) -> dict[str, Any]:
    out = dataclasses.asdict(decision)
    out["action"] = dataclasses.asdict(decision.action) if decision.action else None
    return out


def _decision_from_dict(payload: dict[str, Any]) -> Decision:
    data = dict(payload)
    action = data.pop("action", None)
    return Decision(action=NextAction(**action) if action else None, **data)


def dump_session(session: GameSession, *, mode: str, policy: str,
                 case_is_custom: bool) -> dict[str, Any]:
    """Serialize everything needed to continue this investigation."""
    inv = session.investigator
    toolbox = session.toolbox

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "case_id": session.case.case_id,
        # Built-in cases are reloaded from disk by id; only a generated case has to
        # carry its body, because it exists nowhere else.
        "case_data": session.case._data if case_is_custom else None,
        "mode": mode,
        "policy": policy,
        "requested_mode": session.requested_mode,
        "state": session.state.model_dump(),
        "audit": [a.model_dump() for a in session.audit],
        "agent_conclusion": _jsonable(session.agent_conclusion),
        "conclusion_review": _jsonable(session.conclusion_review),
        "events": _jsonable(session.events),
        "parked": None,
        "interview_states": {
            suspect: st.statements
            for suspect, st in toolbox._human_interview_states.items()
        },
        # Engine-side truthfulness record the evaluator reads. Dropping it would make
        # a resumed human-in-the-loop run score differently from an uninterrupted one.
        "human_interview_audit": _jsonable(toolbox.human_interview_audit),
        "agent": None,
    }

    if session._parked is not None:
        decision, legal_labels = session._parked
        payload["parked"] = {
            "decision": _decision_to_dict(decision),
            "legal_labels": list(legal_labels),
        }

    # Only the native agent carries conversation state worth preserving; the
    # rule-based and schema drivers are stateless policies over InvestigationState.
    if hasattr(inv, "_messages"):
        payload["agent"] = {
            "messages": _jsonable(inv._messages),
            "skills_used": list(getattr(inv, "skills_used", [])),
            "active_skill": getattr(inv, "active_skill", None),
            "revises_left": getattr(inv, "revises_left", 0),
            "usage": dict(getattr(inv, "usage", {})),
            "pending_id": getattr(inv, "_pending_id", None),
            "pending_extras": list(getattr(inv, "_pending_extras", [])),
            "nudged": bool(getattr(inv, "_nudged", False)),
            "seq": int(getattr(inv, "_seq", 0)),
            "final_proposal": _jsonable(getattr(inv, "final_proposal", None)),
            # think_log/event_log are drained by the caller each turn and streamed to
            # the UI immediately, so anything still buffered here has already been
            # delivered. Persisting them would replay old thoughts after a resume.
        }
    return payload


def load_session(payload: dict[str, Any], build_session) -> tuple[GameSession, Case]:
    """Rebuild a GameSession from `dump_session` output.

    `build_session(case, mode, policy) -> GameSession` is injected rather than
    imported so this module does not depend on the web layer's mode resolution and
    HTTP error handling.
    """
    version = payload.get("schema")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"session schema {version!r} is not supported (expected {SCHEMA_VERSION}); "
            f"start a new investigation."
        )

    case = Case(payload["case_data"]) if payload.get("case_data") \
        else load_case(payload["case_id"])
    session = build_session(case, payload["mode"], payload["policy"])

    session.state = InvestigationState.model_validate(payload["state"])
    if session.state.index:
        # Public capabilities/definitions are reconstructible, not evidence. Upgrade
        # schema-2 boards too, without revealing a board before the report was read.
        catalog = case.catalog()
        for field in ("tool_targets", "method_definitions"):
            session.state.index[field] = catalog[field]
    session.audit = [AuditEntry.model_validate(a) for a in payload.get("audit", [])]
    session.requested_mode = payload.get("requested_mode", session.requested_mode)
    session.agent_conclusion = payload.get("agent_conclusion")
    session.conclusion_review = payload.get("conclusion_review")
    session.events = list(payload.get("events") or [])

    for suspect, statements in (payload.get("interview_states") or {}).items():
        st = InterviewState()
        st.statements = {k: list(v) for k, v in statements.items()}
        session.toolbox._human_interview_states[suspect] = st
    session.toolbox.human_interview_audit = list(
        payload.get("human_interview_audit") or [])

    parked = payload.get("parked")
    if parked:
        session._parked = (_decision_from_dict(parked["decision"]),
                           list(parked["legal_labels"]))

    agent = payload.get("agent")
    inv = session.investigator
    if agent and hasattr(inv, "_messages"):
        inv._messages = list(agent.get("messages") or [])
        inv.skills_used = list(agent.get("skills_used") or [])
        inv.active_skill = agent.get("active_skill")
        inv.revises_left = int(agent.get("revises_left", 0))
        inv.usage = dict(agent.get("usage") or
                         {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0})
        inv._pending_id = agent.get("pending_id")
        inv._pending_extras = list(agent.get("pending_extras") or [])
        inv._nudged = bool(agent.get("nudged", False))
        inv._seq = int(agent.get("seq", 0))
        inv.final_proposal = agent.get("final_proposal")

    # The investigator is bound to the session it advances; _build_session wires this
    # for a fresh run, and a rehydrated one must not be left pointing at the old
    # object graph the builder created.
    if hasattr(inv, "session"):
        inv.session = session
    return session, case


def parked_question(session: GameSession) -> Optional[dict[str, Any]]:
    """Convenience mirror of GameSession.pending_question for the web layer."""
    return session.pending_question()
