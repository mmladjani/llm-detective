"""Dynamic tool gating — the *legal action set* for the current state.

This is what lets the agent (not a fixed script) decide "what am I allowed to do right
now, and which of those is worth doing." Which tools/targets are permitted depends on
what has already been discovered:

  * before the report is read, the only legal move is to read it;
  * physical tools and interviews unlock once the board (suspects/location/objects) is known;
  * `verify_alibi(x)` unlocks only once there is a claim to test — x has been interviewed,
    or an open question about x was raised;
  * targets must have an implemented response in the public capability catalog;
  * `compare_testimonies(a, b)` unlocks only once BOTH a and b have been interviewed;
  * an action already spent is no longer offered.

The engine computes this set; the agent chooses within it. The rule-based investigator's
plan already respects these affordances, so it is unchanged — it just never proposes an
illegal move.
"""

from __future__ import annotations

from typing import Any

from .state import InvestigationState

LOCATION_TOOLS = ("check_access_log", "review_camera", "inspect_location")


def action_key(tool: str, arguments: dict[str, Any]) -> Any:
    """Stable identity of a (tool, args) target — mirrors ToolBox.arg_key."""
    if tool == "read_incident_report":
        return "_report"
    if tool in LOCATION_TOOLS:
        return arguments.get("location")
    if tool == "analyze_object":
        return arguments.get("object")
    # One question is one action, so the key carries the question text. Interviewing
    # a person is not a single spendable move — you come back and press harder — but
    # asking the SAME question twice buys nothing and is blocked like any repeat.
    if tool == "interview_human":
        suspect = arguments.get("suspect")
        question = (arguments.get("question") or "").strip().lower()
        return f"{suspect}|{question}" if suspect else None
    if tool in ("interview_character", "verify_alibi"):
        return arguments.get("suspect")
    if tool == "compare_testimonies":
        a, b = arguments.get("a"), arguments.get("b")
        if a and b:
            return "|".join(sorted([a, b]))
        return arguments.get("pair")
    return None


def legal_actions(state: InvestigationState) -> list[dict[str, Any]]:
    """The state-dependent set of permitted actions, each as
    {tool, arguments, skill, reason, key}. Already-spent actions are excluded."""
    idx = state.index
    attempted = {(t["tool"], t.get("arg_key")) for t in state.tools_used}
    out: list[dict[str, Any]] = []

    def add(tool, args, skill, reason):
        key = action_key(tool, args)
        if tool != "read_incident_report" and key not in idx.get("tool_targets", {}).get(tool, []):
            return
        if (tool, key) in attempted:
            return
        out.append({"tool": tool, "arguments": args, "skill": skill,
                    "reason": reason, "key": key})

    # Nothing known yet: the only legal move is to read the report.
    if not idx:
        add("read_incident_report", {}, "timeline_reconstruction",
            "no information yet — read the incident report")
        return out

    loc = idx.get("location")
    suspects = list(idx.get("suspects", []))
    objects = list(idx.get("objects", []))
    interviewed = {t.get("arg_key") for t in state.tools_used
                   if t["tool"] == "interview_character" and t.get("ok")}
    # A person's account is still an account. Questioning a human-played character
    # gives the agent a claim to test, so it must unlock verify_alibi exactly as an
    # AI interview does — otherwise casting someone as human silently removes a
    # branch of the investigation. interview_human keys are "suspect|question".
    interviewed |= {str(t.get("arg_key") or "").split("|")[0]
                    for t in state.tools_used
                    if t["tool"] == "interview_human" and t.get("ok")}
    open_q = {q.suspect for q in state.open_questions if q.suspect}

    if loc:
        add("check_access_log", {"location": loc}, "access_path_analysis",
            "establish who could physically reach the scene")
        add("review_camera", {"location": loc}, "physical_evidence_analysis",
            "footage may show who was present")
        add("inspect_location", {"location": loc}, "physical_evidence_analysis",
            "the scene may reveal method and motive")

    for o in objects:
        add("analyze_object", {"object": o}, "physical_evidence_analysis",
            f"examining {o} may reveal the method")

    for s in suspects:
        add("interview_character", {"suspect": s}, "statement_validation",
            f"gather {s}'s account")

    # Human-played characters are advertised without a question, since the question
    # is the agent's to compose. is_legal() validates the concrete call.
    for s in human_played(state):
        out.append({
            "tool": "interview_human",
            "arguments": {"suspect": s},
            "skill": "statement_validation",
            "reason": f"{s} is played by a person — ask them a question directly",
            "key": s,
        })

    # verify_alibi unlocks only when there is a claim to test.
    for s in suspects:
        if s in open_q:
            add("verify_alibi", {"suspect": s}, "statement_validation",
                f"an open question about {s} must be resolved")
        elif s in interviewed:
            add("verify_alibi", {"suspect": s}, "statement_validation",
                f"test {s}'s stated alibi against the record")

    # compare_testimonies unlocks only when both parties have been interviewed.
    for i, a in enumerate(suspects):
        for b in suspects[i + 1:]:
            if a in interviewed and b in interviewed:
                add("compare_testimonies", {"a": a, "b": b}, "contradiction_resolution",
                    f"{a} and {b} both interviewed — reconcile their accounts")

    return out


def human_played(state: InvestigationState) -> list[str]:
    """Suspect ids played by a person. Learned from the report — not hidden truth."""
    return [c["id"] for c in state.index.get("characters", [])
            if c.get("human_played") and c.get("id")]


def is_legal(state: InvestigationState, tool: str, arguments: dict[str, Any]) -> bool:
    arguments = arguments or {}

    # interview_human is validated rather than enumerated: legal_actions cannot list
    # every question the agent might compose, so check the parts that are checkable.
    if tool == "interview_human":
        suspect = arguments.get("suspect")
        question = (arguments.get("question") or "").strip()
        if not suspect or not question:
            return False
        if suspect not in human_played(state):
            return False
        key = action_key(tool, arguments)
        already = {(t["tool"], t.get("arg_key")) for t in state.tools_used}
        return (tool, key) not in already

    key = action_key(tool, arguments)
    return any(la["tool"] == tool and la["key"] == key for la in legal_actions(state))
