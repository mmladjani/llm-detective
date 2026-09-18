"""Phase 2 — 'deep' cases: physical evidence is weak, the deciding link is a broken
alibi, so the fixed (brute-force) baseline spends SIGNIFICANTLY more steps than on shallow ones — and
build_and_verify still guarantees solvability. This is where an info-gain agent has something
to beat. Offline, no key."""

from __future__ import annotations

import random

from src.generator import build_and_verify
from src.loop import GameSession
from src.investigator import RuleBasedInvestigator

NAMES = ["Mara", "Idris", "Lena", "Tomas", "Sara"]


def _steps_to_solve(case):
    s = GameSession(case, requested_mode="rule_based")
    s.investigator = RuleBasedInvestigator(s.toolbox)
    s.run()
    used = len([a for a in s.state.tools_used if a.get("ok")])
    return s.state.status, used


def _recipe(seed, deep):
    return {"incident_type": "theft", "suspects": NAMES[:5], "culprit": "random",
            "twist": "none", "num_red_herrings": 1, "decoy_objects": 2,
            "deep_evidence": deep, "budget": 14, "seed": seed}


def test_deep_case_still_self_checks_and_is_solved():
    case = build_and_verify(_recipe(seed=3, deep=True))   # raises if not solvable
    status, _ = _steps_to_solve(case)
    assert status == "solved"


def test_deep_costs_more_steps_than_shallow_for_baseline():
    """The point of Phase 2: on deep cases a fixed order is INEFFICIENT."""
    shallow, deep = [], []
    for seed in range(30):
        try:
            sc = build_and_verify(_recipe(seed, deep=False))
            st, n = _steps_to_solve(sc)
            if st == "solved":
                shallow.append(n)
        except ValueError:
            pass
        try:
            dc = build_and_verify(_recipe(seed, deep=True))
            st, n = _steps_to_solve(dc)
            if st == "solved":
                deep.append(n)
        except ValueError:
            pass
    assert shallow and deep
    avg_shallow = sum(shallow) / len(shallow)
    avg_deep = sum(deep) / len(deep)
    # The deep bar is much costlier for brute-force (empirically ~5 vs ~11).
    assert avg_deep >= avg_shallow + 3


def test_deep_flag_defaults_off():
    # without deep_evidence the behavior is unchanged (shallow, ~5 steps)
    case = build_and_verify(_recipe(seed=1, deep=False))
    _, n = _steps_to_solve(case)
    assert n <= 7
