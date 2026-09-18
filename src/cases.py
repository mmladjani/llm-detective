"""Case loading and the hidden-truth boundary.

A case file on disk holds EVERYTHING: the hidden truth, evaluator metadata, and the
tool response tables. `Case` wraps that dict but exposes two clearly separated views:

  * `public_briefing()` -> only what the investigator is allowed to start with.
  * `hidden_truth` / `evaluation` -> used solely by the engine + evaluator.

Only the public briefing and tool results enter model messages. Python drivers receive
the Case wrapper to build that projection, not permission to serialize its hidden data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .methods import method_definitions

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


class Case:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    # ---- engine/evaluator-only views ------------------------------------------
    @property
    def case_id(self) -> str:
        return self._data["case_id"]

    @property
    def hidden_truth(self) -> dict[str, str]:
        return self._data["hidden_truth"]

    @property
    def required_evidence(self) -> list[str]:
        return self._data.get("required_evidence", [])

    @property
    def evaluation(self) -> dict[str, Any]:
        # Evaluator metadata (expected outcome). NEVER exposed to the investigator.
        return self._data.get("evaluation", {"expected_status": "solved"})

    @property
    def budget(self) -> int:
        return int(self._data.get("budget", 8))

    @property
    def inference_mode(self) -> str:
        """Who decides what a piece of evidence MEANS: "authored" or "model".

        "authored" (the default for legacy cases and generated cases): the file labels each
        evidence item with `supports` (which suspect it points at) and `weight` (how
        strongly). The engine sums those numbers, so the attribution and the strength
        of the inference are the case author's, not the model's. If an LLM driver
        is used, it chooses actions but does not own these stored attributions.

        "model": the case file carries no `supports`/`weight` at all. Tool results
        deliver the observation and the bare evidence item; the model must call
        `assess_evidence` to say who it implicates and how heavily, and the engine
        applies the model's own numbers. Report facts also require cited model
        assertions via establish_fact; authored reveals are suppressed.

        Cases 008–013 use model inference; 008–010 also allow human witnesses.
        Interpretation remains fallible: see docs/LIMITATIONS.md.
        """
        mode = str(self._data.get("inference", "authored")).lower()
        return mode if mode in ("authored", "model") else "authored"

    @property
    def tools(self) -> dict[str, Any]:
        return self._data["tools"]

    @property
    def fact_candidates(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._data.get("fact_candidates", {}).items()}

    @property
    def method_definitions(self) -> dict[str, str]:
        return method_definitions(self.fact_candidates.get("method")
                                  if self.inference_mode == "model" else None)

    def tool_targets(self) -> dict[str, list[str]]:
        """Public capabilities, not the observations behind those targets."""
        return {name: sorted(self.tools.get(name, {})) for name in (
            "check_access_log", "review_camera", "inspect_location", "analyze_object",
            "interview_character", "verify_alibi", "compare_testimonies")}

    @property
    def index(self) -> dict[str, Any]:
        return self._data["index"]

    @property
    def incident_report(self) -> str:
        return self._data["incident_report"]

    # ---- investigator-facing view ---------------------------------------------
    def public_briefing(self) -> dict[str, Any]:
        """The ONLY case-level information handed to the investigator up front.

        Contains no selected culprit/method, expected evidence, or evaluator metadata.
        Candidate fact values include alternatives; they do not mark the correct answer.
        """
        return {
            "case_id": self.case_id,
            "title": self._data["title"],
            "difficulty": self._data["difficulty"],
            "short_description": self._data["short_description"],
            "incident_report": self.incident_report,
            "budget": self.budget,
            # Navigational, not a hint: the agent has to know whether evidence arrives
            # pre-labelled or whether weighing it is its own job. Saying "you must
            # interpret this yourself" reveals nothing about the solution.
            "inference": self.inference_mode,
            "method_definitions": self.method_definitions,
            **({"fact_candidates": self.fact_candidates} if self.inference_mode == "model" else {}),
        }

    # A human-played character carries what they witnessed and what they are hiding.
    # That is ground truth about the case, so it is stripped from the navigational
    # catalog — the agent must earn it by interviewing, and the person playing the
    # character gets it via their role card instead.
    _PUBLIC_CHARACTER_FIELDS = ("id", "name", "role", "human_played")

    @classmethod
    def _public_character(cls, char: dict[str, Any]) -> dict[str, Any]:
        return {k: char[k] for k in cls._PUBLIC_CHARACTER_FIELDS if k in char}

    def catalog(self) -> dict[str, Any]:
        """Non-secret 'board' the investigator learns from the incident report:
        who the suspects are, where the incident happened, objects of interest.
        Purely navigational — none of this reveals the solution."""
        idx = self.index
        return {
            "suspects": list(idx.get("suspects", [])),
            "location": idx.get("location"),
            "time_window": idx.get("time_window", {}),
            "objects": list(idx.get("objects", [])),
            "characters": [self._public_character(c)
                           for c in idx.get("characters", [])],
            "locations": idx.get("locations", []),
            "tool_targets": self.tool_targets(),
            "method_definitions": self.method_definitions,
            **({"fact_candidates": self.fact_candidates} if self.inference_mode == "model" else {}),
        }

    def characters_full(self) -> list[dict[str, Any]]:
        """Characters WITH knowledge/secrets/motivation. Engine-side only."""
        return list(self.index.get("characters", []))


# Generated complex cases live separately, so they do not change the default
# built-in corpus or the tests that assume it. They are included explicitly.
GENERATED_DIR = CASES_DIR / "generated"


def list_case_files(include_generated: bool = False) -> list[Path]:
    files = sorted(CASES_DIR.glob("case_*.json"))
    if include_generated and GENERATED_DIR.exists():
        files += sorted(GENERATED_DIR.glob("case_*.json"))
    return files


def load_case(case_id: str) -> Case:
    # Search generated too: the agent (iter_agent) loads by id, so generated cases
    # must be available. list_cases() DELIBERATELY does not show them by default.
    for path in list_case_files(include_generated=True):
        data = json.loads(path.read_text())
        if data["case_id"] == case_id:
            return Case(data)
    raise KeyError(f"Unknown case: {case_id}")


def list_cases(include_generated: bool = False) -> list[dict[str, Any]]:
    out = []
    for path in list_case_files(include_generated=include_generated):
        data = json.loads(path.read_text())
        out.append(
            {
                "case_id": data["case_id"],
                "title": data["title"],
                "difficulty": data["difficulty"],
                "short_description": data["short_description"],
                "budget": int(data.get("budget", 8)),
                "inference": Case(data).inference_mode,
                "human_played": any(c.get("human_played") for c in Case(data).characters_full()),
            }
        )
    return out
