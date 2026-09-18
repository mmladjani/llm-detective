"""Track skill usage and performance across investigations.

This module builds a feedback loop: agent uses skills, we evaluate the outcome,
and future agents get recommendations based on what worked.

Key concepts:
  - skill_record: tracks (case_id, skill, success/failure, score)
  - effectiveness: aggregate success rate + average score for each skill
  - recommendations: skills ranked by effectiveness for current case context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class SkillRecord:
    """Single record of a skill being used in an investigation."""
    case_id: str
    skill_name: str
    used: bool  # True if agent called get_skill for this skill
    phase: str  # "early", "middle", "late" (based on budget remaining)
    final_score: int  # 0-100 from evaluator
    was_successful: bool  # True if case was solved correctly
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SkillEffectiveness:
    """Aggregate performance stats for a skill."""
    skill_name: str
    times_used: int
    times_successful: int
    average_score: float
    success_rate: float  # 0.0-1.0

    def rank_score(self) -> float:
        """Score for ranking (higher = better). Combines success rate and avg score."""
        if self.times_used == 0:
            return 0.0
        # Confidence-adjusted: require multiple uses before recommending
        confidence = min(1.0, self.times_used / 5.0)  # Max confidence at 5+ uses
        return (self.success_rate * 100 + self.average_score) / 2 * confidence


class SkillPerformanceTracker:
    """Tracks skill usage and effectiveness across cases."""

    def __init__(self):
        self.records: list[SkillRecord] = []
        self._effectiveness_cache: dict[str, SkillEffectiveness] = {}
        self._cache_dirty = False

    def record_skill_usage(self, case_id: str, skill_name: str, phase: str,
                          final_score: int, was_successful: bool) -> None:
        """Record that a skill was used in a case."""
        record = SkillRecord(
            case_id=case_id,
            skill_name=skill_name,
            used=True,
            phase=phase,
            final_score=final_score,
            was_successful=was_successful
        )
        self.records.append(record)
        self._cache_dirty = True

    def get_effectiveness(self, skill_name: str) -> SkillEffectiveness | None:
        """Get aggregate performance for a skill."""
        if self._cache_dirty:
            self._rebuild_cache()

        return self._effectiveness_cache.get(skill_name)

    def get_all_effectiveness(self) -> dict[str, SkillEffectiveness]:
        """Get effectiveness for all tracked skills."""
        if self._cache_dirty:
            self._rebuild_cache()
        return dict(self._effectiveness_cache)

    def recommend_skills(self, case_id: str, current_phase: str,
                        tried_skills: list[str] | None = None,
                        top_n: int = 3) -> list[tuple[str, float]]:
        """Recommend skills ranked by effectiveness.

        Args:
            case_id: current investigation (for context, if available)
            current_phase: "early", "middle", "late"
            tried_skills: skills already used in this investigation (exclude from recommendations)
            top_n: how many recommendations to return

        Returns:
            List of (skill_name, rank_score) tuples, highest ranked first.
        """
        tried_skills = tried_skills or []
        if self._cache_dirty:
            self._rebuild_cache()

        # Rank all skills by effectiveness
        rankings = [
            (name, eff.rank_score())
            for name, eff in self._effectiveness_cache.items()
            if name not in tried_skills and eff.times_used > 0
        ]

        # Sort by rank score descending, return top N
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:top_n]

    def _rebuild_cache(self) -> None:
        """Rebuild effectiveness cache from records."""
        effectiveness_map: dict[str, list[SkillRecord]] = {}

        for record in self.records:
            if record.skill_name not in effectiveness_map:
                effectiveness_map[record.skill_name] = []
            effectiveness_map[record.skill_name].append(record)

        self._effectiveness_cache = {}
        for skill_name, records in effectiveness_map.items():
            times_used = len(records)
            times_successful = sum(1 for r in records if r.was_successful)
            average_score = sum(r.final_score for r in records) / times_used if times_used > 0 else 0

            self._effectiveness_cache[skill_name] = SkillEffectiveness(
                skill_name=skill_name,
                times_used=times_used,
                times_successful=times_successful,
                average_score=round(average_score, 1),
                success_rate=round(times_successful / times_used, 2) if times_used > 0 else 0.0
            )

        self._cache_dirty = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for persistence."""
        if self._cache_dirty:
            self._rebuild_cache()

        return {
            "records": [
                {
                    "case_id": r.case_id,
                    "skill_name": r.skill_name,
                    "phase": r.phase,
                    "final_score": r.final_score,
                    "was_successful": r.was_successful,
                    "timestamp": r.timestamp,
                }
                for r in self.records
            ],
            "effectiveness_summary": {
                name: {
                    "times_used": eff.times_used,
                    "times_successful": eff.times_successful,
                    "average_score": eff.average_score,
                    "success_rate": eff.success_rate,
                }
                for name, eff in self._effectiveness_cache.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillPerformanceTracker:
        """Deserialize from dict."""
        tracker = cls()
        for record_data in data.get("records", []):
            record = SkillRecord(
                case_id=record_data["case_id"],
                skill_name=record_data["skill_name"],
                used=True,
                phase=record_data["phase"],
                final_score=record_data["final_score"],
                was_successful=record_data["was_successful"],
                timestamp=record_data.get("timestamp", "")
            )
            tracker.records.append(record)
        tracker._cache_dirty = True
        return tracker


# Global singleton tracker (in real system, would be persisted to disk)
_global_tracker: SkillPerformanceTracker | None = None


def get_global_tracker() -> SkillPerformanceTracker:
    """Get or create the global skill performance tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = SkillPerformanceTracker()
    return _global_tracker


def record_case_outcome(case_id: str, skills_used: list[str], final_score: int,
                       was_successful: bool, budget: int, actions_used: int) -> None:
    """Record the outcome of a complete investigation.

    Called by the engine after a case concludes. Determines phase based on remaining budget.
    """
    tracker = get_global_tracker()

    # Determine phase based on budget usage
    if actions_used <= budget // 3:
        phase = "early"
    elif actions_used <= 2 * budget // 3:
        phase = "middle"
    else:
        phase = "late"

    for skill in skills_used:
        tracker.record_skill_usage(case_id, skill, phase, final_score, was_successful)
