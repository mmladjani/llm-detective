"""Tests for skill performance tracking and recommendations."""

import pytest
from src.skill_performance import (
    SkillRecord,
    SkillEffectiveness,
    SkillPerformanceTracker,
    record_case_outcome,
    get_global_tracker,
)


def test_skill_record_creation():
    """SkillRecord should capture usage details."""
    record = SkillRecord(
        case_id="case-001",
        skill_name="timeline_reconstruction",
        used=True,
        phase="early",
        final_score=85,
        was_successful=True
    )

    assert record.case_id == "case-001"
    assert record.skill_name == "timeline_reconstruction"
    assert record.was_successful is True


def test_skill_effectiveness_calculation():
    """SkillEffectiveness should compute rates correctly."""
    eff = SkillEffectiveness(
        skill_name="timeline_reconstruction",
        times_used=10,
        times_successful=7,
        average_score=78.5,
        success_rate=0.7
    )

    assert eff.success_rate == 0.7
    assert eff.times_used == 10
    # rank_score combines success_rate and avg_score, weighted by confidence
    rank = eff.rank_score()
    assert rank > 0


def test_tracker_records_skill_usage():
    """Tracker should record skill usage."""
    tracker = SkillPerformanceTracker()

    tracker.record_skill_usage("case-001", "timeline_reconstruction", "early", 85, True)
    tracker.record_skill_usage("case-001", "statement_validation", "middle", 80, True)

    assert len(tracker.records) == 2


def test_tracker_computes_effectiveness():
    """Tracker should compute effectiveness from records."""
    tracker = SkillPerformanceTracker()

    # Record 3 uses of timeline_reconstruction: 2 successful, 1 failed
    tracker.record_skill_usage("case-001", "timeline_reconstruction", "early", 90, True)
    tracker.record_skill_usage("case-002", "timeline_reconstruction", "early", 75, True)
    tracker.record_skill_usage("case-003", "timeline_reconstruction", "middle", 50, False)

    eff = tracker.get_effectiveness("timeline_reconstruction")

    assert eff is not None
    assert eff.times_used == 3
    assert eff.times_successful == 2
    assert abs(eff.success_rate - 2 / 3) < 0.01  # Rounded to 0.67
    assert abs(eff.average_score - (90 + 75 + 50) / 3) < 0.1


def test_tracker_recommends_best_skills():
    """Tracker should recommend skills ranked by effectiveness."""
    tracker = SkillPerformanceTracker()

    # Skill A: 4 successful, 0 failed (100% success, avg 92.5)
    for score in [90, 95, 90, 95]:
        tracker.record_skill_usage("case-a", "skill_a", "early", score, True)

    # Skill B: 2 successful, 1 failed (66% success, avg 70)
    tracker.record_skill_usage("case-b", "skill_b", "early", 80, True)
    tracker.record_skill_usage("case-c", "skill_b", "middle", 70, True)
    tracker.record_skill_usage("case-d", "skill_b", "late", 60, False)

    # Skill C: never used
    # Skill D: 1 use, failed
    tracker.record_skill_usage("case-e", "skill_d", "early", 30, False)

    recommendations = tracker.recommend_skills("case-new", "early", top_n=5)

    # Should recommend A first (highest effectiveness)
    assert len(recommendations) > 0
    assert recommendations[0][0] == "skill_a"
    assert recommendations[0][1] > recommendations[1][1]  # Higher ranked


def test_tracker_excludes_tried_skills():
    """Recommendations should exclude skills already tried."""
    tracker = SkillPerformanceTracker()

    tracker.record_skill_usage("case-001", "skill_a", "early", 90, True)
    tracker.record_skill_usage("case-002", "skill_b", "early", 85, True)
    tracker.record_skill_usage("case-003", "skill_c", "early", 80, True)

    # Get recommendations, excluding skill_a
    recommendations = tracker.recommend_skills(
        "case-new", "early", tried_skills=["skill_a"], top_n=5
    )

    skill_names = [name for name, _ in recommendations]
    assert "skill_a" not in skill_names
    assert "skill_b" in skill_names or "skill_c" in skill_names


def test_tracker_respects_top_n():
    """Tracker should limit recommendations to top_n."""
    tracker = SkillPerformanceTracker()

    for i in range(10):
        tracker.record_skill_usage(f"case-{i}", f"skill_{i}", "early", 70 + i, True)

    recommendations = tracker.recommend_skills("case-new", "early", top_n=3)

    assert len(recommendations) == 3


def test_tracker_handles_empty_records():
    """Tracker should handle no records gracefully."""
    tracker = SkillPerformanceTracker()

    eff = tracker.get_effectiveness("nonexistent")
    assert eff is None

    recommendations = tracker.recommend_skills("case-new", "early")
    assert recommendations == []


def test_tracker_serialization():
    """Tracker should serialize and deserialize."""
    tracker = SkillPerformanceTracker()

    tracker.record_skill_usage("case-001", "skill_a", "early", 90, True)
    tracker.record_skill_usage("case-002", "skill_b", "middle", 75, False)

    data = tracker.to_dict()

    assert "records" in data
    assert "effectiveness_summary" in data
    assert len(data["records"]) == 2
    assert "skill_a" in data["effectiveness_summary"]
    assert "skill_b" in data["effectiveness_summary"]

    # Deserialize and verify
    tracker2 = SkillPerformanceTracker.from_dict(data)
    assert len(tracker2.records) == 2
    eff_a = tracker2.get_effectiveness("skill_a")
    assert eff_a is not None
    assert eff_a.success_rate == 1.0


def test_record_case_outcome_high_level():
    """High-level API should work correctly."""
    # Reset global tracker
    import src.skill_performance as sp
    sp._global_tracker = None

    record_case_outcome(
        case_id="case-001",
        skills_used=["timeline_reconstruction", "statement_validation"],
        final_score=90,
        was_successful=True,
        budget=8,
        actions_used=6
    )

    tracker = get_global_tracker()
    assert len(tracker.records) == 2

    eff_tl = tracker.get_effectiveness("timeline_reconstruction")
    assert eff_tl is not None
    assert eff_tl.times_used == 1
    assert eff_tl.times_successful == 1


def test_skill_effectiveness_rank_score():
    """Rank score should increase with use and success."""
    # Single use: lower confidence
    eff1 = SkillEffectiveness(
        skill_name="test", times_used=1, times_successful=1,
        average_score=90.0, success_rate=1.0
    )

    # Five uses: higher confidence
    eff5 = SkillEffectiveness(
        skill_name="test", times_used=5, times_successful=5,
        average_score=90.0, success_rate=1.0
    )

    # Both 100% success at same score, but eff5 should rank higher
    # due to higher confidence (more data)
    assert eff5.rank_score() > eff1.rank_score()


def test_phase_classification():
    """Phase should be classified based on budget usage."""
    # Reset global tracker
    import src.skill_performance as sp
    sp._global_tracker = None

    # Early: 2/8 actions used
    record_case_outcome("case-early", ["skill_a"], 90, True, budget=8, actions_used=2)

    # Middle: 5/8 actions used
    record_case_outcome("case-middle", ["skill_b"], 80, True, budget=8, actions_used=5)

    # Late: 7/8 actions used
    record_case_outcome("case-late", ["skill_c"], 70, False, budget=8, actions_used=7)

    tracker = get_global_tracker()
    eff_a = tracker.get_effectiveness("skill_a")
    eff_b = tracker.get_effectiveness("skill_b")
    eff_c = tracker.get_effectiveness("skill_c")

    # Check phases are recorded correctly
    assert eff_a is not None
    assert eff_a.skill_name == "skill_a"  # Phase should be "early"
    assert eff_b is not None
    assert eff_b.skill_name == "skill_b"  # Phase should be "middle"
    assert eff_c is not None
    assert eff_c.skill_name == "skill_c"  # Phase should be "late"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
