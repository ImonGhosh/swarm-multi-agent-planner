from __future__ import annotations

from aco_tot.heuristics import heuristic_desirability
from aco_tot.types import TaskFeatures


def _features(
    *,
    num_people: int,
    num_days: int,
    duration_hours: float,
    has_preference: bool,
    has_earliest_preference: bool,
) -> TaskFeatures:
    return TaskFeatures(
        task_id="t",
        num_people=num_people,
        num_days=num_days,
        duration_hours=duration_hours,
        has_preference=has_preference,
        has_earliest_preference=has_earliest_preference,
    )


def test_eta_values_within_allowed_set():
    features = _features(
        num_people=6,
        num_days=3,
        duration_hours=1.0,
        has_preference=True,
        has_earliest_preference=True,
    )
    allowed = {1.0, 1.2, 1.5}
    for level in [1, 2, 3, 4, 5]:
        for branch in [
            "constraint_first",
            "people_first",
            "preference_first",
            "participant_wise",
            "day_wise",
            "mixed_normalization",
            "free_slot_computation",
            "candidate_slot_enumeration",
            "progressive_elimination",
            "earliest_valid",
            "preference_priority",
            "conservative_selection",
            "direct_output",
            "one_step_self_check",
            "format_check_output",
        ]:
            try:
                eta = heuristic_desirability(level, branch, features)
            except ValueError:
                continue
            assert eta in allowed


def test_level4_preference_priority_rules():
    with_pref = _features(
        num_people=2,
        num_days=1,
        duration_hours=0.5,
        has_preference=True,
        has_earliest_preference=False,
    )
    without_pref = _features(
        num_people=2,
        num_days=1,
        duration_hours=0.5,
        has_preference=False,
        has_earliest_preference=False,
    )
    assert heuristic_desirability(4, "preference_priority", with_pref) == 1.5
    assert heuristic_desirability(4, "preference_priority", without_pref) == 1.0
