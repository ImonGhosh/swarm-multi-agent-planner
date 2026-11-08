"""Heuristic desirability rules for ACO branch scoring."""

from __future__ import annotations

from .types import TaskFeatures


def _is_many_participants(features: TaskFeatures) -> bool:
    return features.num_people >= 5


def _is_multi_day(features: TaskFeatures) -> bool:
    return features.num_days > 1


def _is_long_duration(features: TaskFeatures) -> bool:
    return features.duration_hours >= 1.0


def _is_small_simple(features: TaskFeatures) -> bool:
    return (
        features.num_people <= 3
        and features.num_days == 1
        and features.duration_hours <= 0.5
    )


def _is_hard_tight(features: TaskFeatures) -> bool:
    return (
        features.num_people >= 5
        or features.num_days > 1
        or features.duration_hours >= 1.0
    )


def heuristic_desirability(level: int, branch_id: str, features: TaskFeatures) -> float:
    """Compute eta_ij in the allowed range {1.0, 1.2, 1.5}."""
    many_people = _is_many_participants(features)
    multi_day = _is_multi_day(features)
    long_duration = _is_long_duration(features)
    has_pref = features.has_preference
    has_earliest_pref = features.has_earliest_preference
    small_simple = _is_small_simple(features)
    hard_tight = _is_hard_tight(features)

    if level == 1 and branch_id == "constraint_first":
        return 1.5 if (long_duration or multi_day) else 1.2
    if level == 1 and branch_id == "people_first":
        return 1.5 if many_people else 1.2
    if level == 1 and branch_id == "preference_first":
        return 1.5 if has_pref else 1.0

    if level == 2 and branch_id == "participant_wise":
        return 1.5 if many_people else 1.2
    if level == 2 and branch_id == "day_wise":
        return 1.5 if multi_day else 1.0
    if level == 2 and branch_id == "mixed_normalization":
        return 1.2 if (many_people and multi_day) else 1.0

    if level == 3 and branch_id == "free_slot_computation":
        return 1.5 if small_simple else 1.0
    if level == 3 and branch_id == "candidate_slot_enumeration":
        return 1.5 if (multi_day or features.duration_hours <= 0.5) else 1.2
    if level == 3 and branch_id == "progressive_elimination":
        return 1.5 if many_people else 1.2

    if level == 4 and branch_id == "earliest_valid":
        if has_earliest_pref:
            return 1.5
        if has_pref:
            return 1.2
        return 1.0
    if level == 4 and branch_id == "preference_priority":
        return 1.5 if has_pref else 1.0
    if level == 4 and branch_id == "conservative_selection":
        return 1.2 if hard_tight else 1.0

    if level == 5 and branch_id == "direct_output":
        return 1.0
    if level == 5 and branch_id == "one_step_self_check":
        return 1.5
    if level == 5 and branch_id == "format_check_output":
        return 1.2

    raise ValueError(f"Unknown (level, branch_id)=({level}, {branch_id})")
