from __future__ import annotations

import pytest

from swarm_planner.verifier.parser import parse_task
from swarm_planner.verifier.scorer import (
    build_task_quality_scorer,
    deposit_quality,
)


def _task(header: str, schedules: str, prefs: str = "", days_word: str = "day") -> str:
    return (
        "You are an expert at scheduling meetings.\n\n"
        f"TASK: {header}\n\n"
        f"Here are the existing schedules for everyone during the {days_word}:\n"
        f"{schedules}\n\n"
        f"{prefs} Find a time that works for everyone's schedule and constraints.\nSOLUTION:"
    )


def _answer(day: str, span: str) -> str:
    return f"Here is the proposed time: {day}, {span}"


def test_task_quality_single_day_hard_constraints():
    task_text = """
You are an expert at scheduling meetings.

TASK: You need to schedule a meeting for Alice and Bob for half an hour between the work hours of 9:00 to 17:00 on Monday.

Here are the existing schedules for everyone during the day:
Alice has meetings on Monday during 9:00 to 9:30;
Bob is free the entire day;

Alice would rather not meet on Monday after 16:00. Find a time that works for everyone's schedule and constraints.
SOLUTION:
""".strip()

    scorer = build_task_quality_scorer(task_text)
    valid = scorer.score_prediction("Here is the proposed time: Monday, 9:30 - 10:00")
    conflict = scorer.score_prediction("Here is the proposed time: Monday, 9:00 - 9:30")

    assert valid.hard_valid is True
    assert valid.score > 0.5
    assert conflict.hard_valid is False
    assert conflict.score < 0.5


def test_task_quality_multi_day_schedule_parsing():
    task_text = """
You are an expert at scheduling meetings.

TASK: You need to schedule a meeting for Bruce and Jessica for half an hour between the work hours of 9:00 to 17:00 on either Monday or Tuesday.

Here are the existing schedules for everyone during the days:
Bruce has blocked their calendar on Monday during 9:00 to 17:00, Tuesday during 9:00 to 10:00;
Jessica has meetings on Monday during 9:00 to 9:30, Tuesday during 9:00 to 9:30;

Find a time that works for everyone's schedule and constraints.
SOLUTION:
""".strip()

    scorer = build_task_quality_scorer(task_text)
    monday = scorer.score_prediction("Here is the proposed time: Monday, 10:00 - 10:30")
    tuesday = scorer.score_prediction("Here is the proposed time: Tuesday, 10:00 - 10:30")

    assert monday.hard_valid is False
    assert tuesday.hard_valid is True
    assert tuesday.score > monday.score


def test_task_quality_typo_normalization_and_earliest_preference():
    task_text = """
You are an expert at scheduling meetings.

TASK: You need to schedule a meeting for Ethan and Bob for half an hour between the work hours of 9:00 to 17:00 on Monday.

Here are the existing schedules for everyone during the day:
Ethanhas meetings on Monday during 9:00 to 9:30;
Bob has no meetings the whole day;

You would like to schedule the meeting at their earlist availability.
Find a time that works for everyone's schedule and constraints.
SOLUTION:
""".strip()

    scorer = build_task_quality_scorer(task_text)
    earliest = scorer.score_prediction("Here is the proposed time: Monday, 9:30 - 10:00")
    later = scorer.score_prediction("Here is the proposed time: Monday, 10:00 - 10:30")
    typo_conflict = scorer.score_prediction("Here is the proposed time: Monday, 9:00 - 9:30")

    assert earliest.hard_valid is True
    assert later.hard_valid is True
    assert earliest.score > later.score
    assert typo_conflict.hard_valid is False


def test_task_quality_handles_malformed_day_splits_as_soft_preferences():
    task_text = """
You are an expert at scheduling meetings.

TASK: You need to schedule a meeting for Scott and Henry for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday, Wednesday or Thursday.

Here are the existing schedules for everyone during the days:
Scott is free the entire day;
Henry is free the entire day;

Scott would like to avoid more meetings on Wednesday after 13:30. Thursday. Henry do not want to meet on Monday. Tuesday. Find a time that works for everyone's schedule and constraints.
SOLUTION:
""".strip()

    scorer = build_task_quality_scorer(task_text)
    tuesday_slot = scorer.score_prediction("Here is the proposed time: Tuesday, 9:00 - 9:30")
    wednesday_slot = scorer.score_prediction("Here is the proposed time: Wednesday, 9:00 - 9:30")

    assert tuesday_slot.hard_valid is True
    assert wednesday_slot.hard_valid is True
    assert wednesday_slot.score > tuesday_slot.score


# --- Phase 1 fixes -----------------------------------------------------------------

_HEADER_MON = (
    "You need to schedule a meeting for {a} and {b} for half an hour between the "
    "work hours of 9:00 to 17:00 on Monday."
)


@pytest.mark.parametrize("name", ["Doris", "Alexis", "Dennis"])
def test_names_ending_in_is_are_not_split(name):
    """F5: v1 rewrote 'Doris' to 'Dor is' and silently dropped that person's calendar."""
    text = _task(
        _HEADER_MON.format(a=name, b="Bob"),
        f"{name} has meetings on Monday during 9:00 to 10:00;\nBob is free the entire day;",
    )
    task = parse_task(text)
    assert name in task.participants
    assert task.busy_by_participant_by_day[name]["Monday"] == [(540, 600)]
    assert task.warnings == ()
    scorer = build_task_quality_scorer(text)
    assert scorer.score_prediction(_answer("Monday", "9:00 - 9:30")).hard_valid is False
    assert scorer.score_prediction(_answer("Monday", "10:00 - 10:30")).hard_valid is True


@pytest.mark.parametrize("name", ["Dennis", "Ethan"])
def test_glued_has_typo_is_split_without_breaking_the_name(name):
    text = _task(
        _HEADER_MON.format(a=name, b="Bob"),
        f"{name}has meetings on Monday during 9:00 to 9:30;\nBob is free the entire day;",
    )
    task = parse_task(text)
    assert task.participants == (name, "Bob")
    assert task.busy_by_participant_by_day[name]["Monday"] == [(540, 570)]
    assert task.warnings == ()


def test_earliest_is_judged_among_slots_that_meet_soft_preferences():
    """F6: v1 ranked 'earliest' among all hard-valid slots, so the right answer lost points."""
    text = _task(
        _HEADER_MON.format(a="Alice", b="Bob"),
        "Alice is free the entire day;\nBob is free the entire day;",
        "Alice would rather not meet on Monday before 10:00. "
        "You would like to schedule the meeting at their earliest availability.",
    )
    scorer = build_task_quality_scorer(text)
    right = scorer.score_prediction(_answer("Monday", "10:00 - 10:30"))
    too_early = scorer.score_prediction(_answer("Monday", "9:00 - 9:30"))
    later = scorer.score_prediction(_answer("Monday", "10:30 - 11:00"))
    assert scorer.best_slot == ("Monday", 600, 630)
    assert right.score == 1.0 and right.slot_rank == 0
    assert too_early.hard_valid and too_early.score == 0.5
    assert later.hard_valid and later.score == 0.75
    assert scorer.top_slots() == [("Monday", 600, 630)]


def test_without_an_earliest_request_every_best_slot_is_a_top_slot():
    text = _task(
        _HEADER_MON.format(a="Alice", b="Bob"),
        "Alice has meetings on Monday during 9:00 to 16:00;\nBob is free the entire day;",
    )
    scorer = build_task_quality_scorer(text)
    top = scorer.top_slots()
    assert top == [("Monday", 960, 990), ("Monday", 990, 1020)]
    assert scorer.best_slot == top[0]
    assert scorer.score_prediction(_answer("Monday", "16:30 - 17:00")).score == 1.0


def test_ranking_orders_by_soft_preferences_then_time():
    text = _task(
        _HEADER_MON.format(a="Alice", b="Bob"),
        "Alice is free the entire day;\nBob is free the entire day;",
        "Bob would like to avoid more meetings on Monday before 10:00.",
    )
    scorer = build_task_quality_scorer(text)
    ranked = scorer.ranked_slots
    assert ranked[0] == ("Monday", 600, 630)  # first slot that meets the preference
    assert ranked[-2:] == [("Monday", 540, 570), ("Monday", 570, 600)]  # violators last
    assert scorer.score_prediction(_answer("Monday", "9:00 - 9:30")).slot_rank == len(ranked) - 2


def test_split_preference_sentences_are_read_one_by_one():
    """'X can not meet on Monday. Tuesday before 11:00.' blocks all of Monday, but Tuesday
    only before 11:00. v1 applied the 'before 11:00' bound to both days."""
    text = _task(
        "You need to schedule a meeting for Alice and Bob for half an hour between the "
        "work hours of 9:00 to 17:00 on either Monday or Tuesday.",
        "Alice is free the entire day;\nBob is free the entire day;",
        "Bob can not meet on Monday. Tuesday before 11:00.",
        days_word="days",
    )
    task = parse_task(text)
    assert [(p.days, p.bound, p.bound_minutes) for p in task.hard_preferences] == [
        (("Monday",), None, None),
        (("Tuesday",), "before", 660),
    ]
    scorer = build_task_quality_scorer(text)
    assert scorer.valid_slots[0] == ("Tuesday", 660, 690)
    assert all(day == "Tuesday" for day, _, _ in scorer.valid_slots)


def test_score_ranges_never_overlap():
    text = _task(
        _HEADER_MON.format(a="Alice", b="Bob"),
        "Alice has meetings on Monday during 9:00 to 9:30;\nBob is free the entire day;",
        "Alice would rather not meet on Monday after 16:00.",
    )
    scorer = build_task_quality_scorer(text)
    invalid = [
        scorer.score_prediction(_answer("Monday", "9:00 - 9:30")),  # busy
        scorer.score_prediction(_answer("Tuesday", "10:00 - 10:30")),  # wrong day
        scorer.score_prediction(_answer("Monday", "10:00 - 11:00")),  # wrong duration
        scorer.score_prediction(_answer("Monday", "8:00 - 8:30")),  # outside work hours
        scorer.score_prediction("no time here"),
    ]
    valid = [
        scorer.score_prediction(_answer("Monday", "10:00 - 10:30")),
        scorer.score_prediction(_answer("Monday", "16:30 - 17:00")),  # violates the soft one
    ]
    assert all(not r.hard_valid and r.score <= 0.49 * 4 / 5 for r in invalid)
    assert all(r.hard_valid and 0.5 <= r.score <= 1.0 for r in valid)


def test_deposit_quality_modes():
    text = _task(
        _HEADER_MON.format(a="Alice", b="Bob"),
        "Alice has meetings on Monday during 9:00 to 9:30;\nBob is free the entire day;",
    )
    scorer = build_task_quality_scorer(text)
    valid = scorer.score_prediction(_answer("Monday", "10:00 - 10:30"))
    invalid = scorer.score_prediction(_answer("Monday", "9:00 - 9:30"))
    assert invalid.score > 0.0
    assert deposit_quality(invalid, "v1_score") == invalid.score  # v1 rule: wrong answers deposit
    assert deposit_quality(invalid, "valid_only") == 0.0
    assert deposit_quality(valid, "valid_only") == valid.score == deposit_quality(valid, "v1_score")
    with pytest.raises(ValueError):
        deposit_quality(valid, "nope")


def test_parser_reports_what_it_cannot_explain():
    header = _HEADER_MON.format(a="Alice", b="Bob")
    ok = _task(header, "Alice is free the entire week.\nBob is free the entire day;")
    assert parse_task(ok).warnings == ()

    stranger = _task(
        header,
        "Alice is free the entire day;\nBob is free the entire day;\n"
        "Carol has meetings on Monday during 9:00 to 10:00;",
    )
    assert any("unknown participant" in w for w in parse_task(stranger).warnings)

    missing = _task(header, "Alice is free the entire day;")
    assert any("no schedule statement" in w for w in parse_task(missing).warnings)

    garbled = _task(
        header, "Alice is free the entire day;\nBob keeps Monday during 9:00 to 10:00;"
    )
    warnings = parse_task(garbled).warnings
    assert any("unparsed schedule statement" in w for w in warnings)
    assert any("busy intervals in the text" in w for w in warnings)


def test_unparsable_allowed_days_raise_instead_of_defaulting_to_monday():
    text = _task(
        "You need to schedule a meeting for Alice and Bob for half an hour between the "
        "work hours of 9:00 to 17:00 on the weekend.",
        "Alice is free the entire day;\nBob is free the entire day;",
    )
    with pytest.raises(ValueError):
        parse_task(text)
