from __future__ import annotations

from aco_tot.task_quality import build_task_quality_scorer


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
