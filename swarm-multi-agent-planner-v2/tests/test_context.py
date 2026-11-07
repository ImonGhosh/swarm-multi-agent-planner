from __future__ import annotations

from swarm_planner.tot.context import (
    build_prompt_for_final,
    build_prompt_for_level,
    build_root_prompt_from_prompt5shot,
    extract_current_task_from_prompt5shot,
    remove_last_final_instruction,
)
from swarm_planner.types import ThoughtResult


def _history() -> list[ThoughtResult]:
    return [
        ThoughtResult(
            level=1,
            branch_id="constraint_first",
            branch_name="Constraint-first",
            thought_prompt="Extract constraints.",
            intermediate_result="participants=[A,B]",
            selection_probability=0.5,
        ),
        ThoughtResult(
            level=2,
            branch_id="participant_wise",
            branch_name="Participant-wise",
            thought_prompt="Build participant schedules.",
            intermediate_result="A_busy=[...], B_busy=[...]",
            selection_probability=0.4,
        ),
    ]


def test_level_prompt_accumulates_in_order():
    prompt = build_prompt_for_level(
        original_prompt="TASK: Demo scheduling task",
        previous_results=_history(),
        current_thought_prompt="Enumerate candidate slots.",
    )
    assert "TASK: Demo scheduling task" in prompt
    assert prompt.index("Extract constraints.") < prompt.index("Build participant schedules.")
    assert prompt.index("Build participant schedules.") < prompt.index("Enumerate candidate slots.")


def test_final_prompt_contains_all_history_and_final_instruction():
    prompt = build_prompt_for_final(
        original_prompt="TASK: Demo scheduling task",
        thought_results=_history(),
        final_instruction="Verify once and return final answer.",
    )
    assert "Extract constraints." in prompt
    assert "Build participant schedules." in prompt
    assert "Verify once and return final answer." in prompt


def test_remove_last_final_instruction_only():
    text = (
        "TASK: A\nFind a time that works for everyone's schedule and constraints.\n"
        "SOLUTION: X\nTASK: B\nFind a time that works for everyone's schedule and constraints."
    )
    cleaned = remove_last_final_instruction(text)
    assert cleaned.count("Find a time that works for everyone's schedule and constraints.") == 1


def test_extract_current_task_without_solution():
    prompt = (
        "Intro\n\nTASK: Example one.\nSOLUTION: A\n\n"
        "TASK: Real task line.\nFind a time that works for everyone's schedule and constraints.\nSOLUTION: "
    )
    current_task = extract_current_task_from_prompt5shot(prompt)
    assert current_task.startswith("TASK:")
    assert "Real task line." in current_task
    assert "SOLUTION:" not in current_task
    assert "Find a time that works for everyone's schedule and constraints." not in current_task


def test_build_root_prompt_from_prompt5shot():
    prompt = (
        "Header\n\nTASK: Example one.\nSOLUTION: A\n\n"
        "TASK: Real task line.\nFind a time that works for everyone's schedule and constraints.\nSOLUTION: "
    )
    root = build_root_prompt_from_prompt5shot(prompt)
    assert "SOLUTION:" not in root[-20:]
    assert "Find a time that works for everyone's schedule and constraints." not in root
