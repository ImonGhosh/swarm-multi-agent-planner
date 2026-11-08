from __future__ import annotations

from aco_tot.prompt_io import (
    build_root_prompt_from_prompt5shot,
    extract_current_task_from_prompt5shot,
    remove_last_final_instruction,
)


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
