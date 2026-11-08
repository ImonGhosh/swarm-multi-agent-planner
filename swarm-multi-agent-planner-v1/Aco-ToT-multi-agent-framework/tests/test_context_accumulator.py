from __future__ import annotations

from aco_tot.context_accumulator import build_prompt_for_final, build_prompt_for_level
from aco_tot.types import ThoughtResult


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
