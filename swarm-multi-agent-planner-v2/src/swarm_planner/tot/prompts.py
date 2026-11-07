"""The exact chat messages the ToT sends to the LLM.

The solver's graph and the path enumerator both build their requests here, so a request is
byte-identical wherever it is made. That is what lets a path table, a ToT-only run and a
ToT+ACO run share cached answers.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from ..types import ThoughtResult
from .context import build_prompt_for_final, build_prompt_for_level
from .levels import LEAF_BEHAVIOR_PROMPTS, BranchConfig, get_level_config

Messages = List[Dict[str, str]]

EMPTY_INTERMEDIATE = "No intermediate result generated; carry forward parsed constraints only."


def level_messages(
    *,
    level: int,
    original_prompt: str,
    previous_results: Iterable[ThoughtResult],
    branch: BranchConfig,
) -> Messages:
    """Messages for the intermediate-result call of one level and branch."""
    level_config = get_level_config(level)
    prompt = build_prompt_for_level(
        original_prompt=original_prompt,
        previous_results=previous_results,
        current_thought_prompt=branch.thought_prompt,
    )
    system_prompt = (
        "You are solving a calendar scheduling task using a fixed Tree-of-Thought path. "
        f"Current level {level}: {level_config.name}. "
        f"Level goal: {level_config.goal}. "
        "Return only the intermediate result for the thought provided in the user input. "
        "Do not provide final solution at this step. "
    )
    user_prompt = (
        f"{prompt}\n\n"
        "Provide a concise intermediate result for the latest thought only, for the last unsolved task."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def final_messages(
    *,
    original_prompt: str,
    thought_results: List[ThoughtResult],
) -> Messages:
    """Messages for the finalize call at the leaf of a path."""
    leaf_branch_id = thought_results[-1].branch_id
    leaf_instruction = LEAF_BEHAVIOR_PROMPTS.get(
        leaf_branch_id,
        "Verify once and return a final answer.",
    )
    prompt = build_prompt_for_final(
        original_prompt=original_prompt,
        thought_results=thought_results,
        final_instruction=leaf_instruction,
    )
    system_prompt = (
        "You are a calendar scheduling solver. "
        "Produce one final answer using format exactly: "
        "Here is the proposed time: <Day>, <HH:MM> - <HH:MM>. "
        "Return only one line."
    )
    user_prompt = f"{prompt}\n\nReturn the final proposed slot now."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def clean_intermediate(text: str) -> str:
    """The intermediate result that is carried into the next level's prompt."""
    cleaned = (text or "").strip()
    return cleaned if cleaned else EMPTY_INTERMEDIATE
