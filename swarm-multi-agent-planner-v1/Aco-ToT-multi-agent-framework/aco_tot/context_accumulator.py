"""Prompt accumulation logic for ToT traversal."""

from __future__ import annotations

from typing import Iterable

from .types import ThoughtResult


def _format_history(thought_results: Iterable[ThoughtResult]) -> str:
    lines = []
    for idx, item in enumerate(thought_results, start=1):
        lines.append(f"Thought {idx}-")
        lines.append(item.thought_prompt)
        lines.append("")
        lines.append(f"Intermediate Result for Thought {idx}-")
        lines.append(item.intermediate_result.strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt_for_level(
    *,
    original_prompt: str,
    previous_results: Iterable[ThoughtResult],
    current_thought_prompt: str,
) -> str:
    """Build accumulated prompt for one level before intermediate generation."""
    parts = [original_prompt.strip(), ""]
    history = _format_history(previous_results)
    if history:
        parts.append(history)
        parts.append("")
    parts.append("Latest Thought-")
    parts.append(current_thought_prompt.strip())
    return "\n".join(parts).strip()


def build_prompt_for_final(
    *,
    original_prompt: str,
    thought_results: Iterable[ThoughtResult],
    final_instruction: str,
) -> str:
    """Build final accumulated prompt at leaf before final answer generation."""
    parts = [original_prompt.strip(), ""]
    history = _format_history(thought_results)
    if history:
        parts.append(history)
        parts.append("")
    parts.append("Latest Thought-")
    parts.append(final_instruction.strip())
    return "\n".join(parts).strip()
