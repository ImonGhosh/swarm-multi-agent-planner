"""Prompt parsing and output formatting helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from .config import FINAL_INSTRUCTION
from .types import TaskFeatures


TASK_MARKER = "\n\nTASK:"
SOLUTION_MARKER = "\nSOLUTION:"


def remove_last_final_instruction(prompt: str) -> str:
    """Remove only the final occurrence of the canonical final instruction."""
    text = (prompt or "").strip()
    idx = text.rfind(FINAL_INSTRUCTION)
    if idx == -1:
        return text
    return (text[:idx] + text[idx + len(FINAL_INSTRUCTION) :]).strip()


def build_root_prompt_from_prompt5shot(prompt_5shot: str) -> str:
    """Build original/root prompt as prompt_5shot without final instruction."""
    text = (prompt_5shot or "").strip()
    text = re.sub(r"\s*SOLUTION:\s*$", "", text, flags=re.IGNORECASE)
    text = remove_last_final_instruction(text)
    return text.strip()


def extract_current_task_from_prompt5shot(prompt_5shot: str) -> str:
    """Extract only the final TASK block, without SOLUTION suffix."""
    prompt = (prompt_5shot or "").strip()
    if not prompt:
        return ""
    last_task_idx = prompt.rfind(TASK_MARKER)
    if last_task_idx == -1:
        return remove_last_final_instruction(re.sub(r"\s*SOLUTION:\s*$", "", prompt))
    tail = prompt[last_task_idx + len(TASK_MARKER) :].lstrip()
    solution_idx = tail.rfind(SOLUTION_MARKER)
    if solution_idx != -1:
        tail = tail[:solution_idx]
    tail = remove_last_final_instruction(tail)
    return f"TASK: {tail.strip()}".strip()


def _extract_duration_hours(text: str) -> float:
    lower = text.lower()
    if "half an hour" in lower or "30-minute" in lower or "30 minute" in lower:
        return 0.5
    if "one hour" in lower or "1 hour" in lower or "60-minute" in lower:
        return 1.0
    return 0.5


def _has_preference(text: str) -> bool:
    lower = text.lower()
    preference_tokens = [
        "prefer",
        "preference",
        "rather",
        "avoid",
        "do not",
        "don't",
        "earliest",
        "earlist",
        "would like",
    ]
    return any(token in lower for token in preference_tokens)


def _has_earliest_preference(text: str) -> bool:
    lower = text.lower()
    return "earliest" in lower or "earlist" in lower


def build_task_features(
    task_id: str,
    task_prompt: str,
    input_meta: Dict[str, Any] | None = None,
) -> TaskFeatures:
    """Build task features required by heuristic desirability rules."""
    meta = input_meta or {}
    num_people = int(meta.get("num_people", 2))
    num_days = int(meta.get("num_days", 1))
    feature_text = task_prompt
    if feature_text.count("TASK:") >= 2:
        feature_text = extract_current_task_from_prompt5shot(feature_text)

    duration_raw = meta.get("duration")
    if duration_raw is None:
        duration_hours = _extract_duration_hours(feature_text)
    else:
        duration_hours = float(duration_raw)

    return TaskFeatures(
        task_id=task_id,
        num_people=num_people,
        num_days=num_days,
        duration_hours=duration_hours,
        has_preference=_has_preference(feature_text),
        has_earliest_preference=_has_earliest_preference(feature_text),
    )


def parse_proposed_time(text: str) -> Tuple[str, str, str] | None:
    """Parse 'Day, HH:MM - HH:MM' from a free-form answer."""
    if not text:
        return None
    match = re.search(
        r"([A-Za-z]+)\s*,\s*([0-9]{1,2}:[0-9]{2})\s*-\s*([0-9]{1,2}:[0-9]{2})",
        text,
    )
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def canonicalize_final_answer(text: str) -> str:
    """Ensure final answer follows evaluator-friendly canonical format."""
    parsed = parse_proposed_time(text or "")
    if not parsed:
        fallback = (text or "").strip()
        if fallback:
            return fallback
        return "Here is the proposed time: Monday, 9:00 - 9:30"
    day, start, end = parsed
    return f"Here is the proposed time: {day}, {start} - {end}"
