"""Natural Plan exact match, as ``scripts/evaluate.py`` (the official evaluator) computes it.

The official script reads a whole predictions file through absl flags. This module repeats its
parsing and comparison so that many prediction sets (every selection rule of every run) can be
scored in memory. ``tests/test_metrics.py`` checks that both give identical numbers.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

_TIME_PATTERN = re.compile(r"[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+")


def _hour_to_num(hr_str: str) -> float:
    return float(hr_str.split(":")[0]) + (0.5 if hr_str.split(":")[1] == "30" else 0.0)


def parse_response(response: str) -> Tuple[str, float, float]:
    """(day, start_hour, end_hour) of the first time in the text, or ('', -1, -1)."""
    time_strs = _TIME_PATTERN.findall(response)
    if not time_strs:
        return "", -1, -1
    time_str = time_strs[0]
    day, hour_str = time_str.split(",")[0].strip(), time_str.split(",")[1].strip()
    start_hour, end_hour = hour_str.split("-")[0].strip(), hour_str.split("-")[1].strip()
    return day, _hour_to_num(start_hour), _hour_to_num(end_hour)


def exact_match(response: str, solution: str) -> bool:
    return parse_response(response) == parse_response(solution)


def correctness(responses: Iterable[str], solutions: Iterable[str]) -> List[bool]:
    return [exact_match(r, s) for r, s in zip(responses, solutions)]


def solve_rate(responses: Iterable[str], solutions: Iterable[str]) -> float:
    flags = correctness(responses, solutions)
    return sum(flags) / len(flags) if flags else 0.0
