"""Small statistics helpers for the Phase 4 tables (numpy only, no scipy)."""

from __future__ import annotations

from math import comb
from typing import Sequence, Tuple

import numpy as np


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample SD (n - 1) over seeds. SD is 0.0 for a single value."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def bootstrap_ci(
    per_task: np.ndarray, *, draws: int = 4000, seed: int = 5, level: float = 95.0
) -> Tuple[float, float]:
    """Percentile bootstrap interval of the mean over tasks (tasks are the resampling unit)."""
    per_task = np.asarray(per_task, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_task), size=(draws, len(per_task)))
    low, high = np.percentile(per_task[idx].mean(axis=1), [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(low), float(high)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs.

    ``only_a``: tasks that A solved and B did not; ``only_b``: the reverse.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0**n)


def discordant(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
    """(tasks only A solved, tasks only B solved) for two 0/1 arrays over the same tasks."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    return int((a & ~b).sum()), int((~a & b).sum())
