"""Per-call random streams derived from (seed, task, iteration, agent, level).

v1 built per-call seeds additively (``seed + task_offset``, ``+ agent_id * 100003``,
...), so streams of different seeds collided: seed 43 on task i was seed 42 on
task i+1. Here every call gets its own stream from ``numpy.random.SeedSequence``,
which hashes the whole entropy tuple, so distinct tuples give independent streams.
"""

from __future__ import annotations

import hashlib

import numpy as np


def task_key(task_id: str) -> int:
    """Stable 64-bit integer for a task id (Python's ``hash`` is salted per process)."""
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_seed_sequence(
    seed: int,
    task_id: str,
    iteration: int,
    agent: int,
    level: int,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [int(seed), task_key(task_id), int(iteration), int(agent), int(level)]
    )


def derive_rng(
    seed: int,
    task_id: str,
    iteration: int,
    agent: int,
    level: int,
) -> np.random.Generator:
    """Generator for one branch decision (one agent, one level, one iteration)."""
    return np.random.default_rng(
        derive_seed_sequence(seed, task_id, iteration, agent, level)
    )
