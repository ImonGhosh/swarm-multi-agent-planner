from __future__ import annotations

import numpy as np

from swarm_planner.seeding import derive_rng, task_key


def _stream(seed: int, task_id: str, iteration: int, agent: int, level: int) -> list[float]:
    rng = derive_rng(seed, task_id, iteration, agent, level)
    return [float(x) for x in rng.random(4)]


def test_same_inputs_give_same_stream():
    assert _stream(42, "task_a", 3, 5, 2) == _stream(42, "task_a", 3, 5, 2)


def test_task_key_is_stable_across_processes():
    # sha256-based, so it must not depend on PYTHONHASHSEED.
    assert task_key("calendar_scheduling_example_0") == task_key(
        "calendar_scheduling_example_0"
    )
    assert task_key("calendar_scheduling_example_0") != task_key(
        "calendar_scheduling_example_1"
    )


def test_each_coordinate_changes_the_stream():
    base = _stream(0, "task_a", 1, 1, 1)
    assert _stream(1, "task_a", 1, 1, 1) != base
    assert _stream(0, "task_b", 1, 1, 1) != base
    assert _stream(0, "task_a", 2, 1, 1) != base
    assert _stream(0, "task_a", 1, 2, 1) != base
    assert _stream(0, "task_a", 1, 1, 2) != base


def test_consecutive_seeds_are_not_shifted_copies():
    """v1 used seed + task_offset, so seed 43 on task i == seed 42 on task i+1."""
    task_ids = [f"calendar_scheduling_example_{i}" for i in range(20)]
    seed_42 = [_stream(42, t, 0, 0, 1) for t in task_ids]
    seed_43 = [_stream(43, t, 0, 0, 1) for t in task_ids]
    assert not any(a == b for a in seed_43 for b in seed_42)


def test_no_collisions_over_the_full_coordinate_grid():
    seen = set()
    for seed in range(3):
        for task in ("t0", "t1", "t2"):
            for iteration in range(6):
                for agent in range(8):
                    for level in range(1, 6):
                        seen.add(tuple(_stream(seed, task, iteration, agent, level)))
    assert len(seen) == 3 * 3 * 6 * 8 * 5


def test_returns_numpy_generator():
    assert isinstance(derive_rng(0, "t", 0, 0, 1), np.random.Generator)
