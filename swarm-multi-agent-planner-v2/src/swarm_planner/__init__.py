"""Decentralized SLM swarm reasoning (ToT + ACO) for constrained calendar planning."""

__all__ = ["solve_dataset", "solve_task"]


def solve_dataset(*args, **kwargs):
    from .solver import solve_dataset as _solve_dataset

    return _solve_dataset(*args, **kwargs)


async def solve_task(*args, **kwargs):
    from .solver import solve_task as _solve_task

    return await _solve_task(*args, **kwargs)
