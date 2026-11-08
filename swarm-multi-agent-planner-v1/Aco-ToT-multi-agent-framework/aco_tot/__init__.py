"""ACO-ToT calendar scheduling package."""

__all__ = ["invoke", "train_dataset", "infer_dataset", "solve_dataset"]


def train_dataset(*args, **kwargs):
    from .engine import train_dataset as _train_dataset

    return _train_dataset(*args, **kwargs)


def infer_dataset(*args, **kwargs):
    from .engine import infer_dataset as _infer_dataset

    return _infer_dataset(*args, **kwargs)


def solve_dataset(*args, **kwargs):
    from .engine import solve_dataset as _solve_dataset

    return _solve_dataset(*args, **kwargs)


async def invoke(*args, **kwargs):
    from .invoke import invoke as _invoke

    return await _invoke(*args, **kwargs)
