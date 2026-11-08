"""Public API entrypoints for ACO-ToT calendar scheduler."""

from __future__ import annotations

from typing import Any, Dict

from .config import (
    DEFAULT_AGENTS_PER_TASK,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_EPSILON,
    DEFAULT_INITIAL_PHEROMONE,
    DEFAULT_RHO,
    DEFAULT_SEED,
)
from .engine import infer_dataset, process_task, solve_dataset, train_dataset
from .store import create_initial_pheromone_state, load_pheromone_state, save_pheromone_state


async def invoke(
    task_prompt: str,
    model: str,
    input_meta: Dict[str, Any],
    mode: str,
    pheromone_path: str | None = None,
    golden_plan: str | None = None,
    agents_per_task: int = DEFAULT_AGENTS_PER_TASK,
    iterations_per_task: int = 3,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    aco_enabled: bool = True,
) -> str:
    """Run one task in train, infer, or solve mode and return final answer."""
    normalized_mode = (mode or "infer").strip().lower()
    if normalized_mode not in {"train", "infer", "solve"}:
        raise ValueError("mode must be one of: 'train', 'infer', 'solve'.")

    task_id = str(input_meta.get("task_id", "single_task"))
    if normalized_mode == "solve":
        pheromone_state = create_initial_pheromone_state(
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
            initial_tau=DEFAULT_INITIAL_PHEROMONE,
        )
    else:
        if not pheromone_path:
            raise ValueError("pheromone_path is required for train/infer invoke modes.")
        pheromone_state = load_pheromone_state(
            pheromone_path,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
        )

    result = await process_task(
        task_id=task_id,
        task_prompt=task_prompt,
        input_meta=input_meta,
        model=model,
        pheromone_state=pheromone_state,
        mode=normalized_mode,
        agents_per_task=agents_per_task,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
        seed=seed,
        aco_enabled=aco_enabled,
        golden_plan=golden_plan,
        iterations_per_task=iterations_per_task,
        scoring_task_text_override=(
            input_meta.get("prompt_0shot")
            if normalized_mode == "solve"
            else None
        ),
    )

    if (
        normalized_mode == "train"
        and result.pheromone_state is not None
        and pheromone_path is not None
    ):
        save_pheromone_state(pheromone_path, result.pheromone_state)

    return result.prediction


__all__ = ["invoke", "train_dataset", "infer_dataset", "solve_dataset"]
