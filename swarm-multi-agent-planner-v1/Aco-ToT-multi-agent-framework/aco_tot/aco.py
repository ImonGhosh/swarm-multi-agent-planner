"""ACO branch selection utilities."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

from .config import edge_key
from .heuristics import heuristic_desirability
from .types import TaskFeatures


def compute_selection_probabilities(
    *,
    level: int,
    branch_ids: Iterable[str],
    task_features: TaskFeatures,
    pheromones: Dict[str, float],
    alpha: float,
    beta: float,
    epsilon: float,
) -> Dict[str, float]:
    """Compute ACO edge probabilities for one level."""
    numerators: Dict[str, float] = {}
    for branch_id in branch_ids:
        edge = edge_key(level, branch_id)
        tau_ij = pheromones.get(edge, 1.0)
        eta_ij = heuristic_desirability(level, branch_id, task_features)
        numerators[branch_id] = ((tau_ij + epsilon) ** alpha) * (
            (eta_ij + epsilon) ** beta
        )

    denominator = sum(numerators.values())
    if denominator <= 0:
        size = len(numerators)
        return {branch_id: 1.0 / size for branch_id in numerators}

    return {branch_id: value / denominator for branch_id, value in numerators.items()}


def sample_branch(probabilities: Dict[str, float], rng: random.Random) -> Tuple[str, float]:
    """Sample one branch id according to probability distribution."""
    draw = rng.random()
    cumulative = 0.0
    last_key = ""
    for branch_id, prob in probabilities.items():
        last_key = branch_id
        cumulative += prob
        if draw <= cumulative:
            return branch_id, prob
    return last_key, probabilities[last_key]


def select_branch(
    *,
    level: int,
    branch_ids: List[str],
    task_features: TaskFeatures,
    pheromones: Dict[str, float],
    alpha: float,
    beta: float,
    epsilon: float,
    rng: random.Random,
) -> Tuple[str, float, Dict[str, float]]:
    """Compute probabilities and sample a branch in one call."""
    probabilities = compute_selection_probabilities(
        level=level,
        branch_ids=branch_ids,
        task_features=task_features,
        pheromones=pheromones,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
    )
    selected_branch, selected_prob = sample_branch(probabilities, rng)
    return selected_branch, selected_prob, probabilities
