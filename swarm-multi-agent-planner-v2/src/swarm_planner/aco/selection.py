"""ACO branch selection: p_ij = (tau_ij + eps)^alpha / sum_k (tau_ik + eps)^alpha."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..tot.levels import edge_key


def compute_selection_probabilities(
    *,
    level: int,
    branch_ids: Iterable[str],
    pheromones: Dict[str, float],
    alpha: float,
    epsilon: float,
) -> Dict[str, float]:
    """Edge probabilities of one level. All pheromones equal gives the uniform distribution."""
    numerators: Dict[str, float] = {}
    for branch_id in branch_ids:
        tau_ij = pheromones.get(edge_key(level, branch_id), 1.0)
        numerators[branch_id] = (tau_ij + epsilon) ** alpha

    denominator = sum(numerators.values())
    if denominator <= 0:
        size = len(numerators)
        return {branch_id: 1.0 / size for branch_id in numerators}

    return {branch_id: value / denominator for branch_id, value in numerators.items()}


def sample_branch(
    probabilities: Dict[str, float], rng: np.random.Generator
) -> Tuple[str, float]:
    """Sample one branch id according to probability distribution."""
    draw = float(rng.random())
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
    pheromones: Dict[str, float],
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[str, float, Dict[str, float]]:
    """Compute probabilities and sample a branch in one call."""
    probabilities = compute_selection_probabilities(
        level=level,
        branch_ids=branch_ids,
        pheromones=pheromones,
        alpha=alpha,
        epsilon=epsilon,
    )
    selected_branch, selected_prob = sample_branch(probabilities, rng)
    return selected_branch, selected_prob, probabilities
