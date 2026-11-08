"""Persistence helpers for pheromone state."""

from __future__ import annotations

import json
from pathlib import Path

from .config import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_EPSILON,
    DEFAULT_INITIAL_PHEROMONE,
    DEFAULT_RHO,
    DEFAULT_SEED,
    PHEROMONE_VERSION,
    all_edge_keys,
)
from .types import PheromoneState


def create_initial_pheromone_state(
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    initial_tau: float = DEFAULT_INITIAL_PHEROMONE,
) -> PheromoneState:
    pheromones = {edge: float(initial_tau) for edge in all_edge_keys()}
    metadata = {
        "version": PHEROMONE_VERSION,
        "alpha": alpha,
        "beta": beta,
        "epsilon": epsilon,
        "rho": rho,
        "seed": seed,
    }
    return PheromoneState(metadata=metadata, pheromones=pheromones)


def _normalize_loaded_state(raw: dict) -> PheromoneState:
    metadata = dict(raw.get("metadata", {}))
    pheromones = {k: float(v) for k, v in dict(raw.get("pheromones", {})).items()}
    for edge in all_edge_keys():
        pheromones.setdefault(edge, DEFAULT_INITIAL_PHEROMONE)
    return PheromoneState(metadata=metadata, pheromones=pheromones)


def load_pheromone_state(
    path: str | Path,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    initial_tau: float = DEFAULT_INITIAL_PHEROMONE,
) -> PheromoneState:
    file_path = Path(path)
    if not file_path.exists():
        return create_initial_pheromone_state(
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
            initial_tau=initial_tau,
        )
    with file_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    loaded = _normalize_loaded_state(raw)
    loaded.metadata.setdefault("alpha", alpha)
    loaded.metadata.setdefault("beta", beta)
    loaded.metadata.setdefault("epsilon", epsilon)
    loaded.metadata.setdefault("rho", rho)
    loaded.metadata.setdefault("seed", seed)
    loaded.metadata.setdefault("version", PHEROMONE_VERSION)
    return loaded


def save_pheromone_state(path: str | Path, pheromone_state: PheromoneState) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(pheromone_state.to_dict(), handle, ensure_ascii=False, indent=2)
