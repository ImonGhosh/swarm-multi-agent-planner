"""Pheromone state, evaporation and the top-only deposit."""

from __future__ import annotations

from typing import Iterable

from ..tot.levels import all_edge_keys
from ..types import AgentTrace, PheromoneState
from . import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_INITIAL_PHEROMONE,
    DEFAULT_RHO,
)

# The v2 rule (decision 10): only an answer the verifier scores 1.0 deposits pheromone.
TOP_SCORE = 1.0
DEPOSIT_PER_VERIFIED_TRACE = 1.0


def create_initial_pheromone_state(
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = 0,
    initial_tau: float = DEFAULT_INITIAL_PHEROMONE,
) -> PheromoneState:
    pheromones = {edge: float(initial_tau) for edge in all_edge_keys()}
    metadata = {
        "alpha": alpha,
        "epsilon": epsilon,
        "rho": rho,
        "seed": seed,
    }
    return PheromoneState(metadata=metadata, pheromones=pheromones)


def trace_deposit(trace: AgentTrace) -> float:
    """Q of one trace: 1 if the verifier scored its answer 1.0, else 0 (top-only deposit)."""
    return DEPOSIT_PER_VERIFIED_TRACE if float(trace.quality_score) >= TOP_SCORE else 0.0


def apply_pheromone_update(
    *,
    pheromone_state: PheromoneState,
    traces: Iterable[AgentTrace],
    rho: float,
) -> PheromoneState:
    """
    Update each edge pheromone:
      tau_ij <- (1-rho)*tau_ij + sum_k Delta_ij^(k)
      Delta_ij^(k) = Q_k if edge in P_k else 0,   Q_k = 1 iff the verifier scores trace k 1.0
    """
    traces = list(traces)
    updated = dict(pheromone_state.pheromones)

    for edge in all_edge_keys():
        tau_ij = updated.get(edge, 1.0)
        deposit = 0.0
        for trace in traces:
            if edge in trace.path:
                deposit += trace_deposit(trace)
        updated[edge] = ((1.0 - rho) * tau_ij) + deposit

    metadata = dict(pheromone_state.metadata)
    metadata["rho"] = rho
    return PheromoneState(metadata=metadata, pheromones=updated)
