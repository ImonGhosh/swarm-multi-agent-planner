"""Pheromone evaporation and deposit updates."""

from __future__ import annotations

from typing import Iterable

from .config import all_edge_keys
from .types import AgentTrace, PheromoneState


def apply_pheromone_update(
    *,
    pheromone_state: PheromoneState,
    traces: Iterable[AgentTrace],
    rho: float,
) -> PheromoneState:
    """
    Update each edge pheromone:
      tau_ij <- (1-rho)*tau_ij + sum_k Delta_ij^(k)
      Delta_ij^(k) = Q_k if edge in P_k else 0
    """
    traces = list(traces)
    updated = dict(pheromone_state.pheromones)

    for edge in all_edge_keys():
        tau_ij = updated.get(edge, 1.0)
        deposit = 0.0
        for trace in traces:
            if edge in trace.path:
                deposit += trace.quality_score
        updated[edge] = ((1.0 - rho) * tau_ij) + deposit

    metadata = dict(pheromone_state.metadata)
    metadata["rho"] = rho
    return PheromoneState(metadata=metadata, pheromones=updated)
