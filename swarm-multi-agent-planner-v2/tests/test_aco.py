from __future__ import annotations

import numpy as np
import pytest

from swarm_planner.aco.pheromone import (
    apply_pheromone_update,
    create_initial_pheromone_state,
    trace_deposit,
)
from swarm_planner.aco.selection import compute_selection_probabilities, select_branch
from swarm_planner.tot.levels import all_edge_keys, edge_key
from swarm_planner.types import AgentTrace
from swarm_planner.verifier.scorer import DeterministicQualityResult, deposit_quality

LEVEL1 = ["constraint_first", "people_first", "preference_first"]


def _trace(path, quality: float) -> AgentTrace:
    return AgentTrace(
        agent_id=0,
        path=list(path),
        thought_results=[],
        final_answer="Here is the proposed time: Monday, 9:00 - 9:30",
        path_score=1.0,
        quality_score=quality,
    )


def test_probabilities_sum_to_one_and_start_uniform():
    state = create_initial_pheromone_state()
    probs = compute_selection_probabilities(
        level=1, branch_ids=LEVEL1, pheromones=state.pheromones, alpha=1.0, epsilon=0.001
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert all(p == pytest.approx(1 / 3) for p in probs.values())  # no eta: nothing favours a branch


def test_probability_follows_the_formula_without_eta():
    pheromones = {edge_key(1, "constraint_first"): 3.0, edge_key(1, "people_first"): 1.0,
                  edge_key(1, "preference_first"): 0.5}
    probs = compute_selection_probabilities(
        level=1, branch_ids=LEVEL1, pheromones=pheromones, alpha=1.0, epsilon=0.001
    )
    weights = {b: pheromones[edge_key(1, b)] + 0.001 for b in LEVEL1}
    total = sum(weights.values())
    for branch in LEVEL1:
        assert probs[branch] == pytest.approx(weights[branch] / total)
    # alpha = 0 switches pheromone off
    flat = compute_selection_probabilities(
        level=1, branch_ids=LEVEL1, pheromones=pheromones, alpha=0.0, epsilon=0.001
    )
    assert all(p == pytest.approx(1 / 3) for p in flat.values())


def test_select_branch_returns_valid_choice():
    state = create_initial_pheromone_state()
    branch, selected_prob, probs = select_branch(
        level=3,
        branch_ids=["free_slot_computation", "candidate_slot_enumeration", "progressive_elimination"],
        pheromones=state.pheromones,
        alpha=1.0,
        epsilon=0.001,
        rng=np.random.default_rng(123),
    )
    assert branch in probs
    assert selected_prob == probs[branch]


def test_verified_trace_deposits_and_evaporation_applies_everywhere():
    state = create_initial_pheromone_state(rho=0.1)
    chosen_edge = edge_key(1, "constraint_first")
    updated = apply_pheromone_update(pheromone_state=state, traces=[_trace([chosen_edge], 1.0)], rho=0.1)

    assert updated.pheromones[chosen_edge] == pytest.approx(0.9 * 1.0 + 1.0)
    untouched_edge = next(edge for edge in all_edge_keys() if edge != chosen_edge)
    assert updated.pheromones[untouched_edge] == pytest.approx(0.9 * 1.0)


@pytest.mark.parametrize("score", [0.0, 0.3, 0.392, 0.5, 0.75, 0.999999])
def test_only_a_score_of_one_deposits(score):
    """Top-only deposit (decision 10): invalid AND imperfect-but-valid answers deposit nothing."""
    state = create_initial_pheromone_state(rho=0.1)
    edge = edge_key(2, "day_wise")
    trace = _trace([edge], score)
    assert trace_deposit(trace) == 0.0
    updated = apply_pheromone_update(pheromone_state=state, traces=[trace], rho=0.1)
    assert updated.pheromones[edge] == pytest.approx(0.9)  # evaporation only


def test_deposit_adds_up_over_traces_and_matches_the_scorer_rule():
    state = create_initial_pheromone_state(rho=0.1)
    edge = edge_key(1, "people_first")
    traces = [_trace([edge], 1.0), _trace([edge], 1.0), _trace([edge], 0.8)]
    updated = apply_pheromone_update(pheromone_state=state, traces=traces, rho=0.1)
    assert updated.pheromones[edge] == pytest.approx(0.9 + 2.0)

    def result(score: float) -> DeterministicQualityResult:
        return DeterministicQualityResult(score, True, True, 5, 5, 0, 0, 1.0, 0, None, None)

    for score in (1.0, 0.8):
        assert deposit_quality(result(score), "top_only") == trace_deposit(_trace([edge], score))


def test_initial_state_covers_every_edge():
    state = create_initial_pheromone_state(alpha=1.0, epsilon=0.001, rho=0.1, seed=7)
    assert set(state.pheromones) == set(all_edge_keys())
    assert len(state.pheromones) == 15
    assert all(tau == 1.0 for tau in state.pheromones.values())
    assert state.metadata["seed"] == 7
    assert "beta" not in state.metadata  # eta and beta are gone from the method
