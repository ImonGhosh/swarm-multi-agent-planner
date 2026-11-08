from __future__ import annotations

import random

from aco_tot.aco import compute_selection_probabilities, select_branch
from aco_tot.config import all_edge_keys, edge_key
from aco_tot.pheromone import apply_pheromone_update
from aco_tot.store import create_initial_pheromone_state
from aco_tot.types import AgentTrace, TaskFeatures, ThoughtResult


def _features() -> TaskFeatures:
    return TaskFeatures(
        task_id="t",
        num_people=4,
        num_days=2,
        duration_hours=0.5,
        has_preference=True,
        has_earliest_preference=False,
    )


def test_probabilities_sum_to_one():
    pheromone_state = create_initial_pheromone_state()
    probs = compute_selection_probabilities(
        level=1,
        branch_ids=["constraint_first", "people_first", "preference_first"],
        task_features=_features(),
        pheromones=pheromone_state.pheromones,
        alpha=1.0,
        beta=2.0,
        epsilon=0.001,
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert all(prob > 0 for prob in probs.values())


def test_select_branch_returns_valid_choice():
    pheromone_state = create_initial_pheromone_state()
    branch, selected_prob, probs = select_branch(
        level=3,
        branch_ids=[
            "free_slot_computation",
            "candidate_slot_enumeration",
            "progressive_elimination",
        ],
        task_features=_features(),
        pheromones=pheromone_state.pheromones,
        alpha=1.0,
        beta=2.0,
        epsilon=0.001,
        rng=random.Random(123),
    )
    assert branch in probs
    assert selected_prob == probs[branch]


def test_pheromone_update_evaporation_and_deposit():
    state = create_initial_pheromone_state(rho=0.1)
    chosen_edge = edge_key(1, "constraint_first")
    trace = AgentTrace(
        agent_id=0,
        path=[chosen_edge],
        thought_results=[
            ThoughtResult(
                level=1,
                branch_id="constraint_first",
                branch_name="Constraint-first",
                thought_prompt="x",
                intermediate_result="y",
                selection_probability=1.0,
            )
        ],
        final_answer="Here is the proposed time: Monday, 9:00 - 9:30",
        path_score=1.0,
        quality_score=1.0,
    )

    updated = apply_pheromone_update(
        pheromone_state=state,
        traces=[trace],
        rho=0.1,
    )

    assert abs(updated.pheromones[chosen_edge] - ((1.0 - 0.1) * 1.0 + 1.0)) < 1e-9
    untouched_edge = next(edge for edge in all_edge_keys() if edge != chosen_edge)
    assert abs(updated.pheromones[untouched_edge] - ((1.0 - 0.1) * 1.0)) < 1e-9
