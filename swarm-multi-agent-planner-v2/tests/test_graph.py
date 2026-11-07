from __future__ import annotations

import asyncio

import swarm_planner.tot.graph as graph_module
from fakes import FakeClient
from swarm_planner.aco.pheromone import create_initial_pheromone_state
from swarm_planner.llm.gateway import LLMGateway

FINAL = "Here is the proposed time: Monday, 9:00 - 9:30"


def _initial_state(*, seed: int, agent_id: int, use_aco: bool) -> dict:
    pheromone_state = create_initial_pheromone_state()
    return {
        "original_prompt": "TASK: demo\nFind a time that works.\nSOLUTION:",
        "pheromones": dict(pheromone_state.pheromones),
        "alpha": 1.0,
        "epsilon": 0.001,
        "use_aco": use_aco,
        "seed": seed,
        "task_id": "t1",
        "iteration": 0,
        "agent_id": agent_id,
        "selected_path": [],
        "thought_results": [],
        "path_score": 1.0,
    }


def _run(state: dict) -> dict:
    client = FakeClient(
        lambda messages: FINAL
        if "Produce one final answer" in messages[0]["content"]
        else "intermediate"
    )
    graph = graph_module.build_agent_graph(LLMGateway(model="dummy-model", client=client))
    return asyncio.run(graph.ainvoke(state))


def test_graph_preserves_selected_path_and_scores():
    output = _run(_initial_state(seed=42, agent_id=0, use_aco=True))

    assert output["final_answer"] == FINAL
    assert len(output["selected_path"]) == 5
    assert len(output["thought_results"]) == 5
    assert output["path_score"] > 0.0


def test_graph_records_branch_probabilities_and_request_hashes():
    for use_aco in (True, False):
        output = _run(_initial_state(seed=5, agent_id=1, use_aco=use_aco))
        for item in output["thought_results"]:
            probabilities = item["probabilities"]
            assert len(probabilities) == 3
            assert abs(sum(probabilities.values()) - 1.0) < 1e-9
            # the probability of the chosen branch is the one that was recorded
            assert probabilities[item["branch_id"]] == item["selection_probability"]
            assert len(item["request_hash"]) == 16
        if not use_aco:
            assert all(
                abs(p - 1 / 3) < 1e-9
                for item in output["thought_results"]
                for p in item["probabilities"].values()
            )
        assert len(output["final_request_hash"]) == 16


def test_path_is_reproducible_for_same_seed_and_differs_across_agents():
    first = _run(_initial_state(seed=1, agent_id=3, use_aco=False))
    again = _run(_initial_state(seed=1, agent_id=3, use_aco=False))
    assert first["selected_path"] == again["selected_path"]

    paths = {
        tuple(_run(_initial_state(seed=1, agent_id=a, use_aco=False))["selected_path"])
        for a in range(8)
    }
    assert len(paths) > 1
