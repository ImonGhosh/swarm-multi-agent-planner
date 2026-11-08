from __future__ import annotations

import asyncio

import aco_tot.llm_nodes as llm_nodes
from aco_tot.store import create_initial_pheromone_state
from aco_tot.types import TaskFeatures


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    async def ainvoke(self, messages):  # type: ignore[no-untyped-def]
        system_prompt = getattr(messages[0], "content", "")
        if "Produce one final answer" in system_prompt:
            return _FakeResponse("Here is the proposed time: Monday, 9:00 - 9:30")
        return _FakeResponse("intermediate")


def test_graph_preserves_selected_path_and_scores(monkeypatch):  # type: ignore[no-untyped-def]
    llm_nodes._GRAPH_CACHE.clear()
    llm_nodes._LLM_CACHE.clear()
    monkeypatch.setattr(llm_nodes, "_get_llm", lambda model: _FakeLLM())

    pheromone_state = create_initial_pheromone_state()
    graph = llm_nodes.build_agent_graph("dummy-model")
    output = asyncio.run(
        graph.ainvoke(
            {
                "original_prompt": "TASK: demo\nFind a time that works.\nSOLUTION:",
                "task_features": TaskFeatures(
                    task_id="t1",
                    num_people=2,
                    num_days=1,
                    duration_hours=0.5,
                    has_preference=False,
                    has_earliest_preference=False,
                ).__dict__,
                "pheromones": dict(pheromone_state.pheromones),
                "alpha": 1.0,
                "beta": 2.0,
                "epsilon": 0.001,
                "use_aco": True,
                "agent_seed": 42,
                "selected_path": [],
                "thought_results": [],
                "path_score": 1.0,
            }
        )
    )

    assert output["final_answer"] == "Here is the proposed time: Monday, 9:00 - 9:30"
    assert len(output["selected_path"]) == 5
    assert len(output["thought_results"]) == 5
    assert output["path_score"] > 0.0
