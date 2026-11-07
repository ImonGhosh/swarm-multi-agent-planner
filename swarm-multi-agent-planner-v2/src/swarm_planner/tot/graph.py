"""LangGraph node builders for one agent's 5-level ToT traversal."""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from ..aco.selection import compute_selection_probabilities, sample_branch
from ..llm.gateway import LLMGateway
from ..seeding import derive_rng
from ..types import ThoughtResult
from ..verifier.parser import canonicalize_final_answer
from .levels import LEVEL_CONFIGS, edge_key, get_level_config
from .prompts import clean_intermediate, final_messages, level_messages


def _deserialize_thought_results(state: Dict[str, Any]) -> List[ThoughtResult]:
    raw = list(state.get("thought_results", []))
    results: List[ThoughtResult] = []
    for item in raw:
        if isinstance(item, ThoughtResult):
            results.append(item)
        elif isinstance(item, dict):
            results.append(ThoughtResult(**item))
    return results


def _call_meta(state: Dict[str, Any], step: str) -> Dict[str, Any]:
    return {
        "task_id": str(state.get("task_id", "")),
        "it": int(state.get("iteration", 0)),
        "agent": int(state.get("agent_id", 0)),
        "step": step,
    }


def _make_level_node(level: int, gateway: LLMGateway):
    level_config = get_level_config(level)

    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        pheromones = dict(state.get("pheromones", {}))
        alpha = float(state.get("alpha", 1.0))
        epsilon = float(state.get("epsilon", 0.001))
        use_aco = bool(state.get("use_aco", True))

        branch_ids = [branch.branch_id for branch in level_config.branches]
        rng = derive_rng(
            seed=int(state.get("seed", 0)),
            task_id=str(state.get("task_id", "")),
            iteration=int(state.get("iteration", 0)),
            agent=int(state.get("agent_id", 0)),
            level=level,
        )
        if use_aco:
            probabilities = compute_selection_probabilities(
                level=level,
                branch_ids=branch_ids,
                pheromones=pheromones,
                alpha=alpha,
                epsilon=epsilon,
            )
            selected_branch_id, selected_prob = sample_branch(probabilities, rng)
        else:
            selected_branch_id = branch_ids[int(rng.integers(len(branch_ids)))]
            selected_prob = 1.0 / float(len(branch_ids))
            probabilities = {branch_id: selected_prob for branch_id in branch_ids}

        branch = next(
            branch
            for branch in level_config.branches
            if branch.branch_id == selected_branch_id
        )
        thought_results = _deserialize_thought_results(state)
        response = await gateway.chat(
            level_messages(
                level=level,
                original_prompt=state["original_prompt"],
                previous_results=thought_results,
                branch=branch,
            ),
            meta=_call_meta(state, f"L{level}"),
        )
        intermediate_result = clean_intermediate(response.text)

        thought_results.append(
            ThoughtResult(
                level=level,
                branch_id=branch.branch_id,
                branch_name=branch.name,
                thought_prompt=branch.thought_prompt,
                intermediate_result=intermediate_result,
                selection_probability=selected_prob,
                probabilities=probabilities,
                request_hash=response.request_hash,
            )
        )

        selected_path = list(state.get("selected_path", []))
        selected_path.append(edge_key(level, branch.branch_id))

        return {
            "original_prompt": state["original_prompt"],
            "pheromones": pheromones,
            "alpha": alpha,
            "epsilon": epsilon,
            "use_aco": use_aco,
            "seed": int(state.get("seed", 0)),
            "task_id": str(state.get("task_id", "")),
            "iteration": int(state.get("iteration", 0)),
            "agent_id": int(state.get("agent_id", 0)),
            "thought_results": [item.__dict__ for item in thought_results],
            "selected_path": selected_path,
            "path_score": float(state.get("path_score", 1.0)) * float(selected_prob),
        }

    return node


def _make_finalize_node(gateway: LLMGateway):
    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        thought_results = _deserialize_thought_results(state)
        if not thought_results:
            raise ValueError("No thought results present at finalize node.")
        response = await gateway.chat(
            final_messages(
                original_prompt=state["original_prompt"],
                thought_results=thought_results,
            ),
            meta=_call_meta(state, "final"),
        )
        answer = canonicalize_final_answer(response.text)
        return {
            "original_prompt": state["original_prompt"],
            "pheromones": dict(state.get("pheromones", {})),
            "alpha": float(state.get("alpha", 1.0)),
            "epsilon": float(state.get("epsilon", 0.001)),
            "use_aco": bool(state.get("use_aco", True)),
            "seed": int(state.get("seed", 0)),
            "task_id": str(state.get("task_id", "")),
            "iteration": int(state.get("iteration", 0)),
            "agent_id": int(state.get("agent_id", 0)),
            "thought_results": [item.__dict__ for item in thought_results],
            "selected_path": list(state.get("selected_path", [])),
            "path_score": float(state.get("path_score", 1.0)),
            "final_answer": answer,
            "final_request_hash": response.request_hash,
        }

    return node


def build_agent_graph(gateway: LLMGateway):
    """Compile the 5-level ToT graph whose LLM calls go through ``gateway``."""
    builder = StateGraph(dict)
    for level_cfg in LEVEL_CONFIGS:
        node_name = f"level_{level_cfg.level}"
        builder.add_node(node_name, _make_level_node(level_cfg.level, gateway))

    builder.add_node("finalize", _make_finalize_node(gateway))

    builder.add_edge(START, "level_1")
    builder.add_edge("level_1", "level_2")
    builder.add_edge("level_2", "level_3")
    builder.add_edge("level_3", "level_4")
    builder.add_edge("level_4", "level_5")
    builder.add_edge("level_5", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
