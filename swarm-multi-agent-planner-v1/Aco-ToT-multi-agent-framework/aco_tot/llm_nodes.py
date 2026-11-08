"""LangGraph node builders for ACO-ToT traversal."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import NotFoundError

from .aco import select_branch
from .config import LEAF_BEHAVIOR_PROMPTS, LEVEL_CONFIGS, edge_key, get_level_config
from .context_accumulator import build_prompt_for_final, build_prompt_for_level
from .openrouter_resilience import (
    OPENROUTER_BASE_URL,
    ainvoke_with_backoff,
    is_openrouter_base_url,
    provider_preferences_from_env,
)
from .prompt_io import canonicalize_final_answer
from .types import TaskFeatures, ThoughtResult


_LLM_CACHE: Dict[str, ChatOpenAI] = {}
_GRAPH_CACHE: Dict[str, Any] = {}
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 400
LLM_TOP_P = 1.0


def _raise_model_unavailable(model: str, err: NotFoundError) -> None:
    raise RuntimeError(
        "Configured endpoint could not serve model "
        f"'{model}' (404/not found). "
        "Use a valid model ID for your selected backend and retry."
    ) from err


def _get_llm(model: str) -> ChatOpenAI:
    cached = _LLM_CACHE.get(model)
    if cached is not None:
        return cached
    base_url = os.getenv("LLM_BASE_URL", OPENROUTER_BASE_URL).strip()
    use_openrouter = is_openrouter_base_url(base_url)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key and not use_openrouter:
        # Ollama and other local OpenAI-compatible servers often ignore the key,
        # but the client still expects a non-empty value.
        api_key = "local"
    if not api_key and use_openrouter:
        raise EnvironmentError(
            "Set OPENROUTER_API_KEY (or OPENAI_API_KEY) before running ACO-ToT."
        )
    client_kwargs: Dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "top_p": LLM_TOP_P,
    }
    if use_openrouter:
        client_kwargs["extra_body"] = {"provider": provider_preferences_from_env()}
    llm = ChatOpenAI(model=model, **client_kwargs)
    _LLM_CACHE[model] = llm
    return llm


def _response_text(response: Any) -> str:
    if response is None:
        return ""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks).strip()
    return str(content)


def _feature_from_state(state: Dict[str, Any]) -> TaskFeatures:
    features = state.get("task_features")
    if isinstance(features, TaskFeatures):
        return features
    if isinstance(features, dict):
        return TaskFeatures(**features)
    raise ValueError("task_features missing from graph state.")


def _deserialize_thought_results(state: Dict[str, Any]) -> List[ThoughtResult]:
    raw = list(state.get("thought_results", []))
    results: List[ThoughtResult] = []
    for item in raw:
        if isinstance(item, ThoughtResult):
            results.append(item)
        elif isinstance(item, dict):
            results.append(ThoughtResult(**item))
    return results


def _make_level_node(level: int, model: str):
    level_config = get_level_config(level)

    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        features = _feature_from_state(state)
        pheromones = dict(state.get("pheromones", {}))
        alpha = float(state.get("alpha", 1.0))
        beta = float(state.get("beta", 2.0))
        epsilon = float(state.get("epsilon", 0.001))
        agent_seed = int(state.get("agent_seed", 0))
        use_aco = bool(state.get("use_aco", True))

        branch_ids = [branch.branch_id for branch in level_config.branches]
        rng = random.Random(agent_seed + (level * 7919))
        if use_aco:
            selected_branch_id, selected_prob, _ = select_branch(
                level=level,
                branch_ids=branch_ids,
                task_features=features,
                pheromones=pheromones,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                rng=rng,
            )
        else:
            selected_branch_id = rng.choice(branch_ids)
            selected_prob = 1.0 / float(len(branch_ids))

        branch = next(
            branch
            for branch in level_config.branches
            if branch.branch_id == selected_branch_id
        )
        thought_results = _deserialize_thought_results(state)
        prompt = build_prompt_for_level(
            original_prompt=state["original_prompt"],
            previous_results=thought_results,
            current_thought_prompt=branch.thought_prompt,
        )

        system_prompt = (
            "You are solving a calendar scheduling task using a fixed Tree-of-Thought path. "
            f"Current level {level}: {level_config.name}. "
            f"Level goal: {level_config.goal}. "
            "Return only the intermediate result for the thought provided in the user input. "
            "Do not provide final solution at this step. "
        )
        user_prompt = (
            f"{prompt}\n\n"
            "Provide a concise intermediate result for the latest thought only, for the last unsolved task."
        )
        llm = _get_llm(model)
        response = await ainvoke_with_backoff(
            llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            model_label=model,
            on_not_found=_raise_model_unavailable,
        )
        intermediate_result = _response_text(response).strip()

        if not intermediate_result:
            intermediate_result = (
                "No intermediate result generated; carry forward parsed constraints only."
            )

        thought_results.append(
            ThoughtResult(
                level=level,
                branch_id=branch.branch_id,
                branch_name=branch.name,
                thought_prompt=branch.thought_prompt,
                intermediate_result=intermediate_result,
                selection_probability=selected_prob,
                accumulated_prompt=prompt,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )

        selected_path = list(state.get("selected_path", []))
        selected_path.append(edge_key(level, branch.branch_id))

        return {
            "original_prompt": state["original_prompt"],
            "task_features": state["task_features"],
            "pheromones": pheromones,
            "alpha": alpha,
            "beta": beta,
            "epsilon": epsilon,
            "use_aco": use_aco,
            "agent_seed": agent_seed,
            "thought_results": [item.__dict__ for item in thought_results],
            "selected_path": selected_path,
            "path_score": float(state.get("path_score", 1.0)) * float(selected_prob),
        }

    return node


def _make_finalize_node(model: str):
    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        thought_results = _deserialize_thought_results(state)
        if not thought_results:
            raise ValueError("No thought results present at finalize node.")
        leaf_branch_id = thought_results[-1].branch_id
        leaf_instruction = LEAF_BEHAVIOR_PROMPTS.get(
            leaf_branch_id,
            "Verify once and return a final answer.",
        )

        prompt = build_prompt_for_final(
            original_prompt=state["original_prompt"],
            thought_results=thought_results,
            final_instruction=leaf_instruction,
        )
        system_prompt = (
            "You are a calendar scheduling solver. "
            "Produce one final answer using format exactly: "
            "Here is the proposed time: <Day>, <HH:MM> - <HH:MM>. "
            "Return only one line."
        )
        user_prompt = f"{prompt}\n\nReturn the final proposed slot now."
        llm = _get_llm(model)
        response = await ainvoke_with_backoff(
            llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            model_label=model,
            on_not_found=_raise_model_unavailable,
        )
        answer = canonicalize_final_answer(_response_text(response))
        return {
            "original_prompt": state["original_prompt"],
            "task_features": state["task_features"],
            "pheromones": dict(state.get("pheromones", {})),
            "alpha": float(state.get("alpha", 1.0)),
            "beta": float(state.get("beta", 2.0)),
            "epsilon": float(state.get("epsilon", 0.001)),
            "use_aco": bool(state.get("use_aco", True)),
            "agent_seed": int(state.get("agent_seed", 0)),
            "thought_results": [item.__dict__ for item in thought_results],
            "selected_path": list(state.get("selected_path", [])),
            "path_score": float(state.get("path_score", 1.0)),
            "final_answer": answer,
            "final_accumulated_prompt": prompt,
            "final_system_prompt": system_prompt,
            "final_user_prompt": user_prompt,
        }

    return node


def build_agent_graph(model: str):
    cached = _GRAPH_CACHE.get(model)
    if cached is not None:
        return cached

    builder = StateGraph(dict)
    for level_cfg in LEVEL_CONFIGS:
        node_name = f"level_{level_cfg.level}"
        builder.add_node(node_name, _make_level_node(level_cfg.level, model))

    builder.add_node("finalize", _make_finalize_node(model))

    builder.add_edge(START, "level_1")
    builder.add_edge("level_1", "level_2")
    builder.add_edge("level_2", "level_3")
    builder.add_edge("level_3", "level_4")
    builder.add_edge("level_4", "level_5")
    builder.add_edge("level_5", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile()
    _GRAPH_CACHE[model] = graph
    return graph
