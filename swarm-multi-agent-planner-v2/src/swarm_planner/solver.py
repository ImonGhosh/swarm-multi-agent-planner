"""Per-instance solve loop: n agents x T iterations of ToT, optionally ACO-guided."""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Dict, Iterable, List

from .aco import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_INITIAL_PHEROMONE,
    DEFAULT_RHO,
)
from .aco.pheromone import apply_pheromone_update, create_initial_pheromone_state
from .llm.gateway import LLMGateway
from .llm.resilience import is_retryable_openai_error
from .logging import EventLog
from .run_io import RunContext, load_dataset
from .selection_rules import (
    HEADLINE_RULE,
    SELECTION_RULES,
    TraceRec,
    prediction_of,
    rec_from_trace,
    select,
)
from .tot.context import coerce_original_prompt, extract_current_task_from_prompt5shot
from .tot.graph import build_agent_graph
from .tot.levels import LEVEL_CONFIGS, all_edge_keys
from .types import AgentTrace, RunResult, ThoughtResult
from .verifier.parser import canonicalize_final_answer
from .verifier.scorer import build_task_quality_scorer

DEFAULT_AGENTS_PER_TASK = 8
DEFAULT_ITERATIONS_PER_TASK = 6
DEFAULT_SEED = 0
DEFAULT_TASK_CONCURRENCY = 1
DEFAULT_MAX_CONCURRENT_REQUESTS = 8


def _branch_order() -> Dict[int, List[str]]:
    return {cfg.level: [b.branch_id for b in cfg.branches] for cfg in LEVEL_CONFIGS}


_coerce_original_prompt = coerce_original_prompt


def _coerce_scoring_task_text(task_prompt: str) -> str:
    """
    Build the text used by the reward scorer.

    For 5-shot prompts, score against the current task block only.
    For single-task prompts, use the normalized task text.
    """
    prompt = (task_prompt or "").strip()
    if prompt.count("TASK:") >= 2:
        return extract_current_task_from_prompt5shot(prompt)
    return _coerce_original_prompt(prompt)


def scorer_for_item(item: Dict[str, Any]):
    """The verifier for one dataset item (0-shot task text, with the 5-shot task block as fallback)."""
    task_prompt = item.get("prompt_5shot", item.get("prompt_0shot", ""))
    scoring_prompt = item.get("prompt_0shot") or _coerce_scoring_task_text(task_prompt)
    try:
        return build_task_quality_scorer(scoring_prompt)
    except ValueError:
        fallback = _coerce_scoring_task_text(task_prompt)
        if fallback == scoring_prompt:
            raise
        return build_task_quality_scorer(fallback)


def _resolve_max_parallel_agents(agents_per_task: int) -> int:
    """
    Cap concurrent in-flight agents of one task. All agents run at once by default; the
    gateway's global request cap (--max_concurrent_requests) is what limits provider load.

    This does not change the total search budget; it only paces execution.
    """
    raw = os.getenv("ACO_MAX_PARALLEL_AGENTS", "").strip()
    if not raw:
        return agents_per_task
    try:
        requested = int(raw)
    except ValueError:
        return agents_per_task
    return max(1, min(agents_per_task, requested))


def _resolve_agent_retry_attempts() -> int:
    raw = os.getenv("ACO_AGENT_MAX_ATTEMPTS", "").strip()
    if not raw:
        return 3
    try:
        value = int(raw)
    except ValueError:
        return 3
    return max(1, min(10, value))


def _resolve_agent_retry_delay(base: bool) -> float:
    name = "ACO_AGENT_RETRY_BASE_SECONDS" if base else "ACO_AGENT_RETRY_MAX_SECONDS"
    default = 3.0 if base else 60.0
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if base:
        return max(0.1, min(60.0, value))
    return max(_resolve_agent_retry_delay(base=True), min(300.0, value))


async def _run_single_agent(
    *,
    graph: Any,
    agent_id: int,
    task_id: str,
    iteration: int,
    original_prompt: str,
    pheromones: Dict[str, float],
    alpha: float,
    epsilon: float,
    seed: int,
    use_aco: bool,
) -> AgentTrace:
    initial_state = {
        "original_prompt": original_prompt,
        "pheromones": dict(pheromones),
        "alpha": alpha,
        "epsilon": epsilon,
        "use_aco": use_aco,
        "seed": seed,
        "task_id": task_id,
        "iteration": iteration,
        "agent_id": agent_id,
        "selected_path": [],
        "thought_results": [],
        "path_score": 1.0,
    }
    output = await graph.ainvoke(initial_state)
    raw_thought_results = list(output.get("thought_results", []))
    thought_results: List[ThoughtResult] = [
        item if isinstance(item, ThoughtResult) else ThoughtResult(**item)
        for item in raw_thought_results
    ]
    return AgentTrace(
        agent_id=agent_id,
        path=list(output.get("selected_path", [])),
        thought_results=thought_results,
        final_answer=canonicalize_final_answer(output.get("final_answer", "")),
        path_score=float(output.get("path_score", 0.0)),
        quality_score=0.0,
        final_request_hash=str(output.get("final_request_hash", "")),
    )


async def _run_multi_agent_for_task(
    *,
    graph: Any,
    task_id: str,
    iteration: int,
    original_prompt: str,
    pheromones: Dict[str, float],
    alpha: float,
    epsilon: float,
    agents_per_task: int,
    seed: int,
    use_aco: bool,
) -> List[AgentTrace]:
    max_parallel_agents = _resolve_max_parallel_agents(agents_per_task)
    semaphore = asyncio.Semaphore(max_parallel_agents)
    agent_attempts = _resolve_agent_retry_attempts()
    agent_base_delay = _resolve_agent_retry_delay(base=True)
    agent_max_delay = _resolve_agent_retry_delay(base=False)

    async def _run_bounded(agent_id: int) -> AgentTrace:
        async with semaphore:
            for attempt in range(agent_attempts):
                try:
                    return await _run_single_agent(
                        graph=graph,
                        agent_id=agent_id,
                        task_id=task_id,
                        iteration=iteration,
                        original_prompt=original_prompt,
                        pheromones=pheromones,
                        alpha=alpha,
                        epsilon=epsilon,
                        seed=seed,
                        use_aco=use_aco,
                    )
                except Exception as err:
                    if (not is_retryable_openai_error(err)) or (attempt + 1 >= agent_attempts):
                        raise
                    delay = min(agent_max_delay, agent_base_delay * float(2**attempt))
                    jitter = random.uniform(0.0, min(1.0, delay * 0.25))
                    await asyncio.sleep(delay + jitter)

            # Should never be reached due to return/raise in loop.
            raise RuntimeError("Agent execution failed unexpectedly without exception context.")

    coroutines = [_run_bounded(agent_id) for agent_id in range(agents_per_task)]
    traces = await asyncio.gather(*coroutines)
    return list(traces)


def _score_traces_with_task_scorer(traces: Iterable[AgentTrace], task_scorer) -> None:
    for trace in traces:
        result = task_scorer.score_prediction(trace.final_answer)
        trace.quality_score = float(result.score)
        trace.quality_details = result.to_dict()


def verifier_record(details: Dict[str, Any]) -> Dict[str, Any]:
    """The verifier fields that the event log keeps for every answer."""
    return {
        "parsed": details.get("candidate_parsed"),
        "valid": bool(details.get("hard_valid", False)),
        "hard": details.get("hard_checks_passed"),
        "soft": details.get("soft_satisfied"),
        "soft_total": details.get("soft_total"),
        "rank": details.get("slot_rank"),
    }


def trace_record(trace: AgentTrace) -> Dict[str, Any]:
    """Compact form of a trace for the event log (no prompt or response text)."""
    order = _branch_order()
    details = trace.quality_details or {}
    return {
        "agent": trace.agent_id,
        "path": trace.path,
        # Probability of every branch at each level, in branch order (see run_start).
        "p": [
            [round(item.probabilities.get(branch, 0.0), 6) for branch in order[item.level]]
            for item in trace.thought_results
        ],
        # Request hashes of the 5 level calls and the finalize call; text is in the LLM cache.
        "req": trace.request_hashes(),
        "ans": trace.final_answer,
        "ps": round(trace.path_score, 8),
        "q": trace.quality_score,
        "v": verifier_record(details),
    }


def _tau_vector(state) -> List[float]:
    return [round(float(state.pheromones.get(edge, 0.0)), 8) for edge in all_edge_keys()]




async def solve_task(
    *,
    task_id: str,
    task_prompt: str,
    input_meta: Dict[str, Any] | None = None,
    gateway: LLMGateway,
    agents_per_task: int = DEFAULT_AGENTS_PER_TASK,
    iterations_per_task: int = DEFAULT_ITERATIONS_PER_TASK,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    aco_enabled: bool = True,
    selection_policy: str = HEADLINE_RULE,
    scoring_task_text_override: str | None = None,
    event_log: EventLog | None = None,
) -> RunResult:
    """Solve one task: T iterations of n ToT agents with local (per-task) pheromones.

    The full budget is always used (no early stopping), so that the pheromone can act. The
    answer is chosen after the last iteration by ``selection_policy`` (see selection_rules.py);
    the event log keeps every trace, so any other rule can be derived offline.
    """
    if selection_policy not in SELECTION_RULES:
        raise ValueError(
            f"Unknown selection policy {selection_policy!r}; expected one of {SELECTION_RULES}"
        )
    original_prompt = _coerce_original_prompt(task_prompt)
    scoring_task_text = (
        scoring_task_text_override
        if scoring_task_text_override is not None
        else _coerce_scoring_task_text(task_prompt)
    )

    try:
        task_scorer = build_task_quality_scorer(scoring_task_text)
    except ValueError:
        fallback_scoring_text = _coerce_scoring_task_text(task_prompt)
        if fallback_scoring_text == scoring_task_text:
            raise
        scoring_task_text = fallback_scoring_text
        task_scorer = build_task_quality_scorer(scoring_task_text)

    graph = build_agent_graph(gateway)
    iterations = max(1, int(iterations_per_task))
    local_state = create_initial_pheromone_state(
        alpha=alpha,
        epsilon=epsilon,
        rho=rho,
        seed=seed,
        initial_tau=DEFAULT_INITIAL_PHEROMONE,
    )

    pheromone_updates_applied = 0
    final_iteration_traces: List[AgentTrace] = []
    per_iteration: List[Dict[str, Any]] = []
    recs: List[TraceRec] = []

    for iteration_idx in range(iterations):
        tau_before = _tau_vector(local_state) if aco_enabled else None
        traces = await _run_multi_agent_for_task(
            graph=graph,
            task_id=task_id,
            iteration=iteration_idx,
            original_prompt=original_prompt,
            pheromones=local_state.pheromones if aco_enabled else {},
            alpha=alpha,
            epsilon=epsilon,
            agents_per_task=agents_per_task,
            seed=seed,
            use_aco=aco_enabled,
        )
        _score_traces_with_task_scorer(traces, task_scorer)
        iteration_recs = [rec_from_trace(iteration_idx, trace) for trace in traces]
        recs.extend(iteration_recs)
        best_rec = select(iteration_recs, "best_of_all")  # the best trace of this iteration
        per_iteration.append(
            {
                "iteration": iteration_idx + 1,
                "best_agent_id": best_rec.agent if best_rec else None,
                "best_quality": best_rec.q if best_rec else 0.0,
                "best_prediction": best_rec.answer if best_rec else "",
                "best_slot_rank": best_rec.rank if best_rec else None,
                "verified_traces": sum(1 for r in iteration_recs if r.verified),
            }
        )
        final_iteration_traces = traces
        if aco_enabled:
            local_state = apply_pheromone_update(
                pheromone_state=local_state,
                traces=traces,
                rho=rho,
            )
            pheromone_updates_applied += 1
        if event_log is not None:
            event_log.log(
                "iteration",
                {
                    "task_id": task_id,
                    "it": iteration_idx,
                    "tau_before": tau_before,
                    "tau_after": _tau_vector(local_state) if aco_enabled else None,
                    "best_agent": best_rec.agent if best_rec else None,
                    "traces": [trace_record(trace) for trace in traces],
                },
            )

    chosen = select(recs, selection_policy)
    prediction = prediction_of(chosen)

    metadata = {
        "mode": "solve",
        "iterations_per_task": iterations,
        "aco_enabled": aco_enabled,
        "scoring_mode": "deterministic_task_based",
        "selection_policy": selection_policy,
        "pheromone_updates_applied": pheromone_updates_applied,
        "selected_iteration": (chosen.it + 1) if chosen is not None else None,
        "selected_agent_id": chosen.agent if chosen is not None else None,
        "selected_quality": chosen.q if chosen is not None else 0.0,
        "selected_slot_rank": chosen.rank if chosen is not None else None,
        "selected_verified": bool(chosen.verified) if chosen is not None else False,
        "verified_traces": sum(1 for r in recs if r.verified),
        "per_iteration": per_iteration,
    }

    return RunResult(
        prediction=prediction,
        traces=final_iteration_traces,
        pheromone_state=local_state if aco_enabled else None,
        metadata=metadata,
    )


async def _solve_dataset_async(
    *,
    data_path: str,
    run_dir: str,
    model: str,
    agents_per_task: int,
    iterations_per_task: int,
    alpha: float,
    epsilon: float,
    rho: float,
    seed: int,
    aco_enabled: bool = True,
    selection_policy: str = HEADLINE_RULE,
    limit: int | None = None,
    task_concurrency: int = DEFAULT_TASK_CONCURRENCY,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_cost_usd: float | None = None,
    cache_path: str | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    data = load_dataset(data_path, limit)
    task_count = len(data)
    ctx = RunContext(
        run_dir=run_dir,
        model=model,
        cache_path=cache_path,
        max_concurrent_requests=max_concurrent_requests,
        max_cost_usd=max_cost_usd,
        client=client,
    )
    event_log, gateway = ctx.event_log, ctx.gateway

    config = {
        "mode": "solve",
        "config_name": "tot_aco" if aco_enabled else "tot_only",
        "limit": limit,
        "tasks": task_count,
        "model": model,
        "agents_per_task": agents_per_task,
        "iterations_per_task": iterations_per_task,
        "alpha": alpha,
        "epsilon": epsilon,
        "rho": rho,
        "deposit": "top_only",
        "early_stopping": False,
        "seed": seed,
        "aco_enabled": aco_enabled,
        "selection_policy": selection_policy,
        "scoring_mode": "deterministic_task_based",
        "generation_prompt": "prompt_5shot",
        "scoring_prompt": "prompt_0shot_with_fallback",
        "cache_path": cache_path,
        "task_concurrency": task_concurrency,
        "max_concurrent_requests": max_concurrent_requests,
        "max_parallel_agents": _resolve_max_parallel_agents(agents_per_task),
    }
    try:
        ctx.start(config, data_path, {"branch_order": _branch_order(), "edge_order": all_edge_keys()})
        print(
            f"Starting solve: tasks={task_count}, agents_per_task={agents_per_task}, "
            f"iterations_per_task={iterations_per_task}, aco_enabled={aco_enabled}, "
            f"selection={selection_policy}, seed={seed}"
        )

        totals = {"tasks": 0, "quality": 0.0, "updates": 0, "verified": 0}
        task_semaphore = asyncio.Semaphore(max(1, int(task_concurrency)))

        async def _run_one(task_offset: int, task_id: str, item: Dict[str, Any]) -> None:
            async with task_semaphore:
                task_started = time.perf_counter()
                task_prompt = item.get("prompt_5shot", item.get("prompt_0shot", ""))
                scoring_prompt = item.get("prompt_0shot") or _coerce_scoring_task_text(task_prompt)
                try:
                    run_result = await solve_task(
                        task_id=task_id,
                        task_prompt=task_prompt,
                        input_meta=item,
                        gateway=gateway,
                        agents_per_task=agents_per_task,
                        iterations_per_task=iterations_per_task,
                        alpha=alpha,
                        epsilon=epsilon,
                        rho=rho,
                        seed=seed,
                        aco_enabled=aco_enabled,
                        selection_policy=selection_policy,
                        scoring_task_text_override=scoring_prompt,
                        event_log=event_log,
                    )
                except BaseException as err:
                    event_log.log(
                        "task_error",
                        {"task_id": task_id, "error": f"{type(err).__name__}: {err}"},
                        flush=True,
                    )
                    raise

                metadata = run_result.metadata or {}
                quality = float(metadata.get("selected_quality", 0.0) or 0.0)
                updates = int(metadata.get("pheromone_updates_applied", 0) or 0)
                totals["tasks"] += 1
                totals["quality"] += quality
                totals["updates"] += updates
                totals["verified"] += int(bool(metadata.get("selected_verified")))

                item["pred_5shot_pro"] = run_result.prediction
                item["pred_model"] = model
                item["aco_tot_agent_count"] = agents_per_task
                item["aco_tot_iterations_per_task"] = int(iterations_per_task)
                item["aco_tot_mode"] = "solve"
                item["aco_tot_scoring_mode"] = "deterministic_task_based"
                item["aco_tot_selection_policy"] = selection_policy
                item["aco_tot_aco_enabled"] = bool(aco_enabled)
                item["aco_tot_selected_quality"] = quality
                item["aco_tot_selected_slot_rank"] = metadata.get("selected_slot_rank")
                item["aco_tot_verified_traces"] = metadata.get("verified_traces")
                item["aco_tot_pheromone_updates_applied"] = updates

                event_log.log(
                    "task_result",
                    {
                        "task_id": task_id,
                        "position": task_offset + 1,
                        "prediction": run_result.prediction,
                        "metadata": metadata,
                        "llm": gateway.stats_for(task_id).to_dict(),
                        "wall_s": round(time.perf_counter() - task_started, 3),
                    },
                    flush=True,
                )
                ctx.progress(
                    "solve",
                    totals["tasks"],
                    task_count,
                    f"task_id={task_id} quality={quality:.3f} "
                    f"verified_traces={metadata.get('verified_traces')}",
                )

        tasks = [
            asyncio.ensure_future(_run_one(offset, task_id, item))
            for offset, (task_id, item) in enumerate(data.items())
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        tasks_processed = totals["tasks"]
        summary = {
            "mode": "solve",
            "config_name": config["config_name"],
            "tasks": tasks_processed,
            "agents_per_task": agents_per_task,
            "iterations_per_task": int(iterations_per_task),
            "aco_enabled": aco_enabled,
            "selection_policy": selection_policy,
            "scoring_mode": "deterministic_task_based",
            "seed": seed,
            "pheromone_updates_applied": totals["updates"],
            "selected_verified_share": totals["verified"] / tasks_processed if tasks_processed else 0.0,
            "final_mean_quality": totals["quality"] / tasks_processed if tasks_processed else 0.0,
        }
        summary = ctx.finish(data, summary)
    finally:
        ctx.close()

    print(f"Solve complete: tasks={tasks_processed}, mean_quality={summary['final_mean_quality']:.4f}")
    return summary


def solve_dataset(
    data_path: str,
    run_dir: str,
    model: str,
    agents_per_task: int = DEFAULT_AGENTS_PER_TASK,
    iterations_per_task: int = DEFAULT_ITERATIONS_PER_TASK,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    aco_enabled: bool = True,
    selection_policy: str = HEADLINE_RULE,
    limit: int | None = None,
    task_concurrency: int = DEFAULT_TASK_CONCURRENCY,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_cost_usd: float | None = None,
    cache_path: str | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    """Solve a dataset and write predictions.json, events.jsonl.gz and summary.json to run_dir."""
    return asyncio.run(
        _solve_dataset_async(
            data_path=data_path,
            run_dir=run_dir,
            model=model,
            agents_per_task=agents_per_task,
            iterations_per_task=iterations_per_task,
            alpha=alpha,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
            aco_enabled=aco_enabled,
            selection_policy=selection_policy,
            limit=limit,
            task_concurrency=task_concurrency,
            max_concurrent_requests=max_concurrent_requests,
            max_cost_usd=max_cost_usd,
            cache_path=cache_path,
            client=client,
        )
    )
