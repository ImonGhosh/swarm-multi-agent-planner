"""Training and inference orchestration for ACO-ToT calendar scheduling."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import (
    DEFAULT_AGENTS_PER_TASK,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_EPSILON,
    DEFAULT_INITIAL_PHEROMONE,
    DEFAULT_RHO,
    DEFAULT_SEED,
)
from .llm_nodes import build_agent_graph
from .openrouter_resilience import is_retryable_openai_error
from .pheromone import apply_pheromone_update
from .prompt_io import (
    build_root_prompt_from_prompt5shot,
    build_task_features,
    canonicalize_final_answer,
    extract_current_task_from_prompt5shot,
    remove_last_final_instruction,
)
from .quality import QualityScoreFn, resolve_quality_score_fn
from .run_logging import JsonlRunLogger, make_run_log_path
from .store import (
    create_initial_pheromone_state,
    load_pheromone_state,
    save_pheromone_state,
)
from .task_quality import build_task_quality_scorer
from .types import AgentTrace, PheromoneState, RunResult, TaskFeatures, ThoughtResult


def _coerce_original_prompt(task_prompt: str) -> str:
    prompt = (task_prompt or "").strip()
    if prompt.count("TASK:") >= 2:
        return build_root_prompt_from_prompt5shot(prompt)
    prompt = prompt.rstrip()
    prompt = prompt.removesuffix("SOLUTION:").rstrip()
    return remove_last_final_instruction(prompt)


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


def _resolve_training_metrics_path(
    out_pheromone_path: str,
    metrics_out_path: str | None,
) -> Path:
    if metrics_out_path:
        return Path(metrics_out_path)
    pheromone_path = Path(out_pheromone_path)
    return pheromone_path.parent / "metrics" / "train_iteration_metrics.csv"


def _write_training_metrics_csv(
    metrics_rows: List[Dict[str, Any]],
    metrics_out_path: str | None,
    out_pheromone_path: str,
) -> str:
    metrics_path = _resolve_training_metrics_path(
        out_pheromone_path=out_pheromone_path,
        metrics_out_path=metrics_out_path,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "iteration",
        "epoch",
        "epoch_iteration",
        "task_id",
        "best_agent_quality",
        "mean_agent_quality",
        "best_agent_path_score",
        "correct_flag",
        "cumulative_correct",
        "cumulative_accuracy",
        "cumulative_mean_quality",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)
    return str(metrics_path)


def _resolve_max_parallel_agents(agents_per_task: int) -> int:
    """
    Cap concurrent in-flight agents to reduce provider burst throttling (429s).

    This does not change the total search budget; it only paces execution.
    """
    default_cap = min(agents_per_task, 2)
    raw = os.getenv("ACO_MAX_PARALLEL_AGENTS", "").strip()
    if not raw:
        return default_cap
    try:
        requested = int(raw)
    except ValueError:
        return default_cap
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
    agent_id: int,
    model: str,
    original_prompt: str,
    task_features: TaskFeatures,
    pheromones: Dict[str, float],
    alpha: float,
    beta: float,
    epsilon: float,
    seed: int,
    use_aco: bool,
) -> AgentTrace:
    graph = build_agent_graph(model)
    initial_state = {
        "original_prompt": original_prompt,
        "task_features": task_features.__dict__,
        "pheromones": dict(pheromones),
        "alpha": alpha,
        "beta": beta,
        "epsilon": epsilon,
        "use_aco": use_aco,
        "agent_seed": seed + (agent_id * 100003),
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
        final_accumulated_prompt=str(output.get("final_accumulated_prompt", "")),
        final_system_prompt=str(output.get("final_system_prompt", "")),
        final_user_prompt=str(output.get("final_user_prompt", "")),
    )


async def _run_multi_agent_for_task(
    *,
    model: str,
    original_prompt: str,
    task_features: TaskFeatures,
    pheromones: Dict[str, float],
    alpha: float,
    beta: float,
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
                        agent_id=agent_id,
                        model=model,
                        original_prompt=original_prompt,
                        task_features=task_features,
                        pheromones=pheromones,
                        alpha=alpha,
                        beta=beta,
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

def _solve_trace_sort_key(
    trace: AgentTrace,
) -> tuple[float, float, float, float, float, float]:
    details = trace.quality_details if isinstance(trace.quality_details, dict) else {}
    hard_valid = 1.0 if bool(details.get("hard_valid", False)) else 0.0
    soft_ratio = float(details.get("soft_satisfaction_ratio", 0.0) or 0.0)
    slot_rank_raw = details.get("slot_rank")
    slot_rank = int(slot_rank_raw) if isinstance(slot_rank_raw, int) else 1_000_000
    return (
        float(trace.quality_score),
        hard_valid,
        soft_ratio,
        float(-slot_rank),
        float(trace.path_score),
        float(-trace.agent_id),
    )


def _pick_solve_best_trace(traces: Iterable[AgentTrace]) -> AgentTrace | None:
    traces = list(traces)
    if not traces:
        return None
    return max(traces, key=_solve_trace_sort_key)


def _pick_prediction(traces: Iterable[AgentTrace], mode: str) -> str:
    traces = list(traces)
    if not traces:
        return "Here is the proposed time: Monday, 9:00 - 9:30"
    if mode == "train":
        best = max(traces, key=lambda trace: (trace.quality_score, trace.path_score))
        return best.final_answer
    if mode == "solve":
        best = _pick_solve_best_trace(traces)
        return (
            best.final_answer
            if best is not None
            else "Here is the proposed time: Monday, 9:00 - 9:30"
        )
    best = max(traces, key=lambda trace: trace.path_score)
    return best.final_answer


def _score_traces_with_task_scorer(traces: Iterable[AgentTrace], task_scorer) -> None:
    for trace in traces:
        result = task_scorer.score_prediction(trace.final_answer)
        trace.quality_score = float(result.score)
        trace.quality_details = result.to_dict()


async def process_task(
    *,
    task_id: str,
    task_prompt: str,
    input_meta: Dict[str, Any] | None,
    model: str,
    pheromone_state: PheromoneState,
    mode: str,
    agents_per_task: int,
    alpha: float,
    beta: float,
    epsilon: float,
    rho: float,
    seed: int,
    aco_enabled: bool,
    golden_plan: str | None,
    quality_score_fn: QualityScoreFn | None = None,
    iterations_per_task: int = 1,
    scoring_task_text_override: str | None = None,
) -> RunResult:
    normalized_mode = (mode or "infer").strip().lower()
    if normalized_mode not in {"train", "infer", "solve"}:
        raise ValueError("mode must be one of: train, infer, solve")

    original_prompt = _coerce_original_prompt(task_prompt)
    scoring_task_text = (
        scoring_task_text_override
        if scoring_task_text_override is not None
        else _coerce_scoring_task_text(task_prompt)
    )
    task_features = build_task_features(task_id, original_prompt, input_meta)

    if normalized_mode in {"train", "infer"}:
        traces = await _run_multi_agent_for_task(
            model=model,
            original_prompt=original_prompt,
            task_features=task_features,
            pheromones=pheromone_state.pheromones if aco_enabled else {},
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            agents_per_task=agents_per_task,
            seed=seed,
            use_aco=aco_enabled,
        )

        updated_state = pheromone_state
        if normalized_mode == "train":
            score_fn = resolve_quality_score_fn(quality_score_fn)
            # Deduplicate scorer calls by prediction text to reduce training cost.
            unique_predictions = list({trace.final_answer for trace in traces})
            scores = await asyncio.gather(
                *[
                    score_fn(prediction, golden_plan, scoring_task_text)
                    for prediction in unique_predictions
                ]
            )
            prediction_to_score = {
                prediction: float(score)
                for prediction, score in zip(unique_predictions, scores, strict=False)
            }
            for trace in traces:
                trace.quality_score = prediction_to_score.get(trace.final_answer, 0.0)
            if aco_enabled:
                updated_state = apply_pheromone_update(
                    pheromone_state=pheromone_state,
                    traces=traces,
                    rho=rho,
                )

        prediction = _pick_prediction(traces, mode=normalized_mode)
        return RunResult(
            prediction=prediction,
            traces=traces,
            pheromone_state=updated_state,
            metadata={
                "mode": normalized_mode,
                "selection_policy": (
                    "quality_then_path"
                    if normalized_mode == "train"
                    else "path_only"
                ),
                "aco_enabled": aco_enabled,
            },
        )

    # solve mode: per-task local ACO with deterministic scoring.
    try:
        task_scorer = build_task_quality_scorer(scoring_task_text)
    except ValueError:
        fallback_scoring_text = _coerce_scoring_task_text(task_prompt)
        if fallback_scoring_text == scoring_task_text:
            raise
        scoring_task_text = fallback_scoring_text
        task_scorer = build_task_quality_scorer(scoring_task_text)

    iterations = max(1, int(iterations_per_task))
    local_state = create_initial_pheromone_state(
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
        seed=seed,
        initial_tau=DEFAULT_INITIAL_PHEROMONE,
    )

    pheromone_updates_applied = 0
    final_iteration_traces: List[AgentTrace] = []
    per_iteration: List[Dict[str, Any]] = []

    for iteration_idx in range(iterations):
        traces = await _run_multi_agent_for_task(
            model=model,
            original_prompt=original_prompt,
            task_features=task_features,
            pheromones=local_state.pheromones if aco_enabled else {},
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            agents_per_task=agents_per_task,
            seed=seed + (iteration_idx * 10_000_000),
            use_aco=aco_enabled,
        )
        _score_traces_with_task_scorer(traces, task_scorer)
        best_trace = _pick_solve_best_trace(traces)
        per_iteration.append(
            {
                "iteration": iteration_idx + 1,
                "best_agent_id": best_trace.agent_id if best_trace else None,
                "best_quality": best_trace.quality_score if best_trace else 0.0,
                "best_prediction": best_trace.final_answer if best_trace else "",
                "best_slot_rank": (
                    (best_trace.quality_details or {}).get("slot_rank")
                    if best_trace is not None
                    else None
                ),
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

    final_best_trace = _pick_solve_best_trace(final_iteration_traces)
    prediction = (
        final_best_trace.final_answer
        if final_best_trace is not None
        else "Here is the proposed time: Monday, 9:00 - 9:30"
    )

    metadata = {
        "mode": "solve",
        "iterations_per_task": iterations,
        "aco_enabled": aco_enabled,
        "scoring_mode": "deterministic_task_based",
        "selection_policy": "final_iteration_best",
        "pheromone_updates_applied": pheromone_updates_applied,
        "final_iteration_best_agent_id": (
            final_best_trace.agent_id if final_best_trace is not None else None
        ),
        "final_iteration_best_quality": (
            final_best_trace.quality_score if final_best_trace is not None else 0.0
        ),
        "final_iteration_best_slot_rank": (
            (final_best_trace.quality_details or {}).get("slot_rank")
            if final_best_trace is not None
            else None
        ),
        "per_iteration": per_iteration,
    }

    return RunResult(
        prediction=prediction,
        traces=final_iteration_traces,
        pheromone_state=local_state if aco_enabled else None,
        metadata=metadata,
    )


async def _train_dataset_async(
    *,
    data_path: str,
    out_pheromone_path: str,
    model: str,
    agents_per_task: int,
    epochs: int,
    alpha: float,
    beta: float,
    epsilon: float,
    rho: float,
    seed: int,
    aco_enabled: bool = True,
    metrics_out_path: str | None = None,
    log_out_path: str | None = None,
) -> Dict[str, Any]:
    with Path(data_path).open("r", encoding="utf-8") as handle:
        data: Dict[str, Dict[str, Any]] = json.load(handle)
    task_count = len(data)

    pheromone_state = create_initial_pheromone_state(
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
        seed=seed,
        initial_tau=DEFAULT_INITIAL_PHEROMONE,
    )
    solved_proxy = 0.0
    total_tasks = 0
    cumulative_correct = 0
    cumulative_quality_sum = 0.0
    pheromone_updates_applied = 0
    metrics_rows: List[Dict[str, Any]] = []
    run_log_path = make_run_log_path(
        mode="train",
        base_dir=Path(out_pheromone_path).parent,
        log_out_path=log_out_path,
    )
    logger = JsonlRunLogger(run_log_path)
    logger.log_event(
        "run_start",
        {
            "mode": "train",
            "data_path": data_path,
            "out_pheromone_path": out_pheromone_path,
            "model": model,
            "agents_per_task": agents_per_task,
            "epochs": epochs,
            "alpha": alpha,
            "beta": beta,
            "epsilon": epsilon,
            "rho": rho,
            "seed": seed,
            "aco_enabled": aco_enabled,
        },
    )
    print(
        f"Starting training: tasks={task_count}, epochs={epochs}, "
        f"agents_per_task={agents_per_task}, aco_enabled={aco_enabled}"
    )

    try:
        for epoch_idx in range(epochs):
            print(f"Epoch {epoch_idx + 1}/{epochs} started")
            for task_offset, (task_id, item) in enumerate(data.items()):
                total_tasks += 1
                task_prompt = item.get("prompt_5shot", item.get("prompt_0shot", ""))
                run_result = await process_task(
                    task_id=task_id,
                    task_prompt=task_prompt,
                    input_meta=item,
                    model=model,
                    pheromone_state=pheromone_state,
                    mode="train",
                    agents_per_task=agents_per_task,
                    alpha=alpha,
                    beta=beta,
                    epsilon=epsilon,
                    rho=rho,
                    seed=seed + (epoch_idx * 1_000_000) + task_offset,
                    aco_enabled=aco_enabled,
                    golden_plan=item.get("golden_plan"),
                )
                pheromone_state = run_result.pheromone_state or pheromone_state
                if aco_enabled:
                    pheromone_updates_applied += 1
                best_agent_quality = max(
                    (trace.quality_score for trace in run_result.traces), default=0.0
                )
                mean_agent_quality = (
                    sum(trace.quality_score for trace in run_result.traces)
                    / float(len(run_result.traces))
                    if run_result.traces
                    else 0.0
                )
                best_agent_path_score = max(
                    (trace.path_score for trace in run_result.traces), default=0.0
                )
                solved_proxy += best_agent_quality
                # Exact-accuracy only: mark correct only for exact-match quality (1.00).
                correct_flag = 1 if best_agent_quality >= 0.999999 else 0
                cumulative_correct += correct_flag
                cumulative_quality_sum += best_agent_quality
                row = {
                    "iteration": total_tasks,
                    "epoch": epoch_idx + 1,
                    "epoch_iteration": task_offset + 1,
                    "task_id": task_id,
                    "best_agent_quality": best_agent_quality,
                    "mean_agent_quality": mean_agent_quality,
                    "best_agent_path_score": best_agent_path_score,
                    "correct_flag": correct_flag,
                    "cumulative_correct": cumulative_correct,
                    "cumulative_accuracy": (
                        cumulative_correct / float(total_tasks)
                        if total_tasks
                        else 0.0
                    ),
                    "cumulative_mean_quality": (
                        cumulative_quality_sum / float(total_tasks)
                        if total_tasks
                        else 0.0
                    ),
                }
                metrics_rows.append(row)
                logger.log_event(
                    "task_result",
                    {
                        "mode": "train",
                        "iteration": total_tasks,
                        "epoch": epoch_idx + 1,
                        "epoch_iteration": task_offset + 1,
                        "task_id": task_id,
                        "original_task_prompt": task_prompt,
                        "golden_plan": item.get("golden_plan"),
                        "prediction": run_result.prediction,
                        "aco_enabled": aco_enabled,
                        **row,
                        "traces": [trace.to_dict() for trace in run_result.traces],
                    },
                )
                print(
                    f"[train] epoch {epoch_idx + 1}/{epochs} "
                    f"task {task_offset + 1}/{task_count} "
                    f"(global {total_tasks}) task_id={task_id} "
                    f"best_quality={best_agent_quality:.4f} "
                    f"best_path_score={best_agent_path_score:.6f}"
                )
        logger.log_event(
            "run_end",
            {
                "mode": "train",
                "tasks_processed": total_tasks,
                "cumulative_correct": cumulative_correct,
                "final_cumulative_accuracy": (
                    cumulative_correct / float(total_tasks) if total_tasks else 0.0
                ),
                "final_cumulative_mean_quality": (
                    cumulative_quality_sum / float(total_tasks)
                    if total_tasks
                    else 0.0
                ),
                "aco_enabled": aco_enabled,
                "pheromone_updates_applied": pheromone_updates_applied,
            },
        )
    finally:
        logger.close()

    save_pheromone_state(out_pheromone_path, pheromone_state)
    metrics_path = _write_training_metrics_csv(
        metrics_rows=metrics_rows,
        metrics_out_path=metrics_out_path,
        out_pheromone_path=out_pheromone_path,
    )
    print(
        f"Training complete: tasks={total_tasks}, "
        f"final_accuracy={(cumulative_correct / float(total_tasks) if total_tasks else 0.0):.4f}, "
        f"final_mean_quality={(cumulative_quality_sum / float(total_tasks) if total_tasks else 0.0):.4f}"
    )
    print(f"Pheromones written to: {out_pheromone_path}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Run log written to: {run_log_path}")

    return {
        "mode": "train",
        "tasks": total_tasks,
        "epochs": epochs,
        "agents_per_task": agents_per_task,
        "aco_enabled": aco_enabled,
        "pheromone_updates_applied": pheromone_updates_applied,
        "quality_proxy_mean": (solved_proxy / float(total_tasks)) if total_tasks else 0.0,
        "pheromone_path": out_pheromone_path,
        "iteration_metrics_path": metrics_path,
        "final_cumulative_accuracy": (
            cumulative_correct / float(total_tasks) if total_tasks else 0.0
        ),
        "final_cumulative_mean_quality": (
            cumulative_quality_sum / float(total_tasks) if total_tasks else 0.0
        ),
        "run_log_path": str(run_log_path),
    }


async def _infer_dataset_async(
    *,
    data_path: str,
    pheromone_path: str,
    out_path: str,
    model: str,
    agents_per_task: int,
    alpha: float,
    beta: float,
    epsilon: float,
    rho: float,
    seed: int,
    aco_enabled: bool = True,
    log_out_path: str | None = None,
) -> None:
    with Path(data_path).open("r", encoding="utf-8") as handle:
        data: Dict[str, Dict[str, Any]] = json.load(handle)
    task_count = len(data)

    pheromone_state = (
        load_pheromone_state(
            pheromone_path,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
        )
        if aco_enabled
        else create_initial_pheromone_state(
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
            initial_tau=DEFAULT_INITIAL_PHEROMONE,
        )
    )
    run_log_path = make_run_log_path(
        mode="infer",
        base_dir=Path(pheromone_path).parent,
        log_out_path=log_out_path,
    )
    logger = JsonlRunLogger(run_log_path)
    logger.log_event(
        "run_start",
        {
            "mode": "infer",
            "data_path": data_path,
            "pheromone_path": pheromone_path,
            "out_path": out_path,
            "model": model,
            "agents_per_task": agents_per_task,
            "alpha": alpha,
            "beta": beta,
            "epsilon": epsilon,
            "rho": rho,
            "seed": seed,
            "aco_enabled": aco_enabled,
        },
    )
    print(
        f"Starting inference: tasks={task_count}, "
        f"agents_per_task={agents_per_task}, aco_enabled={aco_enabled}"
    )

    tasks_processed = 0
    try:
        for task_offset, (task_id, item) in enumerate(data.items()):
            task_prompt = item.get("prompt_5shot", item.get("prompt_0shot", ""))
            run_result = await process_task(
                task_id=task_id,
                task_prompt=task_prompt,
                input_meta=item,
                model=model,
                pheromone_state=pheromone_state,
                mode="infer",
                agents_per_task=agents_per_task,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                rho=rho,
                seed=seed + task_offset,
                aco_enabled=aco_enabled,
                golden_plan=None,
            )
            item["pred_5shot_pro"] = run_result.prediction
            item["pred_model"] = model
            item["aco_tot_agent_count"] = agents_per_task
            logger.log_event(
                "task_result",
                {
                    "mode": "infer",
                    "iteration": task_offset + 1,
                    "task_id": task_id,
                    "original_task_prompt": task_prompt,
                    "prediction": run_result.prediction,
                    "aco_enabled": aco_enabled,
                    "traces": [trace.to_dict() for trace in run_result.traces],
                },
            )
            tasks_processed += 1
            print(
                f"[infer] task {task_offset + 1}/{task_count} "
                f"task_id={task_id} prediction={run_result.prediction}"
            )
        logger.log_event(
            "run_end",
            {
                "mode": "infer",
                "tasks_processed": tasks_processed,
                "output_path": out_path,
                "aco_enabled": aco_enabled,
            },
        )
    finally:
        logger.close()

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(f"Inference complete: tasks={tasks_processed}")
    print(f"Output written to: {output_path}")
    print(f"Run log written to: {run_log_path}")


async def _solve_dataset_async(
    *,
    data_path: str,
    out_path: str,
    model: str,
    agents_per_task: int,
    iterations_per_task: int,
    alpha: float,
    beta: float,
    epsilon: float,
    rho: float,
    seed: int,
    aco_enabled: bool = True,
    log_out_path: str | None = None,
) -> Dict[str, Any]:
    with Path(data_path).open("r", encoding="utf-8") as handle:
        data: Dict[str, Dict[str, Any]] = json.load(handle)
    task_count = len(data)

    run_log_path = make_run_log_path(
        mode="solve",
        base_dir=Path(out_path).parent,
        log_out_path=log_out_path,
    )
    logger = JsonlRunLogger(run_log_path)
    logger.log_event(
        "run_start",
        {
            "mode": "solve",
            "data_path": data_path,
            "out_path": out_path,
            "model": model,
            "agents_per_task": agents_per_task,
            "iterations_per_task": iterations_per_task,
            "alpha": alpha,
            "beta": beta,
            "epsilon": epsilon,
            "rho": rho,
            "seed": seed,
            "aco_enabled": aco_enabled,
            "selection_policy": "final_iteration_best",
            "scoring_mode": "deterministic_task_based",
            "generation_prompt": "prompt_5shot",
            "scoring_prompt": "prompt_0shot_with_fallback",
        },
    )
    print(
        f"Starting solve: tasks={task_count}, agents_per_task={agents_per_task}, "
        f"iterations_per_task={iterations_per_task}, aco_enabled={aco_enabled}"
    )

    tasks_processed = 0
    total_quality = 0.0
    pheromone_updates_applied = 0
    try:
        for task_offset, (task_id, item) in enumerate(data.items()):
            task_prompt = item.get("prompt_5shot", item.get("prompt_0shot", ""))
            scoring_prompt = item.get("prompt_0shot") or _coerce_scoring_task_text(
                task_prompt
            )
            run_result = await process_task(
                task_id=task_id,
                task_prompt=task_prompt,
                input_meta=item,
                model=model,
                pheromone_state=create_initial_pheromone_state(
                    alpha=alpha,
                    beta=beta,
                    epsilon=epsilon,
                    rho=rho,
                    seed=seed + task_offset,
                    initial_tau=DEFAULT_INITIAL_PHEROMONE,
                ),
                mode="solve",
                agents_per_task=agents_per_task,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                rho=rho,
                seed=seed + task_offset,
                aco_enabled=aco_enabled,
                golden_plan=None,
                iterations_per_task=iterations_per_task,
                scoring_task_text_override=scoring_prompt,
            )

            metadata = run_result.metadata or {}
            best_quality = float(
                metadata.get("final_iteration_best_quality", 0.0) or 0.0
            )
            total_quality += best_quality
            pheromone_updates_applied += int(
                metadata.get("pheromone_updates_applied", 0) or 0
            )

            item["pred_5shot_pro"] = run_result.prediction
            item["pred_model"] = model
            item["aco_tot_agent_count"] = agents_per_task
            item["aco_tot_iterations_per_task"] = int(iterations_per_task)
            item["aco_tot_mode"] = "solve"
            item["aco_tot_scoring_mode"] = "deterministic_task_based"
            item["aco_tot_selection_policy"] = "final_iteration_best"
            item["aco_tot_aco_enabled"] = bool(aco_enabled)
            item["aco_tot_final_iteration_best_quality"] = best_quality
            item["aco_tot_final_iteration_best_slot_rank"] = metadata.get(
                "final_iteration_best_slot_rank"
            )
            item["aco_tot_pheromone_updates_applied"] = int(
                metadata.get("pheromone_updates_applied", 0) or 0
            )

            logger.log_event(
                "task_result",
                {
                    "mode": "solve",
                    "iteration": task_offset + 1,
                    "task_id": task_id,
                    "original_task_prompt": task_prompt,
                    "scoring_task_prompt": scoring_prompt,
                    "prediction": run_result.prediction,
                    "aco_enabled": aco_enabled,
                    "metadata": metadata,
                    "traces": [trace.to_dict() for trace in run_result.traces],
                },
            )
            tasks_processed += 1
            print(
                f"[solve] task {task_offset + 1}/{task_count} "
                f"task_id={task_id} quality={best_quality:.4f} "
                f"prediction={run_result.prediction}"
            )

        final_mean_quality = (
            total_quality / float(tasks_processed) if tasks_processed else 0.0
        )
        logger.log_event(
            "run_end",
            {
                "mode": "solve",
                "tasks_processed": tasks_processed,
                "output_path": out_path,
                "aco_enabled": aco_enabled,
                "iterations_per_task": iterations_per_task,
                "selection_policy": "final_iteration_best",
                "scoring_mode": "deterministic_task_based",
                "pheromone_updates_applied": pheromone_updates_applied,
                "final_mean_quality": final_mean_quality,
            },
        )
    finally:
        logger.close()

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    final_mean_quality = total_quality / float(tasks_processed) if tasks_processed else 0.0
    print(f"Solve complete: tasks={tasks_processed}, mean_quality={final_mean_quality:.4f}")
    print(f"Output written to: {output_path}")
    print(f"Run log written to: {run_log_path}")

    return {
        "mode": "solve",
        "tasks": tasks_processed,
        "agents_per_task": agents_per_task,
        "iterations_per_task": int(iterations_per_task),
        "aco_enabled": aco_enabled,
        "selection_policy": "final_iteration_best",
        "scoring_mode": "deterministic_task_based",
        "pheromone_updates_applied": pheromone_updates_applied,
        "final_mean_quality": final_mean_quality,
        "out_path": str(output_path),
        "run_log_path": str(run_log_path),
    }


def train_dataset(
    data_path: str,
    out_pheromone_path: str,
    model: str,
    agents_per_task: int = DEFAULT_AGENTS_PER_TASK,
    epochs: int = 1,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    aco_enabled: bool = True,
    metrics_out_path: str | None = None,
    log_out_path: str | None = None,
) -> Dict[str, Any]:
    return asyncio.run(
        _train_dataset_async(
            data_path=data_path,
            out_pheromone_path=out_pheromone_path,
            model=model,
            agents_per_task=agents_per_task,
            epochs=epochs,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
            aco_enabled=aco_enabled,
            metrics_out_path=metrics_out_path,
            log_out_path=log_out_path,
        )
    )


def infer_dataset(
    data_path: str,
    pheromone_path: str,
    out_path: str,
    model: str,
    agents_per_task: int = DEFAULT_AGENTS_PER_TASK,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    aco_enabled: bool = True,
    log_out_path: str | None = None,
) -> None:
    asyncio.run(
        _infer_dataset_async(
            data_path=data_path,
            pheromone_path=pheromone_path,
            out_path=out_path,
            model=model,
            agents_per_task=agents_per_task,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
            aco_enabled=aco_enabled,
            log_out_path=log_out_path,
        )
    )


def solve_dataset(
    data_path: str,
    out_path: str,
    model: str,
    agents_per_task: int = DEFAULT_AGENTS_PER_TASK,
    iterations_per_task: int = 3,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    epsilon: float = DEFAULT_EPSILON,
    rho: float = DEFAULT_RHO,
    seed: int = DEFAULT_SEED,
    aco_enabled: bool = True,
    log_out_path: str | None = None,
) -> Dict[str, Any]:
    return asyncio.run(
        _solve_dataset_async(
            data_path=data_path,
            out_path=out_path,
            model=model,
            agents_per_task=agents_per_task,
            iterations_per_task=iterations_per_task,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
            seed=seed,
            aco_enabled=aco_enabled,
            log_out_path=log_out_path,
        )
    )
