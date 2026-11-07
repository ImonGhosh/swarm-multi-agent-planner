"""Enumerate every path of the 5-level Tree-of-Thought and score each leaf answer.

For each task the 3 x 3 x 3 x 3 x 3 = 243 paths are walked exhaustively (363 intermediate calls
plus 243 finalize calls = 606 distinct LLM requests). The requests are built by the same
functions as the solver's (``tot.prompts``), so the cached answers are reused by ToT-only and
ToT+ACO runs on the same tasks.

Outputs (in the output directory):

``path_table.csv.gz``   one row per path: branch choices, request hashes, answer, verifier
                        result and exact-match correctness against the gold answer
``nodes.jsonl.gz``      one row per LLM call: node id, step, request hash, answer text, tokens
``events.jsonl.gz``     run_start, llm_call (one per call), task_result, task_error, run_end
``summary.json``, ``config.resolved.json``

Node ids are branch indices joined by dashes ("0-2-1" is branch 0 at level 1, 2 at level 2,
1 at level 3). Branch indices follow the order in ``run_start.branch_order``.
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .llm.cache import LLMCache
from .llm.gateway import LLMGateway
from .logging import EVENTS_FILENAME, EventLog
from .runinfo import collect_environment, file_sha256
from .tot.context import coerce_original_prompt, extract_current_task_from_prompt5shot
from .tot.levels import LEVEL_CONFIGS, all_edge_keys
from .tot.prompts import clean_intermediate, final_messages, level_messages
from .types import ThoughtResult
from .verifier.parser import answers_match, canonicalize_final_answer
from .verifier.scorer import build_task_quality_scorer

PATH_TABLE_FILENAME = "path_table.csv.gz"
NODES_FILENAME = "nodes.jsonl.gz"
SUMMARY_FILENAME = "summary.json"
CONFIG_FILENAME = "config.resolved.json"

LEVELS = len(LEVEL_CONFIGS)
PATH_COLUMNS: List[str] = (
    ["task_id", "path"]
    + [f"b{level}" for level in range(1, LEVELS + 1)]
    + [f"req{level}" for level in range(1, LEVELS + 1)]
    + [
        "req_final",
        "answer",
        "score",
        "hard_valid",
        "soft_satisfied",
        "soft_total",
        "slot_rank",
        "correct",
    ]
)


def paths_per_task() -> int:
    total = 1
    for cfg in LEVEL_CONFIGS:
        total *= len(cfg.branches)
    return total


def calls_per_task() -> int:
    """Distinct LLM requests per task: every tree node plus one finalize call per leaf."""
    nodes, width = 0, 1
    for cfg in LEVEL_CONFIGS:
        width *= len(cfg.branches)
        nodes += width
    return nodes + width


async def enumerate_task(
    *,
    task_id: str,
    item: Dict[str, Any],
    gateway: LLMGateway,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Walk all paths of one task. Returns (path rows, node rows)."""
    prompt_5shot = item.get("prompt_5shot", item.get("prompt_0shot", ""))
    original_prompt = coerce_original_prompt(prompt_5shot)
    scoring_text = item.get("prompt_0shot") or extract_current_task_from_prompt5shot(prompt_5shot)
    scorer = build_task_quality_scorer(scoring_text)
    gold = item["golden_plan"]

    rows: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []

    def node_row(node: str, step: str, response) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "node": node,
            "step": step,
            "req": response.request_hash,
            "text": response.text,
            "in": response.prompt_tokens,
            "out": response.completion_tokens,
            "fin": response.finish_reason,
        }

    async def leaf(results: List[ThoughtResult], indices: Tuple[int, ...]) -> None:
        node = "-".join(map(str, indices))
        response = await gateway.chat(
            final_messages(original_prompt=original_prompt, thought_results=results),
            meta={"task_id": task_id, "step": "final", "node": node},
        )
        answer = canonicalize_final_answer(response.text)
        quality = scorer.score_prediction(answer)
        row: Dict[str, Any] = {"task_id": task_id, "path": node}
        for level, result in enumerate(results, start=1):
            row[f"b{level}"] = result.branch_id
            row[f"req{level}"] = result.request_hash
        row.update(
            {
                "req_final": response.request_hash,
                "answer": answer,
                "score": quality.score,
                "hard_valid": quality.hard_valid,
                "soft_satisfied": quality.soft_satisfied,
                "soft_total": quality.soft_total,
                "slot_rank": quality.slot_rank,
                "correct": answers_match(answer, gold),
            }
        )
        rows.append(row)
        nodes.append(node_row(node, "final", response))

    async def expand(results: List[ThoughtResult], indices: Tuple[int, ...]) -> None:
        level = len(results) + 1
        if level > LEVELS:
            await leaf(results, indices)
            return

        async def child(index: int, branch) -> None:
            node = "-".join(map(str, indices + (index,)))
            response = await gateway.chat(
                level_messages(
                    level=level,
                    original_prompt=original_prompt,
                    previous_results=results,
                    branch=branch,
                ),
                meta={"task_id": task_id, "step": f"L{level}", "node": node},
            )
            nodes.append(node_row(node, f"L{level}", response))
            thought = ThoughtResult(
                level=level,
                branch_id=branch.branch_id,
                branch_name=branch.name,
                thought_prompt=branch.thought_prompt,
                intermediate_result=clean_intermediate(response.text),
                selection_probability=0.0,
                request_hash=response.request_hash,
            )
            await expand(results + [thought], indices + (index,))

        branches = LEVEL_CONFIGS[level - 1].branches
        await asyncio.gather(*[child(i, b) for i, b in enumerate(branches)])

    await expand([], ())
    rows.sort(key=lambda r: tuple(int(i) for i in r["path"].split("-")))
    nodes.sort(key=lambda n: (len(n["node"].split("-")), n["node"], n["step"]))
    return rows, nodes


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def interleave_by_category(data: Dict[str, Dict[str, Any]]) -> List[str]:
    """Task ids in round-robin order over the (num_people, num_days) categories.

    The dataset lists tasks category by category, so a prefix of the file covers only a few
    categories. Round-robin order makes every prefix balanced: with ``--limit 20`` on the dev
    set it yields 2 tasks from each of the 10 categories.
    """
    groups: Dict[Tuple[int, int], List[str]] = {}
    for task_id, item in data.items():
        key = (int(item.get("num_people", 0)), int(item.get("num_days", 0)))
        groups.setdefault(key, []).append(task_id)
    ordered: List[str] = []
    queues = list(groups.values())
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def _write_outputs(
    out: Path,
    *,
    completed: List[str],
    results: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
) -> None:
    with gzip.open(out / PATH_TABLE_FILENAME, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_COLUMNS)
        writer.writeheader()
        for task_id in completed:
            writer.writerows(results[task_id][0])
    with gzip.open(out / NODES_FILENAME, "wt", encoding="utf-8") as handle:
        for task_id in completed:
            for node in results[task_id][1]:
                handle.write(json.dumps(node, ensure_ascii=False, separators=(",", ":")) + "\n")


async def enumerate_dataset_async(
    *,
    data_path: str,
    out_dir: str,
    model: str,
    limit: int | None = None,
    order: str = "interleaved",
    task_concurrency: int = 4,
    max_concurrent_requests: int = 16,
    cache_path: str | None = None,
    max_cost_usd: float | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    data: Dict[str, Dict[str, Any]] = json.loads(Path(data_path).read_text(encoding="utf-8"))
    file_order = list(data)
    if order == "interleaved":
        processing_order = interleave_by_category(data)
    elif order == "file":
        processing_order = file_order
    else:
        raise ValueError(f"order must be 'interleaved' or 'file', got {order!r}")
    if limit is not None:
        processing_order = processing_order[: max(0, int(limit))]
    task_ids = processing_order

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_log = EventLog(out / EVENTS_FILENAME)
    cache = LLMCache(cache_path) if cache_path else None
    gateway = LLMGateway(
        model=model,
        cache=cache,
        client=client,
        max_concurrent_requests=max_concurrent_requests,
        event_sink=event_log.log_row,
    )

    config = {
        "mode": "enumerate_paths",
        "data_path": data_path,
        "data_sha256": file_sha256(data_path),
        "limit": limit,
        "order": order,
        "task_ids": task_ids,
        "tasks": len(task_ids),
        "model": model,
        "paths_per_task": paths_per_task(),
        "llm_calls_per_task": calls_per_task(),
        "llm_params": gateway.params,
        "provider_preferences": gateway.provider_preferences(),
        "cache_path": cache_path,
        "task_concurrency": task_concurrency,
        "max_concurrent_requests": max_concurrent_requests,
        "max_cost_usd": max_cost_usd,
    }
    (out / CONFIG_FILENAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
    event_log.log(
        "run_start",
        {
            "config": config,
            "branch_order": {cfg.level: [b.branch_id for b in cfg.branches] for cfg in LEVEL_CONFIGS},
            "edge_order": all_edge_keys(),
            "path_columns": PATH_COLUMNS,
            "environment": collect_environment(),
        },
        flush=True,
    )
    print(
        f"Enumerating {len(task_ids)} tasks x {paths_per_task()} paths "
        f"({calls_per_task()} LLM calls per task, {len(task_ids) * calls_per_task()} in total, "
        f"fewer when cached). Order: {order}. Cost cap: "
        f"{'none' if max_cost_usd is None else f'${max_cost_usd:.2f}'}."
    )

    results: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    skipped: List[str] = []
    semaphore = asyncio.Semaphore(max(1, int(task_concurrency)))
    totals = {"done": 0, "correct": 0}

    async def run_one(task_id: str) -> None:
        async with semaphore:
            if max_cost_usd is not None and gateway.stats.cost_usd_paid >= max_cost_usd:
                skipped.append(task_id)
                return
            task_started = time.perf_counter()
            try:
                rows, nodes = await enumerate_task(task_id=task_id, item=data[task_id], gateway=gateway)
            except asyncio.CancelledError:
                raise  # stopped because another task failed or the run was interrupted
            except BaseException as err:
                event_log.log(
                    "task_error",
                    {"task_id": task_id, "error": f"{type(err).__name__}: {err}"},
                    flush=True,
                )
                raise
            results[task_id] = (rows, nodes)
            n_correct = sum(1 for r in rows if r["correct"])
            n_valid = sum(1 for r in rows if r["hard_valid"])
            n_top = sum(1 for r in rows if r["score"] == 1.0)
            totals["done"] += 1
            totals["correct"] += n_correct
            event_log.log(
                "task_result",
                {
                    "task_id": task_id,
                    "paths": len(rows),
                    "nodes": len(nodes),
                    "correct": n_correct,
                    "hard_valid": n_valid,
                    "score_1": n_top,
                    "llm": gateway.stats_for(task_id).to_dict(),
                    "wall_s": round(time.perf_counter() - task_started, 3),
                },
                flush=True,
            )
            elapsed = time.perf_counter() - started
            remaining = (elapsed / totals["done"]) * (len(task_ids) - totals["done"] - len(skipped))
            stats = gateway.stats
            print(
                f"[enum] {totals['done']}/{len(task_ids)} {task_id} "
                f"correct={n_correct}/{len(rows)} valid={n_valid} | "
                f"calls={stats.calls} hits={stats.cache_hits} paid=${stats.cost_usd_paid:.3f} | "
                f"elapsed {_fmt_duration(elapsed)} eta {_fmt_duration(remaining)}",
                flush=True,
            )

    pending_error: BaseException | None = None
    try:
        tasks = [asyncio.ensure_future(run_one(task_id)) for task_id in task_ids]
        try:
            await asyncio.gather(*tasks)
        except BaseException as err:  # includes Ctrl+C (cancellation)
            pending_error = err
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Whatever finished is written, also after an error or an interruption.
        completed = [t for t in task_ids if t in results]
        rows_total = sum(len(results[t][0]) for t in completed)
        wall = round(time.perf_counter() - started, 3)
        complete = pending_error is None and len(completed) == len(task_ids)
        run_end = {
            "tasks_completed": len(completed),
            "tasks_skipped_by_cost_cap": len(skipped),
            "complete": complete,
            "interrupted": pending_error is not None,
            "paths": rows_total,
            "correct_paths": totals["correct"],
            "llm": gateway.stats.to_dict(),
            "cache_entries": len(cache) if cache is not None else None,
            "wall_s": wall,
        }
        event_log.log("run_end", run_end, flush=True)
        _write_outputs(out, completed=completed, results=results)
    finally:
        event_log.close()
        if cache is not None:
            cache.close()

    summary = {
        "mode": "enumerate_paths",
        "complete": complete,
        "interrupted": pending_error is not None,
        "error": (
            f"{type(pending_error).__name__}: {pending_error}" if pending_error is not None else None
        ),
        "order": order,
        "tasks_total": len(task_ids),
        "tasks_completed": len(completed),
        "completed_task_ids": completed,
        "tasks_skipped_by_cost_cap": skipped,
        "expected_paths": len(completed) * paths_per_task(),
        "paths": rows_total,
        "correct_paths": totals["correct"],
        "llm": run_end["llm"],
        "cache_entries": run_end["cache_entries"],
        "wall_s": wall,
        "out_dir": str(out),
    }
    (out / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    llm = run_end["llm"]
    print(
        f"{'Done' if complete else 'Partial'}: {len(completed)}/{len(task_ids)} tasks, "
        f"{rows_total} paths, {totals['correct']} correct paths."
    )
    print(
        f"LLM calls={llm['llm_calls']} cache_hits={llm['cache_hits']} "
        f"paid=${llm['cost_usd_paid']:.4f} uncached_equivalent=${llm['cost_usd_uncached_equivalent']:.4f} "
        f"wall={_fmt_duration(wall)}"
    )
    if pending_error is not None:
        print(
            f"The run was interrupted or failed ({type(pending_error).__name__}). The path table "
            f"covers the {len(completed)} task(s) that finished and was written to {out}. "
            "Finished calls are cached; rerun the same command to continue."
        )
        raise pending_error
    if not complete:
        print(
            f"STOPPED by the cost cap: {len(skipped)} tasks were not started. "
            "Finished calls are cached; rerun the same command (optionally with a higher "
            "--max_cost_usd) to continue."
        )
    print(f"Output directory: {out}")
    return summary


def enumerate_dataset(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(enumerate_dataset_async(**kwargs))
