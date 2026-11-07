"""The three non-swarm rows of the Phase 4 table.

``run_baseline``         single-SLM baseline: ONE call per task with the 5-shot prompt, temperature 0.
``run_sampling_control`` equal-budget control: 48 direct calls per task (the same prompt) at
                         temperature 0.7, the answer with the best verifier result is submitted. It
                         separates the effect of the ToT structure from "many samples + verifier".
``run_verifier_solver``  reference row, no LLM: the verifier's own best slot (finding F4).

All three write the same run directory as the swarm runs (see run_io.py).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .llm.client import LLM_MAX_TOKENS, LLM_TOP_P
from .run_io import CONFIG_FILENAME, PREDICTIONS_FILENAME, SUMMARY_FILENAME, RunContext, load_dataset
from .runinfo import collect_environment, file_sha256
from .selection_rules import FALLBACK_PREDICTION, prediction_of, rec_from_log, select
from .solver import scorer_for_item, verifier_record
from .verifier.parser import canonicalize_final_answer
from .verifier.scorer import _format_slot

SAMPLING_TEMPERATURE = 0.7
SAMPLING_SAMPLES = 48  # = agents x iterations of the swarm
BASELINE_RULE = "single_call"
SAMPLING_RULE = "best_of_samples"


def _baseline_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """The single-call request: the dataset's 5-shot prompt as the only (user) message."""
    return [{"role": "user", "content": item["prompt_5shot"]}]


async def _run_tasks(data: Dict[str, Dict[str, Any]], concurrency: int, work) -> None:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def one(offset: int, task_id: str, item: Dict[str, Any]) -> None:
        async with semaphore:
            await work(offset, task_id, item)

    tasks = [asyncio.ensure_future(one(i, t, it)) for i, (t, it) in enumerate(data.items())]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _baseline_async(
    *,
    data_path: str,
    run_dir: str,
    model: str,
    seed: int,
    seed_as_nonce: bool,
    limit: int | None,
    task_concurrency: int,
    max_concurrent_requests: int,
    max_cost_usd: float | None,
    cache_path: str | None,
    client: Any,
) -> Dict[str, Any]:
    data = load_dataset(data_path, limit)
    total = len(data)
    ctx = RunContext(
        run_dir=run_dir,
        model=model,
        cache_path=cache_path,
        max_concurrent_requests=max_concurrent_requests,
        max_cost_usd=max_cost_usd,
        client=client,
    )
    # D4: with the seed as nonce every seed is a real provider call, so the SD over seeds
    # shows the provider's own noise at temperature 0. Without it all seeds replay one answer.
    nonce = f"baseline:seed{seed}" if seed_as_nonce else None
    done = {"n": 0, "valid": 0, "perfect": 0}
    config = {
        "mode": "baseline",
        "config_name": "baseline",
        "model": model,
        "seed": seed,
        "nonce": nonce,
        "calls_per_task": 1,
        "limit": limit,
        "tasks": total,
        "prompt": "prompt_5shot as a single user message",
        "selection_policy": BASELINE_RULE,
        "cache_path": cache_path,
        "task_concurrency": task_concurrency,
        "max_concurrent_requests": max_concurrent_requests,
    }
    try:
        ctx.start(config, data_path)
        print(f"Starting baseline: tasks={total}, seed={seed}, nonce={nonce}")

        async def work(offset: int, task_id: str, item: Dict[str, Any]) -> None:
            started = time.perf_counter()
            response = await ctx.gateway.chat(
                _baseline_messages(item),
                nonce=nonce,
                meta={"task_id": task_id, "step": "baseline"},
            )
            result = scorer_for_item(item).score_prediction(response.text)
            details = result.to_dict()
            item["pred_5shot_pro"] = response.text
            item["pred_model"] = model
            ctx.event_log.log(
                "task_result",
                {
                    "task_id": task_id,
                    "position": offset + 1,
                    "prediction": response.text,
                    "q": result.score,
                    "v": verifier_record(details),
                    "llm": ctx.gateway.stats_for(task_id).to_dict(),
                    "wall_s": round(time.perf_counter() - started, 3),
                },
                flush=True,
            )
            done["n"] += 1
            done["valid"] += int(result.hard_valid)
            done["perfect"] += int(result.score >= 1.0)
            if done["n"] % 20 == 0 or done["n"] == total:
                ctx.progress("baseline", done["n"], total, f"verified so far {done['perfect']}")

        await _run_tasks(data, task_concurrency, work)
        summary = ctx.finish(
            data,
            {
                "mode": "baseline",
                "config_name": "baseline",
                "tasks": done["n"],
                "seed": seed,
                "selection_policy": BASELINE_RULE,
                "hard_valid_share": done["valid"] / done["n"] if done["n"] else 0.0,
                "verified_share": done["perfect"] / done["n"] if done["n"] else 0.0,
            },
        )
    finally:
        ctx.close()
    print(f"Baseline complete: tasks={done['n']}, verified answers={done['perfect']}")
    return summary


def run_baseline(
    data_path: str,
    run_dir: str,
    model: str,
    seed: int = 0,
    seed_as_nonce: bool = True,
    limit: int | None = None,
    task_concurrency: int = 8,
    max_concurrent_requests: int = 16,
    max_cost_usd: float | None = None,
    cache_path: str | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    return asyncio.run(
        _baseline_async(
            data_path=data_path,
            run_dir=run_dir,
            model=model,
            seed=seed,
            seed_as_nonce=seed_as_nonce,
            limit=limit,
            task_concurrency=task_concurrency,
            max_concurrent_requests=max_concurrent_requests,
            max_cost_usd=max_cost_usd,
            cache_path=cache_path,
            client=client,
        )
    )


def sample_record(agent: int, text: str, details: Dict[str, Any], score: float) -> Dict[str, Any]:
    """One sample in the ``samples`` event; the same shape as a trace of an ``iteration`` event."""
    return {
        "agent": agent,
        "ans": canonicalize_final_answer(text),
        "ps": 0.0,
        "q": score,
        "v": verifier_record(details),
    }


async def _sampling_async(
    *,
    data_path: str,
    run_dir: str,
    model: str,
    seed: int,
    samples: int,
    temperature: float,
    limit: int | None,
    task_concurrency: int,
    max_concurrent_requests: int,
    max_cost_usd: float | None,
    cache_path: str | None,
    client: Any,
) -> Dict[str, Any]:
    data = load_dataset(data_path, limit)
    total = len(data)
    ctx = RunContext(
        run_dir=run_dir,
        model=model,
        cache_path=cache_path,
        max_concurrent_requests=max_concurrent_requests,
        max_cost_usd=max_cost_usd,
        temperature=temperature,
        client=client,
    )
    done = {"n": 0, "perfect": 0}
    config = {
        "mode": "sampling_control",
        "config_name": "sampling_control",
        "model": model,
        "seed": seed,
        "samples_per_task": samples,
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
        "top_p": LLM_TOP_P,
        "nonce": f"sample:seed{seed}:<i>",
        "limit": limit,
        "tasks": total,
        "prompt": "prompt_5shot as a single user message",
        "selection_policy": SAMPLING_RULE,
        "cache_path": cache_path,
        "task_concurrency": task_concurrency,
        "max_concurrent_requests": max_concurrent_requests,
    }
    try:
        ctx.start(config, data_path)
        print(f"Starting sampling control: tasks={total}, samples={samples}, T={temperature}, seed={seed}")

        async def work(offset: int, task_id: str, item: Dict[str, Any]) -> None:
            started = time.perf_counter()
            messages = _baseline_messages(item)
            scorer = scorer_for_item(item)

            async def one_sample(i: int):
                response = await ctx.gateway.chat(
                    messages,
                    nonce=f"sample:seed{seed}:{i}",  # per-sample nonce: samples must not collapse
                    meta={"task_id": task_id, "agent": i, "step": "sample"},
                )
                result = scorer.score_prediction(canonicalize_final_answer(response.text))
                return sample_record(i, response.text, result.to_dict(), result.score)

            try:
                records = await asyncio.gather(*[one_sample(i) for i in range(samples)])
            except BaseException as err:
                ctx.event_log.log(
                    "task_error", {"task_id": task_id, "error": f"{type(err).__name__}: {err}"}, flush=True
                )
                raise
            recs = [rec_from_log(0, r) for r in records]
            chosen = select(recs, "best_of_all")
            prediction = prediction_of(chosen)
            item["pred_5shot_pro"] = prediction
            item["pred_model"] = model
            ctx.event_log.log("samples", {"task_id": task_id, "samples": records})
            ctx.event_log.log(
                "task_result",
                {
                    "task_id": task_id,
                    "position": offset + 1,
                    "prediction": prediction,
                    "q": chosen.q if chosen else 0.0,
                    "verified_samples": sum(1 for r in recs if r.verified),
                    "llm": ctx.gateway.stats_for(task_id).to_dict(),
                    "wall_s": round(time.perf_counter() - started, 3),
                },
                flush=True,
            )
            done["n"] += 1
            done["perfect"] += int(bool(chosen and chosen.verified))
            ctx.progress(
                "sampling",
                done["n"],
                total,
                f"task_id={task_id} verified_samples={sum(1 for r in recs if r.verified)}/{samples}",
            )

        await _run_tasks(data, task_concurrency, work)
        summary = ctx.finish(
            data,
            {
                "mode": "sampling_control",
                "config_name": "sampling_control",
                "tasks": done["n"],
                "seed": seed,
                "samples_per_task": samples,
                "temperature": temperature,
                "selection_policy": SAMPLING_RULE,
                "selected_verified_share": done["perfect"] / done["n"] if done["n"] else 0.0,
            },
        )
    finally:
        ctx.close()
    print(f"Sampling control complete: tasks={done['n']}, tasks with a verified sample={done['perfect']}")
    return summary


def run_sampling_control(
    data_path: str,
    run_dir: str,
    model: str,
    seed: int = 0,
    samples: int = SAMPLING_SAMPLES,
    temperature: float = SAMPLING_TEMPERATURE,
    limit: int | None = None,
    task_concurrency: int = 4,
    max_concurrent_requests: int = 32,
    max_cost_usd: float | None = None,
    cache_path: str | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    return asyncio.run(
        _sampling_async(
            data_path=data_path,
            run_dir=run_dir,
            model=model,
            seed=seed,
            samples=samples,
            temperature=temperature,
            limit=limit,
            task_concurrency=task_concurrency,
            max_concurrent_requests=max_concurrent_requests,
            max_cost_usd=max_cost_usd,
            cache_path=cache_path,
            client=client,
        )
    )


def run_verifier_solver(data_path: str, run_dir: str, limit: int | None = None) -> Dict[str, Any]:
    """Answer every task with the verifier's best slot. No LLM, deterministic."""
    data = load_dataset(data_path, limit)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": "verifier_solver",
        "config_name": "verifier_solver",
        "limit": limit,
        "tasks": len(data),
        "data_path": data_path,
        "data_sha256": file_sha256(data_path),
        "selection_policy": "verifier_best_slot",
        "llm_calls": 0,
        "environment": collect_environment(),
    }
    (run_path / CONFIG_FILENAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
    no_slot = 0
    for item in data.values():
        slot = scorer_for_item(item).best_slot
        if slot is None:
            no_slot += 1
            item["pred_5shot_pro"] = FALLBACK_PREDICTION
        else:
            item["pred_5shot_pro"] = f"Here is the proposed time: {_format_slot(*slot)}"
        item["pred_model"] = "verifier_best_slot"
    with (run_path / PREDICTIONS_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    summary = {
        "mode": "verifier_solver",
        "config_name": "verifier_solver",
        "tasks": len(data),
        "tasks_without_valid_slot": no_slot,
        "selection_policy": "verifier_best_slot",
        "out_path": str(run_path / PREDICTIONS_FILENAME),
        "run_dir": str(run_path),
    }
    (run_path / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Verifier-as-solver: tasks={len(data)}, tasks without a valid slot={no_slot}")
    print(f"Run directory: {run_path}")
    return summary


__all__ = [
    "BASELINE_RULE",
    "SAMPLING_RULE",
    "SAMPLING_SAMPLES",
    "SAMPLING_TEMPERATURE",
    "run_baseline",
    "run_sampling_control",
    "run_verifier_solver",
]
