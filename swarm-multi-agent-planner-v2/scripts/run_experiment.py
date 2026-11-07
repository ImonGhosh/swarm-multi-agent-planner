#!/usr/bin/env python3
"""Run the ToT swarm on a calendar-scheduling dataset: ToT-only (--aco false) or ToT+ACO (--aco true).

The frozen v2 method: 8 agents x 6 iterations (48 traversals per task), no early stopping, ACO
without a heuristic (p ~ (tau + eps)^alpha), pheromone deposit only for answers the verifier
scores 1.0, evaporation rho = 0.1. The answer is chosen after the last iteration by
--selection_policy (default: the headline rule verified_last3); the event log keeps every trace,
so scripts/derive_predictions.py can derive every other selection rule afterwards.

Everything a run produces goes into --run_dir (default results/runs/<tot_only|tot_aco>/seed_<seed>):
  predictions.json      the dataset with predictions, in the official evaluator's format
  events.jsonl.gz       run_start, iteration, llm_call, task_result, run_end events
  summary.json          totals, LLM calls, cache hits, cost
  config.resolved.json  the full resolved configuration

LLM responses are cached in --cache_path (SQLite), so a repeated or interrupted run replays
finished calls without API calls. Pass --cache_path none to disable the cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from swarm_planner.aco import DEFAULT_ALPHA, DEFAULT_EPSILON, DEFAULT_RHO
from swarm_planner.selection_rules import HEADLINE_RULE, SELECTION_RULES
from swarm_planner.solver import (
    DEFAULT_AGENTS_PER_TASK,
    DEFAULT_ITERATIONS_PER_TASK,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_SEED,
    DEFAULT_TASK_CONCURRENCY,
    solve_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "results" / "cache" / "llm_cache.sqlite"


def str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", default=str(PROJECT_ROOT / "data" / "curated_400.json"))
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Output directory. Default: results/runs/<tot_only|tot_aco>/seed_<seed>.",
    )
    parser.add_argument(
        "--cache_path",
        default=str(DEFAULT_CACHE),
        help="SQLite LLM response cache, or 'none' to disable it.",
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--agents_per_task", type=int, default=DEFAULT_AGENTS_PER_TASK)
    parser.add_argument("--iterations_per_task", type=int, default=DEFAULT_ITERATIONS_PER_TASK)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--aco",
        type=str_to_bool,
        default=True,
        help="true: ToT+ACO (pheromone-guided branch choice). false: ToT-only (uniform random branches).",
    )
    parser.add_argument(
        "--selection_policy",
        choices=SELECTION_RULES,
        default=HEADLINE_RULE,
        help="Rule that picks the reported answer (all rules are derived offline anyway).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only solve the first N tasks of the data file.")
    parser.add_argument(
        "--task_concurrency",
        type=int,
        default=DEFAULT_TASK_CONCURRENCY,
        help="Tasks solved at the same time. Tasks are independent, so this only changes speed.",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_REQUESTS,
        help="Global cap on in-flight provider requests.",
    )
    parser.add_argument(
        "--max_cost_usd",
        type=float,
        default=None,
        help="Stop (resumably, via the cache) once this run has paid this much for provider calls.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    config_name = "tot_aco" if args.aco else "tot_only"
    run_dir = args.run_dir or str(PROJECT_ROOT / "results" / "runs" / config_name / f"seed_{args.seed}")
    cache_path = None if args.cache_path.strip().lower() == "none" else args.cache_path
    summary = solve_dataset(
        data_path=args.data_path,
        run_dir=run_dir,
        model=args.model,
        agents_per_task=args.agents_per_task,
        iterations_per_task=args.iterations_per_task,
        alpha=args.alpha,
        epsilon=args.epsilon,
        rho=args.rho,
        seed=args.seed,
        aco_enabled=args.aco,
        selection_policy=args.selection_policy,
        limit=args.limit,
        task_concurrency=args.task_concurrency,
        max_concurrent_requests=args.max_concurrent_requests,
        max_cost_usd=args.max_cost_usd,
        cache_path=cache_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
