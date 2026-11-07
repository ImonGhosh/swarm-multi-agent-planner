#!/usr/bin/env python3
"""Single-SLM baseline: one call per task with the 5-shot prompt (temperature 0, no ToT, no ACO).

By default the seed is used as a cache nonce (decision D4), so every seed makes real provider calls
and the SD over seeds shows the provider's own noise at temperature 0. With --seed_as_nonce false
all seeds replay one cached answer per task (SD exactly 0).

Output: results/runs/baseline/seed_<seed>/ (predictions.json, events.jsonl.gz, summary.json, config.resolved.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from run_experiment import DEFAULT_CACHE, PROJECT_ROOT, str_to_bool  # same folder
from swarm_planner.baseline import run_baseline


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", default=str(PROJECT_ROOT / "data" / "curated_400.json"))
    parser.add_argument("--run_dir", default=None, help="Default: results/runs/baseline/seed_<seed>.")
    parser.add_argument("--cache_path", default=str(DEFAULT_CACHE))
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_as_nonce", type=str_to_bool, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task_concurrency", type=int, default=8)
    parser.add_argument("--max_concurrent_requests", type=int, default=16)
    parser.add_argument("--max_cost_usd", type=float, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or str(PROJECT_ROOT / "results" / "runs" / "baseline" / f"seed_{args.seed}")
    cache_path = None if args.cache_path.strip().lower() == "none" else args.cache_path
    summary = run_baseline(
        data_path=args.data_path,
        run_dir=run_dir,
        model=args.model,
        seed=args.seed,
        seed_as_nonce=args.seed_as_nonce,
        limit=args.limit,
        task_concurrency=args.task_concurrency,
        max_concurrent_requests=args.max_concurrent_requests,
        max_cost_usd=args.max_cost_usd,
        cache_path=cache_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
