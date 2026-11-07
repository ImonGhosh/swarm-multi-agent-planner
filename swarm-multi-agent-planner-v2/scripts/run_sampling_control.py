#!/usr/bin/env python3
"""Equal-budget sampling control: 48 direct samples per task at temperature 0.7, best one by the verifier.

Same prompt as the baseline, no Tree-of-Thought and no pheromone. It spends the same number of
traversals as the swarm (8 agents x 6 iterations = 48) but each is ONE direct call, so it separates
the effect of the ToT structure from "many samples + verifier selection". Every sample has its own
cache nonce, so the samples do not collapse into one cached answer.

Output: results/runs/sampling_control/seed_<seed>/ (predictions.json = best sample by the verifier,
events.jsonl.gz with one `samples` event per task, summary.json, config.resolved.json).
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from run_experiment import DEFAULT_CACHE, PROJECT_ROOT
from swarm_planner.baseline import SAMPLING_SAMPLES, SAMPLING_TEMPERATURE, run_sampling_control


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", default=str(PROJECT_ROOT / "data" / "curated_400.json"))
    parser.add_argument("--run_dir", default=None, help="Default: results/runs/sampling_control/seed_<seed>.")
    parser.add_argument("--cache_path", default=str(DEFAULT_CACHE))
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=SAMPLING_SAMPLES)
    parser.add_argument("--temperature", type=float, default=SAMPLING_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task_concurrency", type=int, default=4)
    parser.add_argument("--max_concurrent_requests", type=int, default=32)
    parser.add_argument("--max_cost_usd", type=float, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or str(PROJECT_ROOT / "results" / "runs" / "sampling_control" / f"seed_{args.seed}")
    cache_path = None if args.cache_path.strip().lower() == "none" else args.cache_path
    summary = run_sampling_control(
        data_path=args.data_path,
        run_dir=run_dir,
        model=args.model,
        seed=args.seed,
        samples=args.samples,
        temperature=args.temperature,
        limit=args.limit,
        task_concurrency=args.task_concurrency,
        max_concurrent_requests=args.max_concurrent_requests,
        max_cost_usd=args.max_cost_usd,
        cache_path=cache_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
