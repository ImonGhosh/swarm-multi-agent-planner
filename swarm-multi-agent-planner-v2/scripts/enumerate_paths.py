#!/usr/bin/env python3
"""Build the path table: run ALL 243 ToT paths of every task and score each answer.

This is the one LLM-heavy step of Phase 3: 606 calls per task, about $0.07 per task (about $1.3
for --limit 20, about $7 for all 100 dev tasks). Every answer goes into the LLM cache, so:

  * stopping and rerunning the same command continues where it stopped,
  * an interrupted run (Ctrl+C or an error) still writes the table for the tasks it finished, and
  * later ToT-only / ToT+ACO runs on the same tasks reuse these answers.

Outputs go to --out_dir (see swarm_planner/path_table.py for the file formats).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from swarm_planner.path_table import enumerate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "results" / "cache" / "llm_cache.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", default=str(PROJECT_ROOT / "data" / "dev_100.json"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "path_table" / "dev_100"))
    parser.add_argument(
        "--cache_path",
        default=str(DEFAULT_CACHE),
        help="SQLite LLM response cache, or 'none' to disable it.",
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N tasks, in --order. With the default order, N=20 gives 2 tasks "
        "from each of the 10 categories.",
    )
    parser.add_argument(
        "--order",
        choices=["interleaved", "file"],
        default="interleaved",
        help="interleaved: round-robin over the (people, days) categories, so any prefix is "
        "balanced. file: the order of the data file (category by category).",
    )
    parser.add_argument("--task_concurrency", type=int, default=4, help="Tasks walked at the same time.")
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=16,
        help="Global cap on in-flight provider requests.",
    )
    parser.add_argument(
        "--max_cost_usd",
        type=float,
        default=15.0,
        help="Stop starting new tasks once this much has been paid (the estimate is about $7).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    summary = enumerate_dataset(
        data_path=args.data_path,
        out_dir=args.out_dir,
        model=args.model,
        limit=args.limit,
        order=args.order,
        task_concurrency=args.task_concurrency,
        max_concurrent_requests=args.max_concurrent_requests,
        cache_path=None if args.cache_path.strip().lower() == "none" else args.cache_path,
        max_cost_usd=args.max_cost_usd,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
