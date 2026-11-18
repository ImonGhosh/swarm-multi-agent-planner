#!/usr/bin/env python3
"""Reference row: answer every task with the verifier's own best slot. No LLM, no cost, deterministic.

It shows how much of any result the verifier alone explains (finding F4: 92.75% on the 400 test tasks).

Settings can come from a YAML file: --config configs/verifier_solver.yaml (flags override the file).

Output: results/runs/verifier_solver/ (predictions.json, summary.json, config.resolved.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_planner.baseline import run_verifier_solver
from swarm_planner.run_config import add_config_argument, parse_args_with_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--data_path", default=str(PROJECT_ROOT / "data" / "curated_400.json"))
    parser.add_argument("--run_dir", default=str(PROJECT_ROOT / "results" / "runs" / "verifier_solver"))
    parser.add_argument("--limit", type=int, default=None)
    return parse_args_with_config(parser, argv, PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    print(json.dumps(run_verifier_solver(args.data_path, args.run_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
