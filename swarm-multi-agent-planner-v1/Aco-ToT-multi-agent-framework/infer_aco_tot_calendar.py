#!/usr/bin/env python3
"""Run ACO-ToT inference on calendar scheduling dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from aco_tot.engine import infer_dataset, solve_dataset


def load_project_env() -> None:
    current_file = Path(__file__).resolve()
    for parent in [current_file.parent, *current_file.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return

    load_dotenv()


def str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="infer", choices=["infer", "solve"])
    parser.add_argument(
        "--data_path",
        default=str(project_root / "data" / "calendar_scheduling_input.json"),
    )
    parser.add_argument(
        "--out_path",
        default=str(project_root / "data" / "calendar_scheduling_output.json"),
    )
    parser.add_argument(
        "--pheromone_path",
        default=str(
            project_root / "Aco-ToT-multi-agent-framework" / "pheromones.json"
        ),
    )
    parser.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--agents_per_task", type=int, default=8)
    parser.add_argument(
        "--iterations_per_task",
        type=int,
        default=3,
        help="Only used in solve mode. Number of per-task ACO iterations.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aco",
        type=str_to_bool,
        default=True,
        help="If true, use ACO-guided branch selection. If false, use uniform random ToT traversal.",
    )
    return parser.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    if args.mode == "solve":
        summary = solve_dataset(
            data_path=args.data_path,
            out_path=args.out_path,
            model=args.model,
            agents_per_task=args.agents_per_task,
            iterations_per_task=args.iterations_per_task,
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.epsilon,
            rho=args.rho,
            seed=args.seed,
            aco_enabled=args.aco,
        )
        print(json.dumps(summary, indent=2))
        return

    infer_dataset(
        data_path=args.data_path,
        pheromone_path=args.pheromone_path,
        out_path=args.out_path,
        model=args.model,
        agents_per_task=args.agents_per_task,
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        rho=args.rho,
        seed=args.seed,
        aco_enabled=args.aco,
    )


if __name__ == "__main__":
    main()
