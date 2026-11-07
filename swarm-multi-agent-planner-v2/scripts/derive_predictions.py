#!/usr/bin/env python3
"""Derive every selection rule's predictions from finished runs (no LLM calls).

    python scripts/derive_predictions.py --all                      # every run under results/runs
    python scripts/derive_predictions.py --run_dir results/runs/tot_aco/seed_0

Writes <run_dir>/derived/{predictions_<rule>.json, per_task.csv, selection_report.json}. Each
predictions_<rule>.json can be scored with the official evaluator:
    python scripts/evaluate.py --data_path <run_dir>/derived/predictions_<rule>.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarm_planner.derive import derive_run, find_run_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", action="append", default=[], help="A run directory (repeatable).")
    parser.add_argument("--all", action="store_true", help="Every run directory under --runs_root.")
    parser.add_argument("--runs_root", default=str(PROJECT_ROOT / "results" / "runs"))
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dir]
    if args.all:
        run_dirs += find_run_dirs(args.runs_root)
    if not run_dirs:
        parser.error("give --run_dir or --all")
    for run_dir in run_dirs:
        report = derive_run(run_dir)
        headline = report["headline_rule"]
        em = report["rules"][headline]["exact_match"]
        print(f"{run_dir}: {report['tasks']} tasks, headline rule '{headline}' exact match {em:.4f}")
        for rule, values in report["rules"].items():
            print(f"    {rule:<22} {values['exact_match']:.4f}  verified answers {values['verified_answer_rate']:.4f}")


if __name__ == "__main__":
    main()
