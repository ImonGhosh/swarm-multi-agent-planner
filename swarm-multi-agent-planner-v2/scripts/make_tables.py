"""Tables behind the ACO-effect figures and the cost table (no LLM calls, no plotting library).

Reads the finished runs under results/runs/ and writes CSV files to results/aggregate/analysis/tables/:

  per_iteration.csv          search statistics per run and iteration (verified share, best score, ...)
  verified_share_difference.csv   ACO minus ToT-only verified share per iteration, bootstrap CI over tasks
  first_success.csv          per task: iteration of the first verified / exact trace, distinct paths
  pheromone.csv              ACO colony probability per level, branch and iteration (mean over tasks)
  pheromone_vs_quality.csv   does the pheromone point at the branches that really work (ToT-only reference)
  cost.csv                   LLM calls, tokens, cost and latency per run (incl. the sampling control)
  candidate_diversity.csv    per run: verified share per candidate, tasks with a verified candidate, distinct answers per task

Run:  .venv\\Scripts\\python.exe scripts\\make_tables.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm_planner.analysis_data import load_runs  # noqa: E402
from swarm_planner.analysis_metrics import (  # noqa: E402
    agreement_rows,
    cost_rows_from_events,
    diversity_row_from_samples,
    diversity_rows_from_runs,
    first_success_rows,
    per_iteration_rows,
    pheromone_rows,
    share_difference_rows,
)


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "aggregate" / "analysis" / "tables"))
    args = parser.parse_args()
    out = Path(args.out_dir)

    print("loading runs ...")
    tot = load_runs(args.runs_root, "tot_only")
    aco = load_runs(args.runs_root, "tot_aco")

    write_csv(out / "per_iteration.csv", per_iteration_rows(tot) + per_iteration_rows(aco))
    write_csv(out / "verified_share_difference.csv", share_difference_rows(tot, aco))
    write_csv(out / "first_success.csv", first_success_rows(tot) + first_success_rows(aco))
    write_csv(out / "pheromone.csv", pheromone_rows(aco))
    write_csv(out / "pheromone_vs_quality.csv", agreement_rows(aco, tot))

    costs = []
    for name in ("baseline", "tot_only", "tot_aco", "sampling_control"):
        for seed in (0, 1, 2):
            costs.append(cost_rows_from_events(Path(args.runs_root) / name / f"seed_{seed}", name, seed))
    write_csv(out / "cost.csv", costs)

    # how many different candidates does each method produce per task, and how many are verified
    sampling_dir = Path(args.runs_root) / "sampling_control"
    diversity = diversity_rows_from_runs(tot) + diversity_rows_from_runs(aco)
    for seed in (0, 1, 2):
        diversity.append(diversity_row_from_samples(sampling_dir / f"seed_{seed}", "sampling_control", seed))
    write_csv(out / "candidate_diversity.csv", diversity)


if __name__ == "__main__":
    main()
