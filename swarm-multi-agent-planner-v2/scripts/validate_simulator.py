#!/usr/bin/env python3
"""Check the offline simulator against the real solver, replayed from the LLM cache.

The real solver (ToT-only and ToT+ACO v1) is run on the tasks of a path table for several
seeds. Every LLM answer is already in the cache, so this costs nothing. The API key is NOT
loaded: a cache miss would fail instead of spending money. The real accuracy is then compared
with the simulator's prediction for the same tasks.

Usage:
  python scripts/validate_simulator.py --table results/path_table/dev_20 --seeds 30
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)

from swarm_planner.simulate import load_path_table, simulate  # noqa: E402
from swarm_planner.solver import solve_dataset  # noqa: E402
from swarm_planner.verifier.parser import answers_match  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, default=PROJECT_ROOT / "results" / "path_table" / "dev_20")
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "dev_100.json")
    parser.add_argument("--cache_path", type=Path, default=PROJECT_ROOT / "results" / "cache" / "llm_cache.sqlite")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--reps", type=int, default=600, help="Simulated searches per task.")
    args = parser.parse_args()

    tables = load_path_table(args.table, args.data_path)
    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    # Tasks without any correct path give 0 in both worlds; the comparison uses the others.
    use = [t for t in tables if t.correct.any()]
    ids = [t.task_id for t in use]

    work = Path(tempfile.mkdtemp(prefix="validate_simulator_"))
    subset = work / "subset.json"
    subset.write_text(json.dumps({i: data[i] for i in ids}), encoding="utf-8")
    real = {mode: np.zeros((args.seeds, len(ids))) for mode in ("tot", "aco")}
    try:
        for mode, aco in (("tot", False), ("aco", True)):
            for seed in range(args.seeds):
                run_dir = work / f"{mode}_{seed}"
                summary = solve_dataset(
                    data_path=str(subset),
                    run_dir=str(run_dir),
                    model="google/gemma-3-4b-it",
                    seed=seed,
                    aco_enabled=aco,
                    task_concurrency=len(ids),
                    cache_path=str(args.cache_path),
                )
                if summary["llm"]["cache_hits"] != summary["llm"]["llm_calls"]:
                    raise RuntimeError("a request was not in the cache; the table does not cover this run")
                predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
                for j, task_id in enumerate(ids):
                    real[mode][seed, j] = answers_match(
                        predictions[task_id]["pred_5shot_pro"], data[task_id]["golden_plan"]
                    )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    sim = {
        "tot": simulate(use, reps=args.reps, n_agents=8, n_iterations=6)["last_correct"],
        "aco": simulate(use, reps=args.reps, n_agents=8, n_iterations=6, alpha=1.0, beta=2.0)["last_correct"],
    }
    print(f"\nSolvable tasks: {len(ids)}; {args.seeds} real seeds per setting; selection = best of the last iteration.")
    print("task  category  p(path) | ToT-only real  sim | ACO v1 real  sim")
    for j, table in enumerate(use):
        print(f"{table.task_id[-3:]:>4} {table.category:>8} {table.correct.mean():7.3f} |"
              f"       {real['tot'][:, j].mean():5.2f} {sim['tot'][j]:5.2f} |"
              f"       {real['aco'][:, j].mean():5.2f} {sim['aco'][j]:5.2f}")
    for name, key in (("ToT-only", "tot"), ("ACO v1", "aco")):
        per_task = real[key].mean(axis=0)
        se = np.sqrt((per_task * (1 - per_task)).mean() / (args.seeds * len(ids)))
        corr = np.corrcoef(per_task, sim[key])[0, 1]
        print(f"{name}: real {per_task.mean():.3f} vs simulated {sim[key].mean():.3f} "
              f"(difference {per_task.mean() - sim[key].mean():+.3f}, approx. SE {se:.3f}); per-task correlation {corr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
