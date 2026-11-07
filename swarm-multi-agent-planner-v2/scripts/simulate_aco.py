#!/usr/bin/env python3
"""Replay ToT-only and ToT+ACO settings offline from a path table (no LLM calls).

Prints a report with: the ACO variants compared with ToT-only, accuracy against budget, the
n x T split, early stopping, and how far the pheromone moves toward the best level-1 branch.
Differences carry a 95% bootstrap interval over tasks. Run it on the path table written by
scripts/enumerate_paths.py.

Usage:
  python scripts/simulate_aco.py --table results/path_table/dev_20 --reps 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swarm_planner.simulate import (  # noqa: E402
    DEPOSIT_RULES,
    LEVELS,
    bootstrap_ci,
    load_path_table,
    simulate,
    simulate_task,
)

N_AGENTS, N_ITERATIONS = 8, 6


class Report:
    def __init__(self, path: Path | None) -> None:
        self.lines: list[str] = []
        self.path = path

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:5.1f}"


def diff_text(d: np.ndarray) -> str:
    lo, hi = bootstrap_ci(d)
    return f"{100 * d.mean():+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, default=PROJECT_ROOT / "results" / "path_table" / "dev_20")
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "dev_100.json")
    parser.add_argument("--reps", type=int, default=400, help="Simulated searches per task and setting.")
    parser.add_argument("--out", type=Path, default=None, help="Also write the report to this file.")
    args = parser.parse_args()

    tables = load_path_table(args.table, args.data_path)
    out = Report(args.out)
    solvable = [t for t in tables if t.correct.any()]
    out(f"Path table: {len(tables)} tasks, {len(solvable)} with at least one correct path "
        f"(pass@243 = {len(solvable)}/{len(tables)}). {args.reps} simulated searches per task and setting.")
    out(f"Budget: {N_AGENTS} agents x {N_ITERATIONS} iterations = {N_AGENTS * N_ITERATIONS} traversals unless stated.")
    out("Accuracy is exact match with gold, averaged over all tasks. 'last' selects the best answer of the last")
    out("iteration (what the v1 solver does), 'all' the best answer over all iterations.")

    # ---- 1. variants -------------------------------------------------------------------
    out("\n=== 1. ACO variants against ToT-only (equal budget) ===")
    settings = {
        "ToT-only (uniform)": dict(alpha=0.0, beta=0.0, deposit=DEPOSIT_RULES["v1_score"]),
        "ACO v1 rule (a=1, b=2, deposit=score)": dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["v1_score"]),
        "eta only (a=0, b=2)": dict(alpha=0.0, beta=2.0, deposit=DEPOSIT_RULES["v1_score"]),
        "pheromone only (a=1, b=0, v1 deposit)": dict(alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["v1_score"]),
        "ACO, valid-only deposit": dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["valid_only"]),
        "ACO, top-only deposit": dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["top_only"]),
        "ACO, best-of-iteration deposit": dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["best_of_iteration"]),
        "ACO, (score - mean) deposit": dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["score_minus_mean"]),
        "pheromone only, top-only deposit": dict(alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"]),
    }
    base = None
    out(f"{'setting':<40} {'last':>5} {'all':>5} {'valid(last)':>11} | difference vs ToT-only, last | all")
    for name, kw in settings.items():
        r = simulate(tables, reps=args.reps, n_agents=N_AGENTS, n_iterations=N_ITERATIONS, **kw)
        line = f"{name:<40} {pct(r['last_correct'].mean())} {pct(r['all_correct'].mean())} {pct(r['last_top'].mean()):>11}"
        if base is None:
            base = r
        else:
            line += f" | {diff_text(r['last_correct'] - base['last_correct'])} | {diff_text(r['all_correct'] - base['all_correct'])}"
        out(line)
    out(f"(oracle: a task counts as solvable if any of its 243 paths is correct: {pct(len(solvable) / len(tables))}%)")

    # ---- 2. budget ---------------------------------------------------------------------
    out("\n=== 2. Accuracy against budget: 8 agents x T iterations, selection = best over all iterations ===")
    out(f"{'traversals':>10} {'ToT-only':>9} {'ACO v1':>7} {'ACO top-only/pher.':>19}")
    for t_iter in (1, 2, 3, 4, 6):
        row = []
        for kw in (
            dict(alpha=0.0, beta=0.0, deposit=DEPOSIT_RULES["v1_score"]),
            dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["v1_score"]),
            dict(alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"]),
        ):
            r = simulate(tables, reps=max(100, args.reps // 2), n_agents=N_AGENTS, n_iterations=t_iter, **kw)
            row.append(pct(r["all_correct"].mean()))
        out(f"{N_AGENTS * t_iter:>10} {row[0]:>9} {row[1]:>7} {row[2]:>19}")

    # ---- 3. n x T split ----------------------------------------------------------------
    out("\n=== 3. How the 48 traversals are split into agents x iterations ===")
    out(f"{'n x T':>8} | ToT-only last   all | ACO top-only/pheromone last   all")
    for n, t_iter in ((48, 1), (24, 2), (12, 4), (8, 6), (4, 12), (2, 24), (1, 48)):
        a = simulate(tables, reps=max(60, args.reps // 5), n_agents=n, n_iterations=t_iter,
                     alpha=0.0, beta=0.0)
        b = simulate(tables, reps=max(60, args.reps // 5), n_agents=n, n_iterations=t_iter,
                     alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"])
        out(f"{n:>3} x {t_iter:<2} |       {pct(a['last_correct'].mean())} {pct(a['all_correct'].mean())} |"
            f"                    {pct(b['last_correct'].mean())} {pct(b['all_correct'].mean())}")

    # ---- 4. early stopping -------------------------------------------------------------
    out("\n=== 4. Early stopping after the first iteration with a verifier score-1.0 answer (ToT-only) ===")
    full = simulate(tables, reps=args.reps, n_agents=N_AGENTS, n_iterations=N_ITERATIONS)
    early = simulate(tables, reps=args.reps, n_agents=N_AGENTS, n_iterations=N_ITERATIONS, early_stop=True)
    used = early["iterations_used"]
    solv = np.array([t.correct.any() for t in tables])
    out(f"accuracy (best over all): full budget {pct(full['all_correct'].mean())}%, early stopping "
        f"{pct(early['all_correct'].mean())}% (difference {diff_text(early['all_correct'] - full['all_correct'])})")
    out(f"mean iterations used: {used.mean():.2f} of {N_ITERATIONS} = {100 * used.mean() / N_ITERATIONS:.0f}% of the budget "
        f"(solvable tasks {used[solv].mean():.2f}, unsolvable tasks {used[~solv].mean():.2f})")

    # ---- 5. pheromone evidence -----------------------------------------------------------
    out("\n=== 5. Does the pheromone move toward the best level-1 branch? ===")
    rng = np.random.default_rng(3)
    rules = {
        "ACO v1 (deposit = score)": dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["v1_score"]),
        "ACO, top-only deposit": dict(alpha=1.0, beta=2.0, deposit=DEPOSIT_RULES["top_only"]),
        "pheromone only, top-only deposit": dict(alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"]),
    }
    informative = [t for t in tables if np.ptp([t.correct[b].mean() for b in range(3)]) > 1e-9]
    out(f"tasks where level-1 branches differ in accuracy: {len(informative)} of {len(tables)}")
    for name, kw in rules.items():
        first, last = [], []
        for table in informative:
            best = int(np.argmax([table.correct[b].mean() for b in range(3)]))
            runs = [simulate_task(table, n_agents=N_AGENTS, n_iterations=N_ITERATIONS, rng=rng, **kw)
                    for _ in range(min(args.reps, 100))]
            first.append(np.mean([r["level1_probabilities"][0][best] for r in runs]))
            last.append(np.mean([r["level1_probabilities"][-1][best] for r in runs]))
        out(f"{name:<36} P(best level-1 branch): iteration 1 {np.mean(first):.3f} -> iteration {N_ITERATIONS} "
            f"{np.mean(last):.3f} (uniform = 0.333)")

    # ---- 6. what ACO changes in a full-budget run -------------------------------------------
    out("\n=== 6. The frozen v2 method (no eta, top-only deposit) against ToT-only, full budget, no early stopping ===")
    out("Same budget (8 x 6), the answer is the best trace over all iterations. 'verified share' = share of a")
    out("iteration's traces that the verifier scores 1.0.")
    tot = simulate(tables, reps=args.reps, seed=11, n_agents=N_AGENTS, n_iterations=N_ITERATIONS)
    v2 = simulate(tables, reps=args.reps, seed=11, n_agents=N_AGENTS, n_iterations=N_ITERATIONS,
                  alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"])
    rows6 = [
        ("verified share, iteration 1", "top_share_first", True),
        ("verified share, iteration 6", "top_share_last", True),
        ("verified share, mean over the run", "top_share_all", True),
        ("accuracy, best over all iterations", "all_correct", True),
        ("accuracy, best of the last 3 iterations", "last3_correct", True),
        ("accuracy, best of the last 2 iterations", "last2_correct", True),
        ("accuracy, best of the last iteration", "last_correct", True),
        ("accuracy, plain majority vote over the 48 traces (no verifier)", "majority_correct", True),
        ("accuracy, verified vote, all iterations", "vote_verified_all_correct", True),
        ("accuracy, verified vote, last 3 iterations", "vote_verified_last3_correct", True),
        ("accuracy, verified vote, last 2 iterations", "vote_verified_last2_correct", True),
        ("accuracy, hard-valid vote, all iterations", "vote_valid_all_correct", True),
        ("accuracy, hard-valid vote, last 3 iterations", "vote_valid_last3_correct", True),
        ("accuracy, hard-valid vote, last 2 iterations", "vote_valid_last2_correct", True),
    ]
    out(f"{'measure':<58} {'ToT-only':>9} {'ACO v2':>8}   ACO v2 minus ToT-only")
    for label, key, is_pct in rows6:
        d = v2[key] - tot[key]
        lo, hi = bootstrap_ci(d) if np.any(d) else (0.0, 0.0)
        out(f"{label:<58} {pct(tot[key].mean()):>9} {pct(v2[key].mean()):>8}   {diff_text(d)}")
    first = []
    last = []
    for table in informative:
        best = int(np.argmax([table.correct[b].mean() for b in range(3)]))
        runs = [simulate_task(table, n_agents=N_AGENTS, n_iterations=N_ITERATIONS, rng=rng, alpha=1.0, beta=0.0,
                              deposit=DEPOSIT_RULES["top_only"]) for _ in range(min(args.reps, 100))]
        first.append(np.mean([r["level1_probabilities"][0][best] for r in runs]))
        last.append(np.mean([r["level1_probabilities"][-1][best] for r in runs]))
    out(f"probability of the best level-1 branch (tasks where it is defined): iteration 1 {np.mean(first):.3f} -> "
        f"iteration {N_ITERATIONS} {np.mean(last):.3f} (ToT-only stays at 0.333)")

    # ---- 7. cost of the Phase 4 runs ----------------------------------------------------------
    out("\n=== 7. Estimated cost of the Phase 4 ToT-based runs (full budget, shared cache) ===")
    import gzip as _gzip
    import json as _json

    price_in, price_out = 0.05e-6, 0.10e-6  # DeepInfra, google/gemma-3-4b-it, USD per token
    cost = {}
    with _gzip.open(args.table / "nodes.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            n = _json.loads(line)
            cost[(n["task_id"], n["step"], n["node"])] = n["in"] * price_in + n["out"] * price_out
    rng7 = np.random.default_rng(17)

    def node_keys(table, paths):
        keys = set()
        for p in paths:
            for lv in range(1, LEVELS + 1):
                keys.add((table.task_id, f"L{lv}", "-".join(map(str, p[:lv]))))
            keys.add((table.task_id, "final", "-".join(map(str, p))))
        return keys

    full_table_cost = sum(cost.values()) / len(tables)
    reps7 = 40
    per_run, six_runs = [], []
    for table in tables:
        singles, unions = [], []
        for _ in range(reps7):
            seen_all = set()
            for r in range(6):  # 3 ToT-only seeds and 3 ACO v2 seeds
                kw = dict(alpha=0.0, beta=0.0) if r < 3 else dict(alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"])
                run = simulate_task(table, n_agents=N_AGENTS, n_iterations=N_ITERATIONS, rng=rng7, **kw)
                keys = node_keys(table, run["paths"])
                if r == 0:
                    singles.append(sum(cost[k] for k in keys))
                seen_all |= keys
            unions.append(sum(cost[k] for k in seen_all))
        per_run.append(np.mean(singles))
        six_runs.append(np.mean(unions))
    tasks_test = 400
    out(f"average cost of all 606 requests of a task (every path): ${full_table_cost:.4f}")
    out(f"one full-budget ToT-only run: ${np.mean(per_run):.4f} per task, ${np.mean(per_run) * tasks_test:.1f} for {tasks_test} tasks")
    out(f"six runs sharing the cache (3 ToT-only seeds + 3 ACO v2 seeds): ${np.mean(six_runs):.4f} per task, "
        f"${np.mean(six_runs) * tasks_test:.1f} for {tasks_test} tasks")
    out("(extrapolated from the 20 dev tasks, which have the same category mix as the test set)")

    out.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
