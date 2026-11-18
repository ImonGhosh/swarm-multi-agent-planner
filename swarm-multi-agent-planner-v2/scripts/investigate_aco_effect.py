"""Why does ToT+ACO beat ToT-only under best-of-all selection? (no LLM calls)

Phase 3's simulation predicted an exact tie under best-of-all selection: until the first verified
answer of a task nothing is deposited, the pheromone stays uniform, and a search with the same
seed draws the same paths in both modes. Phase 4 measured +2.0 pp for ACO. This script checks the
logs for the reason:

  1. Are the two modes' traces identical until ACO's first deposit (same paths, same answers)?
  2. Where do they first differ, and how does that relate to the first verified trace?
  3. In tasks where only one mode ever finds a verified trace, what happened?

Output: results/aggregate/analysis/best_of_all_investigation.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm_planner.analysis_data import Run, TaskLog, load_runs  # noqa: E402


def trace_key(t):
    return (t.it, t.agent)


def compare_task(a: TaskLog, b: TaskLog) -> dict:
    """a = ToT-only, b = ToT+ACO. Traces are matched by (iteration, agent)."""
    ta = {trace_key(t): t for t in a.traces}
    tb = {trace_key(t): t for t in b.traces}
    first_path_diff = None
    first_answer_diff_same_path = None
    for it in range(a.iterations):
        for agent in range(8):
            x, y = ta[(it, agent)], tb[(it, agent)]
            if x.path != y.path and first_path_diff is None:
                first_path_diff = it
            if x.path == y.path and x.answer != y.answer and first_answer_diff_same_path is None:
                first_answer_diff_same_path = it
    return {
        "first_path_diff": first_path_diff,
        "first_answer_diff_same_path": first_answer_diff_same_path,
        "a_first_verified": a.first_verified_iteration(),
        "b_first_verified": b.first_verified_iteration(),
        "a_any_verified": a.first_verified_iteration() is not None,
        "b_any_verified": b.first_verified_iteration() is not None,
        "a_any_exact": a.first_exact_iteration() is not None,
        "b_any_exact": b.first_exact_iteration() is not None,
    }


def binom_two_sided(k1: int, k2: int) -> float:
    from math import comb
    n, k = k1 + k2, min(k1, k2)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def extra_diagnostics(tot, aco, emit) -> None:
    import numpy as np

    emit("=== DIAGNOSTICS: can the pheromone matter before a task's first verified trace? ===")
    dev = 0.0
    checked = 0
    for run in aco:
        for t in run.tasks.values():
            for it in range(t.iterations):
                if t.first_verified_iteration() is not None and it > t.first_verified_iteration():
                    break
                if any(x.verified for x in t.at(it)):
                    break
                before, after = np.array(t.tau_before[it]), np.array(t.tau_after[it])
                dev = max(dev, float(np.abs(after - 0.9 * before).max()))
                checked += 1
    emit(f"ACO, iterations before the first verified trace: {checked} checked; largest deviation of tau_after "
         f"from pure evaporation 0.9*tau_before = {dev:.2e}  (so the probabilities are exactly uniform there)")

    emit("\nFirst-success rate per iteration among tasks without a verified trace so far (3 seeds pooled):")
    emit("  it   ToT-only (at risk)    ToT+ACO (at risk)")
    for it in range(6):
        cells = []
        for runs in (tot, aco):
            at_risk = succ = 0
            for r in runs:
                for t in r.tasks.values():
                    if not any(x.verified for x in t.traces if x.it < it):
                        at_risk += 1
                        succ += any(x.verified for x in t.at(it))
            cells.append(f"{succ / at_risk:6.3f} ({at_risk:4d})")
        emit(f"  {it}   {cells[0]}        {cells[1]}")

    emit("\nCumulative share of tasks with a verified trace after iteration k (mean over 3 seeds):")
    for it in range(6):
        cells = [
            float(np.mean([np.mean([any(x.verified for x in t.traces if x.it <= it) for t in r.tasks.values()]) for r in runs]))
            for runs in (tot, aco)
        ]
        emit(f"  after iteration {it + 1}: ToT-only {cells[0]:.4f}, ToT+ACO {cells[1]:.4f}, gap {100 * (cells[1] - cells[0]):+.2f} pp")
    emit("  (the gap exists after iteration 1, before any pheromone, and does not grow afterwards)")

    emit("\nIteration-0 branch shares (no pheromone is possible yet), mean over levels and seeds, per branch index:")
    for name, runs in (("ToT-only", tot), ("ToT+ACO", aco)):
        c = np.zeros((5, 3))
        for r in runs:
            for t in r.tasks.values():
                for x in t.at(0):
                    for lvl, b in enumerate(x.branch_idx):
                        c[lvl, b] += 1
        emit(f"  {name}: {np.round((c / c.sum(1, keepdims=True)).mean(0), 4).tolist()}   (n = {int(c[0].sum())} traces)")
    emit("\nShare of iteration-0 traces that are verified:")
    for name, runs in (("ToT-only", tot), ("ToT+ACO", aco)):
        vals = [np.mean([x.verified for t in r.tasks.values() for x in t.at(0)]) for r in runs]
        emit(f"  {name}: " + ", ".join(f"{v:.4f}" for v in vals))

    emit("\nSeed-to-seed discordance of 'any verified trace' inside one mode (independent draws, same distribution):")
    def anyv(r):
        return np.array([t.first_verified_iteration() is not None for t in r.tasks.values()])
    for name, runs in (("ToT-only", tot), ("ToT+ACO", aco)):
        for i, j in ((0, 1), (0, 2), (1, 2)):
            a, b = anyv(runs[i]), anyv(runs[j])
            emit(f"  {name} seeds {i} vs {j}: only first {int((a & ~b).sum())}, only second {int((b & ~a).sum())}")
    emit("\nWhy the two modes never share a path (even at iteration 0): ToT-only draws a branch with "
         "rng.integers(3) and ToT+ACO with rng.random() (tot/graph.py), from the same per-call seed stream. "
         "The draws are therefore independent, not coupled, so the paired McNemar test pairs tasks, not draws.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--out", default=str(ROOT / "results" / "aggregate" / "analysis" / "best_of_all_investigation.txt"))
    args = parser.parse_args()

    tot = load_runs(args.runs_root, "tot_only")
    aco = load_runs(args.runs_root, "tot_aco")
    lines = []
    emit = lines.append

    emit("WHY DOES ACO BEAT TOT-ONLY UNDER BEST-OF-ALL?  (ToT-only = A, ToT+ACO = B; matched by task, iteration, agent)\n")
    totals = Counter()
    for a_run, b_run in zip(tot, aco):
        emit(f"=== seed {a_run.seed} ===")
        rows = [compare_task(a_run.tasks[t], b_run.tasks[t]) for t in a_run.tasks]
        n = len(rows)
        same_it0 = sum(1 for r in rows if r["first_path_diff"] != 0)
        emit(f"tasks {n}; iteration-0 paths identical in A and B for {same_it0} tasks")
        emit(f"tasks with any path difference: {sum(r['first_path_diff'] is not None for r in rows)}")
        emit(f"tasks where an identical path got a different answer (LLM answer differs): "
             f"{sum(r['first_answer_diff_same_path'] is not None for r in rows)}")
        # when do paths diverge relative to the first verified trace of either mode
        div = Counter()
        for r in rows:
            if r["first_path_diff"] is None:
                continue
            first_ver_a = r["a_first_verified"]
            div["diverge_after_first_verified_of_A" if first_ver_a is not None and r["first_path_diff"] > first_ver_a
                else "diverge_at_or_before_first_verified_of_A" if first_ver_a is not None
                else "diverge_but_A_never_verified"] += 1
        emit(f"divergence timing (tasks): {dict(div)}")
        both = sum(r["a_any_verified"] and r["b_any_verified"] for r in rows)
        only_a = sum(r["a_any_verified"] and not r["b_any_verified"] for r in rows)
        only_b = sum(r["b_any_verified"] and not r["a_any_verified"] for r in rows)
        emit(f"any verified trace in 48: both {both}, only A (ToT-only) {only_a}, only B (ACO) {only_b}")
        ex_a = sum(r["a_any_exact"] for r in rows)
        ex_b = sum(r["b_any_exact"] for r in rows)
        emit(f"any exact-match trace in 48 (oracle pass@48): A {ex_a / n:.4f}, B {ex_b / n:.4f}")
        totals["only_a"] += only_a
        totals["only_b"] += only_b
        # for the 'only B' tasks: was the first verified trace in B before any divergence?
        explained = Counter()
        for r in rows:
            if r["b_any_verified"] and not r["a_any_verified"]:
                explained["B's first verified trace comes BEFORE the first path difference"
                          if r["first_path_diff"] is None or r["b_first_verified"] < r["first_path_diff"]
                          else "B's first verified trace comes at/after the first path difference"] += 1
        emit(f"'only B' tasks: {dict(explained)}")
        emit("")
    emit(f"all seeds: tasks verified only by ToT-only {totals['only_a']}, only by ACO {totals['only_b']}")
    emit(f"  sign test on these discordant tasks (null: equal chance): p = "
         f"{binom_two_sided(totals['only_a'], totals['only_b']):.4f}\n")
    extra_diagnostics(tot, aco, emit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
