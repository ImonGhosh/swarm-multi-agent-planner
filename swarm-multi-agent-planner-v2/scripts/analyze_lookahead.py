#!/usr/bin/env python3
"""Does running the verifier on INTERMEDIATE answers help the search? (offline, no LLM calls)

The verifier is exact and cheap, so it can also judge a slot that a level 3, 4 or 5 answer
already states. Three questions, answered from the path table's node texts:

  1. Does that verdict predict whether the path ends up correct?
  2. Pruning: if a path's partial answer is already invalid, stop it. Does this lose correct
     paths, and does it pay off at an equal number of LLM calls?
  3. Credit: does an early verified slot give the pheromone a useful learning signal sooner?

Only a node that states exactly ONE slot is judged. Usage:
  python scripts/analyze_lookahead.py --table results/path_table/dev_20 --out report.txt
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swarm_planner.simulate import (  # noqa: E402
    DEPOSIT_RULES,
    bootstrap_ci,
    load_path_table,
    simulate_task,
)
from swarm_planner.verifier.scorer import build_task_quality_scorer  # noqa: E402

SLOT = re.compile(r"(monday|tuesday|wednesday|thursday|friday),?\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})")
SHAPE = (3,) * 5


class Report:
    def __init__(self, path):
        self.lines, self.path = [], path

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def save(self):
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, default=PROJECT_ROOT / "results" / "path_table" / "dev_20")
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "dev_100.json")
    parser.add_argument("--reps", type=int, default=1500, help="Replications for the pruning simulation.")
    parser.add_argument("--aco_reps", type=int, default=200, help="Replications for the ACO credit test (slow).")
    parser.add_argument("--skip_aco", action="store_true", help="Skip part 3 (it takes a few minutes).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = Report(args.out)

    tables = load_path_table(args.table, args.data_path)
    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    with gzip.open(args.table / "path_table.csv.gz", "rt", encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    nodes = {}
    with gzip.open(args.table / "nodes.jsonl.gz", "rt", encoding="utf-8") as h:
        for line in h:
            n = json.loads(line)
            if n["step"] != "final":
                nodes[(n["task_id"], n["node"])] = n
    scorers = {t.task_id: build_task_quality_scorer(data[t.task_id]["prompt_0shot"]) for t in tables}

    def judge(task_id, node):
        slots = set(SLOT.findall(nodes[(task_id, node)]["text"].lower()))
        if len(slots) != 1:
            return None
        day, start, end = next(iter(slots))
        return scorers[task_id].score_prediction(f"Here is the proposed time: {day.capitalize()}, {start} - {end}")

    judged = {key: judge(*key) for key in nodes}
    corr = np.array([r["correct"] == "True" for r in rows])

    # ---- 1. predictive value ----------------------------------------------------------------
    out("=== 1. Does the verdict on a level-L answer predict the final result? ===")
    out("(a node is judged only if its text states exactly one slot)")
    out("level  judged (share of paths) | verdict    paths   final answer correct   final answer hard-valid")
    for level in (2, 3, 4, 5):
        verdicts = []
        for r in rows:
            j = judged[(r["task_id"], "-".join(r["path"].split("-")[:level]))]
            verdicts.append(None if j is None else ("top" if j.score == 1.0 else "valid" if j.hard_valid else "invalid"))
        share = np.mean([v is not None for v in verdicts])
        out(f"L{level}   {sum(v is not None for v in verdicts):>6} ({100 * share:4.1f}%)")
        for v in ("top", "valid", "invalid"):
            mask = np.array([x == v for x in verdicts])
            if mask.any():
                valid_final = np.mean([rows[i]["hard_valid"] == "True" for i in np.where(mask)[0]])
                out(f"                              | {v:<8} {int(mask.sum()):>6}   {100 * corr[mask].mean():>12.1f}%   {100 * valid_final:>18.1f}%")

    # ---- 2. pruning -------------------------------------------------------------------------
    out("\n=== 2. Pruning a path when a level states one hard-invalid slot ===")
    rules = {
        "prune at L3": (3, 3, "invalid"),
        "prune at L4 only": (4, 4, "invalid"),
        "prune at L3-L5 (first invalid)": (3, 5, "invalid"),
        "prune at L3 if not top-scoring": (3, 3, "not_top"),
    }

    def prune_array(table, first, last, mode):
        arr = np.zeros(SHAPE, dtype=int)  # 0 = never pruned, k = pruned after the call of level k
        for idx in np.ndindex(*SHAPE):
            for level in range(first, last + 1):
                j = judged[(table.task_id, "-".join(map(str, idx[:level])))]
                if j is not None and ((not j.hard_valid) if mode == "invalid" else j.score < 1.0):
                    arr[idx] = level
                    break
        return arr

    pruned = {name: [prune_array(t, *spec) for t in tables] for name, spec in rules.items()}
    out("rule                              share pruned   correct paths wrongly pruned   'prune' is right   calls saved per traversal")
    for name, arrs in pruned.items():
        flat_pruned = np.concatenate([a.ravel() > 0 for a in arrs])
        # rows are ordered task by task and path by path, like the arrays
        saved = np.concatenate([np.where(a.ravel() > 0, 6 - a.ravel(), 0) for a in arrs]).mean()
        out(f"{name:<32} {100 * flat_pruned.mean():>9.1f}%   {100 * (flat_pruned & corr).sum() / corr.sum():>14.1f}% "
            f"({int((flat_pruned & corr).sum())}/{int(corr.sum())})   {100 * (~corr[flat_pruned]).mean():>13.1f}%   {saved:>8.2f} of 6")
    out("tasks that would lose ALL correct paths under 'prune at L3-L5': " + str(sum(
        1 for t, a in zip(tables, pruned["prune at L3-L5 (first invalid)"])
        if t.correct.any() and not ((a == 0) & (t.correct > 0.5)).any())))

    def dense_rank(table):
        keys = {i: (table.score[i], table.valid[i], table.soft_ratio[i], -table.rank[i]) for i in np.ndindex(*SHAPE)}
        order = {k: r for r, k in enumerate(sorted(set(keys.values())))}
        arr = np.zeros(SHAPE, dtype=int)
        for i, k in keys.items():
            arr[i] = order[k]
        return arr

    ranks = [dense_rank(t) for t in tables]

    def run_budget(budget, arrs, early_stop=False, seed=0):
        rng = np.random.default_rng(seed)
        acc, used_calls = [], []
        for t, arr, rk in zip(tables, arrs, ranks):
            p, r_, correct = arr.ravel(), rk.ravel(), t.correct.ravel() > 0.5
            top = t.score.ravel() == 1.0
            cost = np.where(p > 0, p, 6)
            draws = rng.integers(0, 243, size=(args.reps, 300))
            inside = np.cumsum(cost[draws], axis=1) <= budget
            done = inside & (p[draws] == 0)
            if early_stop:
                first = np.where(done & top[draws], np.arange(300)[None, :], 10**6).min(axis=1)
                inside &= np.arange(300)[None, :] <= first[:, None]
                done &= inside
            best = np.where(done, r_[draws], -1).argmax(axis=1)
            acc.append((done.any(axis=1) & correct[draws[np.arange(args.reps), best]]).mean())
            used_calls.append(np.where(inside, cost[draws], 0).sum(axis=1).mean())
        return np.array(acc), np.array(used_calls)

    none = [np.zeros(SHAPE, dtype=int) for _ in tables]
    budgets = (72, 144, 216, 288, 432)
    out("\naccuracy (best over all completed traversals) at an equal budget of LLM calls (a full traversal = 6 calls; 288 = today's 48 traversals):")
    out(f"{'rule':<34}" + "".join(f"{b:>13}" for b in budgets))
    base = {b: run_budget(b, none)[0] for b in budgets}
    out(f"{'no pruning':<34}" + "".join(f"{100 * base[b].mean():>13.1f}" for b in budgets))
    for name, arrs in pruned.items():
        cells = []
        for b in budgets:
            a = run_budget(b, arrs)[0]
            cells.append(f"{100 * a.mean():5.1f} ({100 * (a - base[b]).mean():+.1f})")
        out(f"{name:<34}" + "".join(f"{c:>13}" for c in cells))
    out("difference at 288 calls, 95% interval over tasks:")
    for name, arrs in pruned.items():
        d = run_budget(288, arrs)[0] - base[288]
        lo, hi = bootstrap_ci(d)
        out(f"  {name:<34} {100 * d.mean():+.2f} points [{100 * lo:+.2f}, {100 * hi:+.2f}]")

    out("\nwith early stopping at the first completed top-scoring answer (cap 288 calls):")
    solvable = np.array([t.correct.any() for t in tables])
    for name, arrs in {"no pruning": none, **pruned}.items():
        a, u = run_budget(288, arrs, early_stop=True)
        out(f"  {name:<34} accuracy {100 * a.mean():5.1f}%  mean calls {u.mean():6.1f}  "
            f"(solvable tasks {u[solvable].mean():6.1f}, unsolvable {u[~solvable].mean():6.1f})")

    # ---- 3. credit for the pheromone ----------------------------------------------------------
    if args.skip_aco:
        out.save()
        return 0
    out("\n=== 3. An early verified slot as an extra pheromone signal (pheromone only, 8 agents) ===")
    for t in tables:
        t.l3top = np.zeros((3, 3, 3))
        for idx in np.ndindex(3, 3, 3):
            j = judged[(t.task_id, "-".join(map(str, idx)))]
            t.l3top[idx] = 1.0 if (j is not None and j.score == 1.0) else 0.0

    def deposit_final_or_l3(table, paths, scores):
        return np.array([1.0 if (table.score[p] == 1.0 or table.l3top[p[:3]] == 1.0) else 0.0 for p in paths])

    def trial(n_iter, deposit, alpha, seed=0):
        rng = np.random.default_rng(seed)
        all_c, p_best = [], []
        for t in tables:
            runs = [simulate_task(t, n_agents=8, n_iterations=n_iter, rng=rng, alpha=alpha, beta=0.0, deposit=deposit)
                    for _ in range(args.aco_reps)]
            all_c.append(np.mean([r["all_correct"] for r in runs]))
            level1 = [t.correct[b].mean() for b in range(3)]
            if np.ptp(level1) > 1e-9:
                p_best.append(np.mean([r["level1_probabilities"][-1][int(np.argmax(level1))] for r in runs]))
        return np.array(all_c), float(np.mean(p_best))

    out("best-of-all accuracy by iterations (8 agents each)   ToT-only | final top-only deposit | final-or-L3-top deposit")
    for n_iter in (1, 2, 3, 4, 6):
        a, _ = trial(n_iter, DEPOSIT_RULES["v1_score"], 0.0)
        b, pb = trial(n_iter, DEPOSIT_RULES["top_only"], 1.0)
        c, pc = trial(n_iter, deposit_final_or_l3, 1.0)
        d = c - b
        lo, hi = bootstrap_ci(d) if np.any(d) else (0.0, 0.0)
        extra = f"  P(best L1 branch): {pb:.3f} vs {pc:.3f}" if n_iter == 6 else ""
        out(f"  {n_iter} iterations: {100 * a.mean():5.1f} | {100 * b.mean():5.1f} | {100 * c.mean():5.1f} "
            f"(extra signal {100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]){extra}")
    out.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
