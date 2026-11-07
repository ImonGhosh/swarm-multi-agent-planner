#!/usr/bin/env python3
"""Analyse a path table: integrity, task groups, level effects, heuristic, failure modes.

Usage:
  python scripts/analyze_path_table.py --table results/path_table/dev_20 --out report.txt

Intervals are 95% bootstrap intervals over tasks. The task, not the path, is the unit that
counts: the 243 paths of a task are strongly correlated.
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

from swarm_planner.logging import read_events  # noqa: E402
from swarm_planner.simulate import BRANCH_IDS, LEVELS, load_path_table  # noqa: E402
from swarm_planner.tot.levels import LEVEL_CONFIGS  # noqa: E402

SLOT = re.compile(r"(mon|tues|wednes|thurs|fri)day,?\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})")


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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = Report(args.out)
    rng = np.random.default_rng(0)

    with gzip.open(args.table / "path_table.csv.gz", "rt", encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    with gzip.open(args.table / "nodes.jsonl.gz", "rt", encoding="utf-8") as h:
        nodes = [json.loads(line) for line in h]
    events = list(read_events(args.table / "events.jsonl.gz"))
    tables = load_path_table(args.table, args.data_path)
    tasks = [t.task_id for t in tables]
    n_tasks = len(tasks)

    # ---- 1. integrity ---------------------------------------------------------------------
    out("=== 1. Integrity ===")
    calls = [e for e in events if e["event"] == "llm_call"]
    paid = [c for c in calls if not c["hit"]]
    per_task_rows = collections.Counter(r["task_id"] for r in rows)
    per_task_reqs = collections.defaultdict(set)
    for n in nodes:
        per_task_reqs[n["task_id"]].add(n["req"])
    out(f"tasks {n_tasks}, path rows {len(rows)}, node rows {len(nodes)}, llm_call events {len(calls)}")
    out(f"paths per task {sorted(set(per_task_rows.values()))}, distinct requests per task "
        f"{sorted({len(v) for v in per_task_reqs.values()})}")
    out(f"correct implies score 1.0 and hard-valid: "
        f"{all(r['score'] == '1.0' and r['hard_valid'] == 'True' for r in rows if r['correct'] == 'True')}")
    out(f"task_error events: {sum(e['event'] == 'task_error' for e in events)}; "
        f"providers of paid calls: {dict(collections.Counter(c['prov'] for c in paid))}")
    finish = collections.Counter(n["fin"] for n in nodes)
    out(f"finish reasons: {dict(finish)} ({100 * finish.get('length', 0) / len(nodes):.2f}% stopped at max_tokens)")
    cats = collections.Counter(t.category for t in tables)
    out(f"categories: {dict(sorted(cats.items()))}")

    # ---- 2. task groups -------------------------------------------------------------------
    out("\n=== 2. Task groups (243 paths per task) ===")
    out("task  category  correct  hard-valid  score-1.0  distinct answers")
    answers = collections.defaultdict(set)
    for r in rows:
        answers[r["task_id"]].add(r["answer"])
    for t in tables:
        out(f"{t.task_id[-3:]:>4} {t.category:>8} {int(t.correct.sum()):>6}/243 {int(t.valid.sum()):>8} "
            f"{int((t.score == 1.0).sum()):>10}   {len(answers[t.task_id])}")
    none = [t for t in tables if not t.correct.any()]
    allc = [t for t in tables if t.correct.all()]
    some = n_tasks - len(none) - len(allc)
    out(f"tasks with no correct path: {len(none)}, every path correct: {len(allc)}, some correct: {some}")
    p = np.array([t.correct.mean() for t in tables])
    out(f"mean path accuracy {p.mean():.3f} (median {np.median(p):.3f}); pooled "
        f"{int(sum(t.correct.sum() for t in tables))}/{243 * n_tasks}")
    out("failure mode of the tasks with no correct path:")
    for t in none:
        if (t.score == 1.0).any():
            kind = "a path scores 1.0 but is not the gold answer (several valid answers, F11)"
        elif t.valid.any():
            kind = f"{int(t.valid.sum())} paths are hard-valid but violate a preference or the earliest rule"
        else:
            kind = "no path produces even a hard-valid answer"
        out(f"  {t.task_id[-3:]:>4} {t.category:>6}: {kind}")
    by_cat = collections.defaultdict(list)
    for t in tables:
        by_cat[t.category].append(f"{t.correct.mean():.2f}")
    out("path accuracy by category (two tasks each): " + "; ".join(f"{k} {v}" for k, v in by_cat.items()))

    # ---- 3. level effects -----------------------------------------------------------------
    out("\n=== 3. Level and branch effects ===")
    idx = np.array(list(np.ndindex(*[len(b) for b in BRANCH_IDS])))
    draws = 4000
    boot = rng.integers(0, n_tasks, size=(draws, n_tasks))
    out(f"overall task-averaged path accuracy {p.mean():.3f}; branch accuracy and its difference from the level mean "
        "(points, 95% interval over tasks; * = interval excludes 0)")
    flat = {t.task_id: t.correct[tuple(idx.T)] for t in tables}
    spreads = {}
    for lv in range(LEVELS):
        per_branch = []
        for b in range(len(BRANCH_IDS[lv])):
            mask = idx[:, lv] == b
            per_branch.append(np.array([flat[t][mask].mean() for t in tasks]))
        level_mean = np.mean(per_branch, axis=0)
        for b, vals in enumerate(per_branch):
            d = vals - level_mean
            lo, hi = np.percentile(d[boot].mean(axis=1), [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else ""
            out(f"  L{lv + 1} {BRANCH_IDS[lv][b]:<28} acc {vals.mean():.3f}  diff {100 * d.mean():+5.1f} "
                f"[{100 * lo:+5.1f}, {100 * hi:+5.1f}] {star}")
        spreads[lv] = max(v.mean() for v in per_branch) - min(v.mean() for v in per_branch)
    out("permutation test (branch labels shuffled within each task, 2000 times):")
    for lv in range(LEVELS):
        labels = {t: idx[:, lv] for t in tasks}

        def spread(lab):
            rates = [np.mean([flat[t][lab[t] == b].mean() for t in tasks]) for b in range(len(BRANCH_IDS[lv]))]
            return max(rates) - min(rates)

        null = np.array([spread({t: rng.permutation(labels[t]) for t in tasks}) for _ in range(2000)])
        pval = (np.sum(null >= spreads[lv]) + 1) / (len(null) + 1)
        out(f"  L{lv + 1}: spread {spreads[lv]:.3f}, chance 95th percentile {np.percentile(null, 95):.3f}, p = {pval:.3f}")

    out("\nvariance of 'correct': share explained by the task alone / by the branch labels within a task")
    total = p.mean() * (1 - p.mean())
    within = np.mean(p * (1 - p))
    out(f"  task alone explains {100 * (1 - within / total):.0f}% of the variance")
    for interact in (False, True):
        ss_res = ss_tot = 0.0
        for t in tables:
            y = flat[t.task_id]
            if y.var() == 0:
                continue
            cols = [(idx[:, lv] == b).astype(float) for lv in range(LEVELS) for b in range(len(BRANCH_IDS[lv]) - 1)]
            if interact:
                cols += [((idx[:, 0] == a) & (idx[:, 1] == b)).astype(float)
                         for a in range(3) for b in range(3) if (a, b) != (2, 2)]
            x = np.column_stack([np.ones(len(y))] + cols)
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            ss_res += ((y - x @ beta) ** 2).sum()
            ss_tot += ((y - y.mean()) ** 2).sum()
        label = "main effects + level-1 x level-2" if interact else "main effects of the 5 levels"
        out(f"  {label}: within-task R^2 = {1 - ss_res / ss_tot:.3f} (tasks with some correct path; descriptive, 25 parameters)")

    # ---- 4. heuristic ---------------------------------------------------------------------
    out("\n=== 4. Heuristic eta (tau = 1, beta = 2, as at the start of an ACO run) ===")
    eps = 0.001
    eta_acc, uni_acc = [], []
    informative = {lv: [] for lv in range(LEVELS)}
    for t in tables:
        w = (1 + eps) * (t.eta + eps) ** 2
        probs = w / w.sum(axis=1, keepdims=True)
        path_p = np.prod([probs[lv][idx[:, lv]] for lv in range(LEVELS)], axis=0)
        eta_acc.append(float((path_p * flat[t.task_id]).sum()))
        uni_acc.append(float(flat[t.task_id].mean()))
        for lv in range(LEVELS):
            emp = np.array([flat[t.task_id][idx[:, lv] == b].mean() for b in range(3)])
            if np.ptp(emp) > 1e-9 and np.ptp(probs[lv]) > 1e-9:
                informative[lv].append((probs[lv], emp))
    eta_acc, uni_acc = np.array(eta_acc), np.array(uni_acc)
    d = eta_acc - uni_acc
    lo, hi = np.percentile(d[boot].mean(axis=1), [2.5, 97.5])
    out(f"accuracy of one agent: uniform {100 * uni_acc.mean():.2f}%, eta-guided {100 * eta_acc.mean():.2f}% "
        f"(difference {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] points)")
    for lv in range(LEVELS):
        pairs = informative[lv]
        if not pairs:
            out(f"  L{lv + 1}: no task where both eta and the branches differ")
            continue
        gain = np.mean([(pr * emp).sum() - emp.mean() for pr, emp in pairs])
        best = np.mean([emp[pr.argmax()] >= emp.max() - 1e-9 for pr, emp in pairs])
        worst = np.mean([emp[pr.argmax()] <= emp.min() + 1e-9 for pr, emp in pairs])
        out(f"  L{lv + 1}: accuracy change from eta at this level {100 * gain:+.2f} points; in the {len(pairs)} tasks "
            f"where branches differ, eta's favourite is the best branch in {100 * best:.0f}% and the worst in {100 * worst:.0f}%")

    # ---- 5. what the intermediate answers look like -------------------------------------
    out("\n=== 5. The intermediate answers (nodes) ===")
    thought = {(cfg.level, i): re.sub(r"[^a-z0-9: ]+", " ", b.thought_prompt.lower()).strip()
               for cfg in LEVEL_CONFIGS for i, b in enumerate(cfg.branches)}

    def echo(text, level, branch):
        t = re.sub(r"[^a-z0-9: ]+", " ", text.lower()).strip()
        t = re.sub(r"^intermediate result( for (the )?(latest )?thought)?", "", t).strip()
        return len(t) < 5 or t in thought[(level, branch)] or (
            thought[(level, branch)] in t and len(t) < len(thought[(level, branch)]) + 60)

    stats = collections.defaultdict(collections.Counter)
    inter = {(n["task_id"], n["node"]): n for n in nodes if n["step"] != "final"}
    for (task, node), n in inter.items():
        level = int(n["step"][1])
        s = stats[level]
        s["n"] += 1
        s["echo"] += echo(n["text"], level, int(node.split("-")[-1]))
        s["premature"] += bool(re.search(r"proposed time|best slot|best time", n["text"].lower()))
        s["slot"] += bool(SLOT.search(n["text"].lower()))
        s["tokens"] += n["out"]
    out("level   calls  repeats-the-instruction  announces-a-final-answer  states-a-slot  mean output tokens")
    for lv in range(1, LEVELS + 1):
        s = stats[lv]
        out(f"L{lv}  {s['n']:>7}  {100 * s['echo'] / s['n']:>10.1f}%  {100 * s['premature'] / s['n']:>18.1f}%  "
            f"{100 * s['slot'] / s['n']:>11.1f}%  {s['tokens'] / s['n']:>12.0f}")

    def last_slot(text):
        m = SLOT.findall(text.lower())
        return m[-1] if m else None

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    out("slot stated at level L (paths where it states one): share equal to the final answer / to gold")
    for lv in (3, 4, 5):
        stated = same = gold_ok = 0
        for r in rows:
            n = inter[(r["task_id"], "-".join(r["path"].split("-")[:lv]))]
            s = last_slot(n["text"])
            if s:
                stated += 1
                same += s == last_slot(r["answer"])
                gold_ok += s == last_slot(data[r["task_id"]]["golden_plan"])
        out(f"  L{lv}: states a slot in {100 * stated / len(rows):.0f}% of paths; equals the final answer in "
            f"{100 * same / stated:.0f}%, equals gold in {100 * gold_ok / stated:.1f}%")
    out("(the finalize call is separate from level 5 and changes the slot stated at level 5 in the remaining paths)")

    out.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
