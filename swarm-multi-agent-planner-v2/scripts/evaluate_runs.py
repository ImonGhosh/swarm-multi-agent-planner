#!/usr/bin/env python3
"""Evaluation: derive every selection rule, then aggregate over seeds.

    python scripts/evaluate_runs.py                 # derives (if needed) and aggregates results/runs
    python scripts/evaluate_runs.py --no_derive     # reuse the existing derived/ folders

Reads  results/runs/<config>/seed_<s>/derived/per_task.csv  (written by derive_predictions.py)
Writes results/aggregate/main/
    table_main.csv       one row per (config, selection rule): mean exact match +- SD over seeds,
                         95% bootstrap CI over tasks, every seed's value, verified-answer rate
    per_category.csv     exact match per (config, rule, category), mean over seeds
    paired_tests.csv     paired comparisons per seed (McNemar, exact) and pooled over seeds
    report.txt           the same, readable

Exact match is the official Natural Plan metric (swarm_planner.metrics mirrors scripts/evaluate.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from swarm_planner.derive import derive_run, find_run_dirs
from swarm_planner.selection_rules import HEADLINE_RULE, SELECTION_RULES
from swarm_planner.stats import bootstrap_ci, discordant, mcnemar_exact, mean_sd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_ORDER = ["baseline", "tot_only", "tot_aco", "sampling_control", "verifier_solver"]
RULE_ORDER = {
    "baseline": ["single_call"],
    "tot_only": list(SELECTION_RULES),
    "tot_aco": list(SELECTION_RULES),
    "sampling_control": ["best_of_samples", "majority_vote"],
    "verifier_solver": ["verifier_best_slot"],
}

Run = Dict[str, Any]  # {"tasks": [...], "categories": [...], "em": {rule: array}, "ver": {rule: array}, "seed": int}


def load_runs(runs_root: Path) -> Dict[str, Dict[int, Run]]:
    """config name -> seed -> per-task arrays (tasks in file order)."""
    runs: Dict[str, Dict[int, Run]] = defaultdict(dict)
    for run_dir in find_run_dirs(runs_root):
        report_path = run_dir / "derived" / "selection_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        with (run_dir / "derived" / "per_task.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rules = list(report["rules"])
        runs[report["config_name"]][int(report.get("seed") or 0)] = {
            "tasks": [r["task_id"] for r in rows],
            "categories": [r["category"] for r in rows],
            "em": {rule: np.array([int(r[rule]) for r in rows], dtype=float) for rule in rules},
            "ver": {rule: np.array([int(r[f"{rule}.verified"]) for r in rows], dtype=float) for rule in rules},
            "report": report,
        }
    return runs


def _stack(seed_runs: Dict[int, Run], key: str, rule: str) -> Tuple[List[int], np.ndarray]:
    """(seeds, array of shape (seeds, tasks)); every seed must cover the same tasks."""
    seeds = sorted(seed_runs)
    first = seed_runs[seeds[0]]["tasks"]
    for s in seeds:
        if seed_runs[s]["tasks"] != first:
            raise ValueError(f"seed {s} covers different tasks than seed {seeds[0]}")
    return seeds, np.stack([seed_runs[s][key][rule] for s in seeds])


def main_table(runs: Dict[str, Dict[int, Run]]) -> List[Dict[str, Any]]:
    rows = []
    for config in CONFIG_ORDER:
        if config not in runs:
            continue
        for rule in RULE_ORDER[config]:
            if rule not in next(iter(runs[config].values()))["em"]:
                continue
            seeds, em = _stack(runs[config], "em", rule)
            _, ver = _stack(runs[config], "ver", rule)
            per_seed = em.mean(axis=1)
            mean, sd = mean_sd(per_seed)
            low, high = bootstrap_ci(em.mean(axis=0))
            rows.append(
                {
                    "config": config,
                    "rule": rule,
                    "headline": int(rule == HEADLINE_RULE or config in ("baseline", "verifier_solver")),
                    "seeds": len(seeds),
                    "tasks": em.shape[1],
                    "exact_match_mean": round(mean, 4),
                    "exact_match_sd": round(sd, 4),
                    "ci95_low": round(low, 4),
                    "ci95_high": round(high, 4),
                    "verified_answer_rate": round(float(ver.mean()), 4),
                    **{f"seed_{s}": round(float(v), 4) for s, v in zip(seeds, per_seed)},
                }
            )
    return rows


def category_table(runs: Dict[str, Dict[int, Run]]) -> List[Dict[str, Any]]:
    rows = []
    for config in CONFIG_ORDER:
        if config not in runs:
            continue
        first = next(iter(runs[config].values()))
        cats = np.array(first["categories"])
        for rule in RULE_ORDER[config]:
            if rule not in first["em"]:
                continue
            _, em = _stack(runs[config], "em", rule)
            for cat in sorted(set(cats)):
                rows.append(
                    {
                        "config": config,
                        "rule": rule,
                        "category": cat,
                        "tasks": int((cats == cat).sum()),
                        "exact_match_mean": round(float(em[:, cats == cat].mean()), 4),
                    }
                )
    return rows


def _comparisons(runs: Dict[str, Dict[int, Run]]) -> List[Tuple[str, Tuple[str, str], Tuple[str, str]]]:
    """(label, (config A, rule A), (config B, rule B)): A is the 'treatment', B the reference."""
    pairs: List[Tuple[str, Tuple[str, str], Tuple[str, str]]] = []
    if "tot_aco" in runs and "tot_only" in runs:
        for rule in SELECTION_RULES:
            pairs.append((f"ACO vs ToT-only [{rule}]", ("tot_aco", rule), ("tot_only", rule)))
    if "baseline" in runs:
        for config in ("tot_only", "tot_aco"):
            if config in runs:
                pairs.append((f"{config} vs baseline [{HEADLINE_RULE}]", (config, HEADLINE_RULE), ("baseline", "single_call")))
    if "sampling_control" in runs:
        for config in ("tot_only", "tot_aco"):
            if config in runs:
                for rule in ("best_of_all", HEADLINE_RULE):
                    pairs.append(
                        (f"{config} [{rule}] vs sampling control [best_of_samples]", (config, rule), ("sampling_control", "best_of_samples"))
                    )
    return pairs


def paired_table(runs: Dict[str, Dict[int, Run]]) -> List[Dict[str, Any]]:
    rows = []
    for label, (ca, ra), (cb, rb) in _comparisons(runs):
        seeds = sorted(set(runs[ca]) & set(runs[cb]))
        if not seeds:
            continue
        diffs = []
        for s in seeds:
            ta, tb = runs[ca][s]["tasks"], runs[cb][s]["tasks"]
            if ta != tb:
                raise ValueError(f"{label}: seed {s} runs cover different tasks")
            a, b = runs[ca][s]["em"][ra], runs[cb][s]["em"][rb]
            only_a, only_b = discordant(a, b)
            diffs.append(a - b)
            rows.append(
                {
                    "comparison": label,
                    "seed": s,
                    "A_exact_match": round(float(a.mean()), 4),
                    "B_exact_match": round(float(b.mean()), 4),
                    "difference_pp": round(100 * float(a.mean() - b.mean()), 2),
                    "only_A_correct": only_a,
                    "only_B_correct": only_b,
                    "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4),
                }
            )
        pooled = np.stack(diffs).mean(axis=0)  # per task, mean over seeds
        low, high = bootstrap_ci(pooled)
        rows.append(
            {
                "comparison": label,
                "seed": "pooled",
                "difference_pp": round(100 * float(pooled.mean()), 2),
                "diff_ci95_low_pp": round(100 * low, 2),
                "diff_ci95_high_pp": round(100 * high, 2),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def render_report(main_rows, cat_rows, paired_rows, runs) -> str:
    lines = ["PHASE 4 RESULTS (exact match, official Natural Plan metric)", ""]
    lines.append("Seeds per config: " + ", ".join(f"{c}={sorted(runs[c])}" for c in CONFIG_ORDER if c in runs))
    lines.append("")
    lines.append(f"{'config':<17}{'rule':<22}{'EM mean':>8}{'SD':>8}{'95% CI (tasks)':>20}{'verified':>10}   per seed")
    for r in main_rows:
        seeds = [f"{r[k]:.4f}" for k in r if k.startswith("seed_")]
        star = " *" if r["headline"] else ""
        lines.append(
            f"{r['config']:<17}{r['rule']:<22}{r['exact_match_mean']:>8.4f}{r['exact_match_sd']:>8.4f}"
            f"{'[' + format(r['ci95_low'], '.4f') + ', ' + format(r['ci95_high'], '.4f') + ']':>20}"
            f"{r['verified_answer_rate']:>10.4f}   {' '.join(seeds)}{star}"
        )
    lines += ["", "* headline rule / reference row. 'verified' = share of answers the verifier scores 1.0 (valid-answer rate).", ""]
    lines.append("PAIRED COMPARISONS (A minus B; McNemar exact per seed; pooled = mean over seeds, bootstrap over tasks)")
    for r in paired_rows:
        if r["seed"] == "pooled":
            lines.append(
                f"    pooled: {r['difference_pp']:+.2f} pp  95% CI [{r['diff_ci95_low_pp']:+.2f}, {r['diff_ci95_high_pp']:+.2f}]"
            )
        else:
            lines.append(
                f"  {r['comparison']}  seed {r['seed']}: {r['A_exact_match']:.4f} vs {r['B_exact_match']:.4f} "
                f"({r['difference_pp']:+.2f} pp), only A {r['only_A_correct']}, only B {r['only_B_correct']}, p={r['mcnemar_p']:.4f}"
            )
    lines += ["", "PER CATEGORY (headline rules, mean over seeds)"]
    cats = sorted({r["category"] for r in cat_rows})
    lines.append(f"{'config / rule':<40}" + "".join(f"{c:>8}" for c in cats))
    for config in CONFIG_ORDER:
        for rule in RULE_ORDER[config]:
            sel = {r["category"]: r["exact_match_mean"] for r in cat_rows if r["config"] == config and r["rule"] == rule}
            if sel and (rule == HEADLINE_RULE or config in ("baseline", "verifier_solver", "sampling_control")):
                lines.append(f"{config + ' / ' + rule:<40}" + "".join(f"{sel.get(c, float('nan')):>8.3f}" for c in cats))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_root", default=str(PROJECT_ROOT / "results" / "runs"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "aggregate" / "main"))
    parser.add_argument("--no_derive", action="store_true", help="Reuse existing derived/ folders.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not args.no_derive:
        for run_dir in find_run_dirs(runs_root):
            derive_run(run_dir)
    runs = load_runs(runs_root)
    if not runs:
        raise SystemExit(f"No derived runs found under {runs_root}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_rows, cat_rows, paired_rows = main_table(runs), category_table(runs), paired_table(runs)
    write_csv(out / "table_main.csv", main_rows)
    write_csv(out / "per_category.csv", cat_rows)
    write_csv(out / "paired_tests.csv", paired_rows)
    report = render_report(main_rows, cat_rows, paired_rows, runs)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
