#!/usr/bin/env python3
"""Audit the deterministic verifier against the gold answers of all 1000 tasks.

For every task the script scores the gold answer and every hard-valid slot, then sorts the
task into one class:

  unique             the gold slot is the only top-scoring slot
  tie_gold_is_best   several top-scoring slots, gold is the one the verifier ranks first
  tie_gold_not_best  several top-scoring slots, gold is not the one ranked first. The task has
                     several equally good answers and no deterministic verifier can tell which
                     one the gold is. These tasks are the documented exceptions.
  gold_not_top       gold is hard-invalid or scores below the top slot: a verifier defect
  parse_error        the task text could not be parsed

Outputs (in --out-dir):
  audit_tasks.csv         one row per task
  audit_summary.json      class counts, precision/recall, verifier-as-solver accuracy, per subset
  known_exceptions.json   the tie_gold_not_best tasks (data/verifier_known_exceptions.json is a copy)

Usage:
  python scripts/audit_verifier.py            # all 1000 tasks of the Natural Plan source
  python scripts/audit_verifier.py --source data/curated_400.json --out-dir /tmp/audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swarm_planner.verifier.parser import (  # noqa: E402
    minutes_to_time,
    parse_proposed_time,
    time_to_minutes,
)
from swarm_planner.verifier.scorer import build_task_quality_scorer  # noqa: E402

DEFAULT_SOURCE = (
    PROJECT_ROOT.parents[1] / "natural-plan" / "data" / "calendar_scheduling.json"
)
DEFAULT_OUT = PROJECT_ROOT / "results" / "verifier_audit"
SUBSETS = {"test_400": "curated_400.json", "dev_100": "dev_100.json"}
CLASSES = ("unique", "tie_gold_is_best", "tie_gold_not_best", "gold_not_top", "parse_error")


def _fmt(slot) -> str:
    day, start, end = slot
    return f"{day}, {minutes_to_time(start)} - {minutes_to_time(end)}"


def _gold_slot(golden_plan: str):
    parsed = parse_proposed_time(golden_plan)
    if parsed is None:
        return None
    day, start, end = parsed
    return day.capitalize(), time_to_minutes(start), time_to_minutes(end)


def audit_task(task_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    gold = _gold_slot(item["golden_plan"])
    row: Dict[str, Any] = {
        "task_id": task_id,
        "num_people": item.get("num_people"),
        "num_days": item.get("num_days"),
        "duration": item.get("duration"),
        "gold_slot": _fmt(gold) if gold else "",
        "class": "parse_error",
        "warnings": "",
    }
    try:
        scorer = build_task_quality_scorer(item["prompt_0shot"])
    except ValueError as err:
        row["warnings"] = f"parse_error: {err}"
        return row

    gold_result = scorer.score_prediction(item["golden_plan"])
    top = scorer.top_slots()
    best = scorer.best_slot
    gold_in_top = gold in top
    if not gold_in_top:
        cls = "gold_not_top"
    elif len(top) == 1:
        cls = "unique"
    elif gold == best:
        cls = "tie_gold_is_best"
    else:
        cls = "tie_gold_not_best"

    row.update(
        {
            "class": cls,
            "warnings": " | ".join(scorer.task.warnings),
            "n_hard_valid": len(scorer.valid_slots),
            "n_top": len(top),
            "gold_hard_valid": gold_result.hard_valid,
            "gold_score": gold_result.score,
            "gold_soft": f"{gold_result.soft_satisfied}/{gold_result.soft_total}",
            "gold_in_top": gold_in_top,
            "best_slot": _fmt(best) if best else "",
            "solver_correct": best == gold,
            "n_soft_prefs": len(scorer.task.soft_preferences),
            "n_hard_prefs": len(scorer.task.hard_preferences),
            "earliest_requested": scorer.task.earliest_requested,
        }
    )
    return row


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    parsed = [r for r in rows if r["class"] != "parse_error"]
    classes = Counter(r["class"] for r in rows)
    solver_correct = sum(1 for r in parsed if r["solver_correct"])
    gold_valid = sum(1 for r in parsed if r["gold_hard_valid"])
    gold_top = sum(1 for r in parsed if r["gold_in_top"])
    valid_slots = sum(int(r["n_hard_valid"]) for r in parsed)
    top_slots = sum(int(r["n_top"]) for r in parsed)

    def ratio(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    summary: Dict[str, Any] = {
        "tasks": n,
        "classes": {c: classes.get(c, 0) for c in CLASSES},
        "tasks_with_parser_warnings": sum(1 for r in rows if r["warnings"]),
        "gold_score_1": sum(1 for r in parsed if r["gold_score"] == 1.0),
        # Hard-validity filter, slot level: positives = the gold slot.
        "hard_valid_recall": ratio(gold_valid, len(parsed)),
        "hard_valid_precision": ratio(gold_valid, valid_slots),
        # Accept = top-scoring slots, slot level.
        "accept_recall": ratio(gold_top, len(parsed)),
        "accept_precision": ratio(gold_top, top_slots),
        "tasks_with_several_top_slots": sum(1 for r in parsed if r["n_top"] > 1),
        # Verifier-as-solver: answer with the verifier's first-ranked slot.
        "verifier_as_solver_correct": solver_correct,
        "verifier_as_solver_accuracy": ratio(solver_correct, n),
    }

    per_cat: Dict[str, Counter] = defaultdict(Counter)
    for r in parsed:
        key = f"{r['num_people']}p_{r['num_days']}d"
        per_cat[key]["tasks"] += 1
        per_cat[key]["solver_correct"] += int(r["solver_correct"])
        per_cat[key][r["class"]] += 1
    summary["per_category"] = {k: dict(v) for k, v in sorted(per_cat.items())}

    subsets: Dict[str, Any] = {}
    for name, filename in SUBSETS.items():
        path = PROJECT_ROOT / "data" / filename
        if not path.exists():
            continue
        ids = set(json.loads(path.read_text(encoding="utf-8")))
        sub = [r for r in rows if r["task_id"] in ids]
        if not sub:
            continue
        sub_ok = sum(1 for r in sub if r.get("solver_correct"))
        subsets[name] = {
            "tasks": len(sub),
            "classes": dict(Counter(r["class"] for r in sub)),
            "verifier_as_solver_correct": sub_ok,
            "verifier_as_solver_accuracy": ratio(sub_ok, len(sub)),
        }
    if subsets:
        summary["subsets"] = subsets
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = json.loads(args.source.read_text(encoding="utf-8"))
    rows = [audit_task(task_id, item) for task_id, item in data.items()]
    summary = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with (args.out_dir / "audit_tasks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    exceptions = [
        {
            "task_id": r["task_id"],
            "reason": "several equally good answers; gold is not the earliest of them",
            "n_top_slots": r["n_top"],
            "gold": r["gold_slot"],
            "verifier_best": r["best_slot"],
        }
        for r in rows
        if r["class"] == "tie_gold_not_best"
    ]
    (args.out_dir / "known_exceptions.json").write_text(
        json.dumps(exceptions, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_category"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
