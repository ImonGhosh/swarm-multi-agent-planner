"""Derive every selection rule's predictions from a finished run's logs. No LLM calls.

A swarm run logs every trace of every iteration (``iteration`` events), a sampling-control run
logs every sample (``samples`` events). From those logs this module re-applies each rule of
``selection_rules`` and writes, into ``<run_dir>/derived/``:

    predictions_<rule>.json   that rule's answers with the fields scripts/evaluate.py reads (no prompts)
    per_task.csv              one row per task: exact match and verifier-verified flag per rule,
                              oracle pass@k, early-stopping iterations (read by evaluate_runs.py)
    selection_report.json     the same numbers aggregated (exact match overall and per category)

Baseline and verifier-as-solver runs have a single answer per task, so their report has one rule.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .logging import EVENTS_FILENAME, read_events
from .metrics import exact_match
from .run_io import CONFIG_FILENAME, PREDICTIONS_FILENAME
from .selection_rules import (
    HEADLINE_RULE,
    SELECTION_RULES,
    TraceRec,
    iterations_until_first_verified,
    prediction_of,
    rec_from_log,
    select,
)
from .solver import scorer_for_item

DERIVED_DIRNAME = "derived"
SAMPLING_RULES: Tuple[str, ...] = ("best_of_samples", "majority_vote")
CALLS_PER_TRAVERSAL = 6  # 5 level calls + the finalize call


def category_of(item: Dict[str, Any]) -> str:
    return f"{item['num_people']}p{item['num_days']}d"


def _read_logged_traces(run_dir: Path) -> Dict[str, List[TraceRec]]:
    """All traces (or samples) of every task, from the run's event log."""
    recs: Dict[str, List[TraceRec]] = defaultdict(list)
    for event in read_events(run_dir / EVENTS_FILENAME):
        kind = event.get("event")
        if kind == "iteration":
            recs[event["task_id"]].extend(rec_from_log(int(event["it"]), t) for t in event["traces"])
        elif kind == "samples":
            recs[event["task_id"]] = [rec_from_log(0, s) for s in event["samples"]]
    return recs


def _rules_for(mode: str) -> Tuple[str, ...]:
    if mode == "solve":
        return SELECTION_RULES
    if mode == "sampling_control":
        return SAMPLING_RULES
    return ()


def _select_for_rule(recs: List[TraceRec], rule: str) -> TraceRec | None:
    if rule == "best_of_samples":
        return select(recs, "best_of_all")
    return select(recs, rule)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def derive_run(run_dir: str | Path, *, write_predictions: bool = True) -> Dict[str, Any]:
    """Apply every rule to one run. Returns the selection report (also written to disk)."""
    run_path = Path(run_dir)
    config = json.loads((run_path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    mode = config["mode"]
    data: Dict[str, Dict[str, Any]] = json.loads((run_path / PREDICTIONS_FILENAME).read_text(encoding="utf-8"))
    rules = _rules_for(mode)
    logged = _read_logged_traces(run_path) if rules else {}
    if mode == "baseline":
        rules = ("single_call",)
    elif mode == "verifier_solver":
        rules = ("verifier_best_slot",)

    task_ids = list(data)
    predictions: Dict[str, Dict[str, str]] = {rule: {} for rule in rules}
    rows: List[Dict[str, Any]] = []
    iteration_verified: Dict[int, List[float]] = defaultdict(list)
    iteration_best_correct: Dict[int, List[float]] = defaultdict(list)

    for task_id in task_ids:
        item = data[task_id]
        gold = item["golden_plan"]
        scorer = scorer_for_item(item)
        row: Dict[str, Any] = {"task_id": task_id, "category": category_of(item)}
        recs = logged.get(task_id, [])
        if rules and mode in ("solve", "sampling_control") and not recs:
            raise ValueError(f"{run_path}: no logged traces for task {task_id}; is the run complete?")
        for rule in rules:
            if mode in ("solve", "sampling_control"):
                prediction = prediction_of(_select_for_rule(recs, rule))
            else:
                prediction = item["pred_5shot_pro"]
            predictions[rule][task_id] = prediction
            result = scorer.score_prediction(prediction)
            row[rule] = int(exact_match(prediction, gold))
            row[f"{rule}.verified"] = int(result.score >= 1.0)
            row[f"{rule}.hard_valid"] = int(result.hard_valid)
        if recs:
            correct = [exact_match(r.answer, gold) for r in recs]
            row["oracle_pass48" if mode == "solve" else "oracle_pass_all"] = int(any(correct))
            if mode == "solve":
                row["oracle_pass8"] = int(any(c for r, c in zip(recs, correct) if r.it == 0))
                row["early_stop_iterations"] = iterations_until_first_verified(recs)
                for it in sorted({r.it for r in recs}):
                    in_it = [r for r in recs if r.it == it]
                    iteration_verified[it].append(sum(r.verified for r in in_it) / len(in_it))
                    best = select(in_it, "best_of_all")
                    iteration_best_correct[it].append(float(exact_match(best.answer, gold)))
            else:
                row["single_sample_acc"] = sum(correct) / len(correct)
        rows.append(row)

    out_dir = run_path / DERIVED_DIRNAME
    out_dir.mkdir(exist_ok=True)
    if write_predictions and mode in ("solve", "sampling_control"):
        for rule in rules:
            # only what the official evaluator reads (the long prompts stay in the run's predictions.json)
            copy = {
                tid: {
                    "num_people": data[tid]["num_people"],
                    "num_days": data[tid]["num_days"],
                    "golden_plan": data[tid]["golden_plan"],
                    "pred_5shot_pro": predictions[rule][tid],
                }
                for tid in task_ids
            }
            copy_path = out_dir / f"predictions_{rule}.json"
            copy_path.write_text(json.dumps(copy, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames: List[str] = []
    for row in rows:
        fieldnames.extend(k for k in row if k not in fieldnames)
    with (out_dir / "per_task.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    categories = sorted({r["category"] for r in rows})
    report: Dict[str, Any] = {
        "run_dir": str(run_path),
        "config_name": config.get("config_name", mode),
        "mode": mode,
        "seed": config.get("seed"),
        "tasks": len(rows),
        "headline_rule": HEADLINE_RULE if mode == "solve" else rules[0],
        "rules": {},
    }
    for rule in rules:
        flags = [r[rule] for r in rows]
        report["rules"][rule] = {
            "exact_match": _mean(flags),
            "correct": sum(flags),
            "verified_answer_rate": _mean([r[f"{rule}.verified"] for r in rows]),
            "hard_valid_rate": _mean([r[f"{rule}.hard_valid"] for r in rows]),
            "per_category": {
                c: _mean([r[rule] for r in rows if r["category"] == c]) for c in categories
            },
        }
    if mode == "solve":
        used = [r["early_stop_iterations"] for r in rows]
        iterations = int(config.get("iterations_per_task", 6))
        agents = int(config.get("agents_per_task", 8))
        report["early_stop"] = {
            "mean_iterations": _mean(used),
            "mean_traversals": _mean(used) * agents,
            "mean_llm_calls": _mean(used) * agents * CALLS_PER_TRAVERSAL,
            "full_budget_iterations": iterations,
            "share_of_budget": _mean(used) / iterations,
        }
        report["oracle"] = {
            "pass_at_first_iteration": _mean([r["oracle_pass8"] for r in rows]),
            "pass_at_all_traces": _mean([r["oracle_pass48"] for r in rows]),
        }
        report["per_iteration"] = {
            "verified_trace_share": [_mean(iteration_verified[i]) for i in sorted(iteration_verified)],
            "best_of_iteration_exact_match": [
                _mean(iteration_best_correct[i]) for i in sorted(iteration_best_correct)
            ],
        }
    elif mode == "sampling_control":
        report["oracle"] = {"pass_at_all_samples": _mean([r["oracle_pass_all"] for r in rows])}
        report["single_sample_exact_match"] = _mean([r["single_sample_acc"] for r in rows])
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def find_run_dirs(runs_root: str | Path) -> List[Path]:
    """Every run directory below results/runs (a directory that has a config.resolved.json)."""
    return sorted(p.parent for p in Path(runs_root).rglob(CONFIG_FILENAME) if DERIVED_DIRNAME not in p.parts)
