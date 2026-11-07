"""The verifier must agree with the gold answers of the Natural Plan calendar tasks.

Natural Plan does not guarantee a single solution: when a task does not ask for the earliest
availability it can have several equally good answers, and the official metric accepts only
the gold one. Those tasks are listed in data/verifier_known_exceptions.json (regenerate with
scripts/audit_verifier.py after any verifier change). For every other task the gold answer
must be the unique top-scoring slot, or the verifier's first-ranked slot among ties.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FULL_SOURCE = ROOT.parents[1] / "natural-plan" / "data" / "calendar_scheduling.json"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_verifier", ROOT / "scripts" / "audit_verifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit_module()


@pytest.fixture(scope="module")
def known_exceptions():
    return {e["task_id"] for e in json.loads((DATA / "verifier_known_exceptions.json").read_text("utf-8"))}


def _audit_rows(audit, tasks):
    return [audit.audit_task(task_id, item) for task_id, item in tasks.items()]


def _check(rows, known_exceptions):
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row["task_id"])

    assert by_class.get("parse_error", []) == []
    assert by_class.get("gold_not_top", []) == []
    assert [r["task_id"] for r in rows if r["warnings"]] == []
    assert all(r["gold_hard_valid"] and r["gold_score"] == 1.0 for r in rows)

    ids = {r["task_id"] for r in rows}
    assert set(by_class.get("tie_gold_not_best", [])) == known_exceptions & ids
    # With an earliest request the answer is a single slot, so these ties cannot happen there.
    assert all(
        not r["earliest_requested"] for r in rows if r["class"] in ("tie_gold_is_best", "tie_gold_not_best")
    )


def test_gold_answers_in_repo_datasets(audit, known_exceptions):
    tasks = {}
    for name in ("curated_400.json", "dev_100.json"):
        tasks.update(json.loads((DATA / name).read_text("utf-8")))
    assert len(tasks) == 500
    _check(_audit_rows(audit, tasks), known_exceptions)


@pytest.mark.skipif(not FULL_SOURCE.exists(), reason="Natural Plan source file not available")
def test_gold_answers_in_all_1000_tasks(audit, known_exceptions):
    tasks = json.loads(FULL_SOURCE.read_text("utf-8"))
    assert len(tasks) == 1000
    rows = _audit_rows(audit, tasks)
    _check(rows, known_exceptions)
    assert len(known_exceptions) == 66
    summary = audit.summarize(rows)
    assert summary["accept_recall"] == 1.0
    assert summary["verifier_as_solver_correct"] == 934
