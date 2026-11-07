from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
V1_CURATED = (
    ROOT.parent
    / "swarm-multi-agent-planner-v1"
    / "data"
    / "calendar_scheduling_input_curated.json"
)

_spec = importlib.util.spec_from_file_location(
    "build_datasets", ROOT / "scripts" / "build_datasets.py"
)
build_datasets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_datasets)  # type: ignore[union-attr]


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _category(item: dict) -> tuple[int, int]:
    return int(item["num_people"]), int(item["num_days"])


def _item(people: int, days: int, n: int) -> dict:
    return {
        "num_people": str(people),
        "num_days": str(days),
        "duration": "1",
        "prompt_5shot": f"5shot {n}",
        "prompt_0shot": f"0shot {n}",
        "golden_plan": "Here is the proposed time: Monday, 9:00 - 9:30",
        "pred_5shot_pro": "stale prediction",
    }


def test_split_takes_first_n_per_category_in_file_order_and_drops_predictions():
    source = {}
    for n in range(10):  # interleave two categories
        source[f"a{n}"] = _item(2, 1, n)
        source[f"b{n}"] = _item(3, 1, n)
    test, dev = build_datasets.split_source(
        source, test_per_category=4, dev_per_category=2
    )
    assert list(test) == [f"{c}{n}" for n in range(4) for c in "ab"]
    assert list(dev) == [f"{c}{n}" for n in range(4, 6) for c in "ab"]
    assert not set(test) & set(dev)
    assert all("pred_5shot_pro" not in item for item in (*test.values(), *dev.values()))


def test_unexpected_source_fields_are_rejected():
    item = _item(2, 1, 0)
    item["surprise"] = "x"
    with pytest.raises(ValueError):
        build_datasets.clean_item("t", item)


def test_curated_set_shape():
    curated = _load("curated_400.json")
    assert len(curated) == 400
    counts = Counter(_category(item) for item in curated.values())
    assert len(counts) == 10 and set(counts.values()) == {40}


def test_dev_set_shape_and_no_overlap_with_test():
    curated, dev = _load("curated_400.json"), _load("dev_100.json")
    assert len(dev) == 100
    counts = Counter(_category(item) for item in dev.values())
    assert len(counts) == 10 and set(counts.values()) == {10}
    assert not set(curated) & set(dev)


def test_no_leftover_prediction_fields():
    for name in ("curated_400.json", "dev_100.json"):
        for task_id, item in _load(name).items():
            assert set(item) == build_datasets.EXPECTED_FIELDS, task_id


@pytest.mark.skipif(not V1_CURATED.exists(), reason="v1 curated set not available")
def test_curated_set_matches_v1_exactly():
    v1 = json.loads(V1_CURATED.read_text(encoding="utf-8"))
    curated = _load("curated_400.json")
    assert list(curated) == list(v1)
    for task_id, item in curated.items():
        assert item == {k: v for k, v in v1[task_id].items() if not k.startswith("pred_")}
