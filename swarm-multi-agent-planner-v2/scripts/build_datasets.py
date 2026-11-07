#!/usr/bin/env python3
"""Build the curated test set and the dev set from the Natural Plan source file.

Tasks are grouped into the 10 Natural Plan categories (num_people, num_days) in
source-file order:

  * curated_400.json -- the first 40 tasks of each category (test set).
  * dev_100.json     -- tasks 41-50 of each category (dev set, disjoint from test).

The source's leftover pred_* fields are dropped so no stale predictions can be
mistaken for our results. If the v1 curated set is available it is compared with the
rebuilt test set and the script fails on any difference in task ids, order or content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT.parents[1] / "natural-plan" / "data" / "calendar_scheduling.json"
)
DEFAULT_V1_CURATED = (
    PROJECT_ROOT.parent
    / "swarm-multi-agent-planner-v1"
    / "data"
    / "calendar_scheduling_input_curated.json"
)

TEST_PER_CATEGORY = 40
DEV_PER_CATEGORY = 10
EXPECTED_FIELDS = {
    "num_people",
    "num_days",
    "duration",
    "prompt_5shot",
    "prompt_0shot",
    "golden_plan",
}

Category = Tuple[int, int]
Dataset = Dict[str, Dict[str, Any]]


def category_of(item: Dict[str, Any]) -> Category:
    return int(item["num_people"]), int(item["num_days"])


def clean_item(task_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {k: v for k, v in item.items() if not k.startswith("pred_")}
    if set(cleaned) != EXPECTED_FIELDS:
        raise ValueError(
            f"{task_id}: unexpected fields {sorted(set(cleaned) ^ EXPECTED_FIELDS)}"
        )
    return cleaned


def split_source(
    source: Dataset,
    *,
    test_per_category: int = TEST_PER_CATEGORY,
    dev_per_category: int = DEV_PER_CATEGORY,
) -> Tuple[Dataset, Dataset]:
    """Return (test, dev), each in source-file order."""
    seen: Dict[Category, int] = defaultdict(int)
    test: Dataset = {}
    dev: Dataset = {}
    for task_id, item in source.items():
        cat = category_of(item)
        position = seen[cat]
        seen[cat] += 1
        if position < test_per_category:
            test[task_id] = clean_item(task_id, item)
        elif position < test_per_category + dev_per_category:
            dev[task_id] = clean_item(task_id, item)
    return test, dev


def category_counts(data: Dataset) -> Dict[str, int]:
    counts: Dict[Category, int] = defaultdict(int)
    for item in data.values():
        counts[category_of(item)] += 1
    return {f"{p}p_{d}d": n for (p, d), n in sorted(counts.items())}


def write_json(path: Path, data: Any) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_matches_v1(test: Dataset, v1_path: Path) -> None:
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    if list(v1) != list(test):
        only_v1 = sorted(set(v1) - set(test))
        only_v2 = sorted(set(test) - set(v1))
        raise SystemExit(
            f"Test set differs from v1 curated set: {len(only_v1)} only in v1 "
            f"{only_v1[:3]}, {len(only_v2)} only in rebuilt {only_v2[:3]}, "
            f"or the order differs."
        )
    for task_id, item in test.items():
        v1_item = {k: v for k, v in v1[task_id].items() if not k.startswith("pred_")}
        if v1_item != item:
            raise SystemExit(f"{task_id}: content differs from v1 curated set.")
    print(f"OK: rebuilt test set matches v1 curated set exactly ({len(test)} tasks).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--v1_curated",
        type=Path,
        default=DEFAULT_V1_CURATED,
        help="v1 curated input to compare against (skipped with a warning if missing).",
    )
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source: Dataset = json.loads(source_bytes.decode("utf-8"))
    test, dev = split_source(source)

    if set(test) & set(dev):
        raise SystemExit("Test and dev sets overlap.")

    if args.v1_curated.exists():
        check_matches_v1(test, args.v1_curated)
    else:
        print(
            f"WARNING: {args.v1_curated} not found; skipped v1 comparison.",
            file=sys.stderr,
        )

    manifest: Dict[str, Any] = {
        "source": {
            "file": args.source.name,
            "tasks": len(source),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "split_rule": (
            "category = (num_people, num_days), in source-file order; "
            f"test = first {TEST_PER_CATEGORY} per category; "
            f"dev = next {DEV_PER_CATEGORY} per category"
        ),
        "files": {},
    }
    for name, data in (("curated_400.json", test), ("dev_100.json", dev)):
        digest = write_json(args.out_dir / name, data)
        manifest["files"][name] = {
            "tasks": len(data),
            "sha256": digest,
            "per_category": category_counts(data),
        }
        print(f"Wrote {args.out_dir / name}: {len(data)} tasks")
    write_json(args.out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
