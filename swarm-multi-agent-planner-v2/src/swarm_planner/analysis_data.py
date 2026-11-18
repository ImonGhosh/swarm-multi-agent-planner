"""Load a finished swarm run (ToT-only or ToT+ACO) into plain Python structures for analysis.

Everything is read from the run's own files (``events.jsonl.gz``, ``predictions.json``,
``config.resolved.json``); nothing calls an LLM. Used by the analysis scripts that investigate how
ACO changed the search and by the figure and table scripts.

A run becomes a ``Run`` with one ``TaskLog`` per task. A task has ``T`` iterations, each with
``n`` traces (one per agent), the pheromone before and after the update, and the per-level
branch probabilities each agent used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .logging import EVENTS_FILENAME, read_events
from .metrics import exact_match
from .run_io import CONFIG_FILENAME, PREDICTIONS_FILENAME

LEVELS = 5
BRANCHES = 3


@dataclass
class Trace:
    it: int
    agent: int
    path: Tuple[str, ...]  # 5 edge keys, e.g. ("L1:constraint_first", ..., "L5:...")
    branch_idx: Tuple[int, ...]  # chosen branch index per level (0..2)
    probs: Tuple[Tuple[float, ...], ...]  # branch probabilities the agent used, per level
    answer: str
    score: float  # verifier score
    hard_valid: bool
    verified: bool  # score >= 1.0
    exact: bool  # official exact match against gold


@dataclass
class TaskLog:
    task_id: str
    category: str
    gold: str
    iterations: int
    traces: List[Trace] = field(default_factory=list)
    tau_before: Dict[int, List[float]] = field(default_factory=dict)  # it -> 15 values
    tau_after: Dict[int, List[float]] = field(default_factory=dict)

    def at(self, it: int) -> List[Trace]:
        return [t for t in self.traces if t.it == it]

    def first_verified_iteration(self) -> int | None:
        for it in range(self.iterations):
            if any(t.verified for t in self.at(it)):
                return it
        return None

    def first_exact_iteration(self) -> int | None:
        for it in range(self.iterations):
            if any(t.exact for t in self.at(it)):
                return it
        return None


@dataclass
class Run:
    name: str
    seed: int
    run_dir: Path
    config: Dict[str, Any]
    edge_order: List[str]  # 15 edge keys, level-major
    branch_order: Dict[int, List[str]]
    tasks: Dict[str, TaskLog]


def category_of(item: Dict[str, Any]) -> str:
    return f"{item['num_people']}p{item['num_days']}d"


def load_run(run_dir: str | Path) -> Run:
    run_path = Path(run_dir)
    config = json.loads((run_path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    data = json.loads((run_path / PREDICTIONS_FILENAME).read_text(encoding="utf-8"))
    iterations = int(config.get("iterations_per_task", 6))
    tasks: Dict[str, TaskLog] = {
        tid: TaskLog(tid, category_of(item), item["golden_plan"], iterations) for tid, item in data.items()
    }
    edge_order: List[str] = []
    branch_order: Dict[int, List[str]] = {}
    for event in read_events(run_path / EVENTS_FILENAME):
        kind = event.get("event")
        if kind == "run_start":
            edge_order = list(event["edge_order"])
            branch_order = {int(k): list(v) for k, v in event["branch_order"].items()}
        elif kind == "iteration":
            task = tasks[event["task_id"]]
            it = int(event["it"])
            if event.get("tau_before") is not None:  # ToT-only runs log no pheromone
                task.tau_before[it] = list(event["tau_before"])
                task.tau_after[it] = list(event["tau_after"])
            for tr in event["traces"]:
                path = tuple(tr["path"])
                idx = tuple(branch_order[lvl + 1].index(edge.split(":", 1)[1]) for lvl, edge in enumerate(path))
                answer = tr["ans"]
                v = tr["v"]
                task.traces.append(
                    Trace(
                        it=it,
                        agent=int(tr["agent"]),
                        path=path,
                        branch_idx=idx,
                        probs=tuple(tuple(level) for level in tr["p"]),
                        answer=answer,
                        score=float(tr["q"]),
                        hard_valid=bool(v["valid"]),
                        verified=float(tr["q"]) >= 1.0,
                        exact=exact_match(answer, task.gold),
                    )
                )
    return Run(
        name=config.get("config_name", run_path.parent.name),
        seed=int(config.get("seed", 0)),
        run_dir=run_path,
        config=config,
        edge_order=edge_order,
        branch_order=branch_order,
        tasks=tasks,
    )


def load_runs(runs_root: str | Path, name: str, seeds: Sequence[int] = (0, 1, 2)) -> List[Run]:
    return [load_run(Path(runs_root) / name / f"seed_{s}") for s in seeds]
