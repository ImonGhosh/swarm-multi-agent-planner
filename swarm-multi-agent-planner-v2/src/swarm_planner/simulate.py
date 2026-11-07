"""Offline replay of ToT and ToT+ACO from a path table. No LLM calls.

The path table (``path_table.py``) holds the verifier result and correctness of every path
of every task. A search run is then just a sampling process over that table: every agent
picks one branch per level (uniformly for ToT-only, with the ACO probabilities for ToT+ACO),
the path's stored result is looked up, and the answer is selected with the solver's own
ordering. Hundreds of seeds, any budget and any update rule can be replayed in seconds.

The branch probabilities, the pheromone update (``tau <- (1-rho)*tau + sum of deposits``)
and the selection key mirror ``aco.selection``, ``aco.pheromone`` and
``solver._solve_trace_sort_key``.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .aco import DEFAULT_ALPHA, DEFAULT_EPSILON, DEFAULT_RHO
from .legacy_eta import LEGACY_BETA as DEFAULT_BETA
from .legacy_eta import heuristic_desirability
from .tot.context import build_task_features, coerce_original_prompt
from .tot.levels import LEVEL_CONFIGS

LEVELS = len(LEVEL_CONFIGS)
BRANCH_IDS = [[b.branch_id for b in cfg.branches] for cfg in LEVEL_CONFIGS]
SHAPE = tuple(len(ids) for ids in BRANCH_IDS)
NO_RANK = 1_000_000


@dataclass
class TaskTable:
    task_id: str
    category: str
    correct: np.ndarray  # one entry per path, indexed by branch indices
    score: np.ndarray
    valid: np.ndarray
    soft_ratio: np.ndarray
    rank: np.ndarray
    eta: np.ndarray  # (levels, branches)
    answer_id: np.ndarray | None = None  # integer id of the final answer text of every path


def load_path_table(table_dir: str | Path, data_path: str | Path) -> List[TaskTable]:
    """Read ``path_table.csv.gz`` and compute each task's heuristic values."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    with gzip.open(Path(table_dir) / "path_table.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = list(dict.fromkeys(r["task_id"] for r in rows))
    tables: Dict[str, TaskTable] = {}
    for task_id in order:
        item = data[task_id]
        features = build_task_features(task_id, coerce_original_prompt(item["prompt_5shot"]), item)
        tables[task_id] = TaskTable(
            task_id=task_id,
            category=f"{item['num_people']}p{item['num_days']}d",
            correct=np.zeros(SHAPE),
            score=np.zeros(SHAPE),
            valid=np.zeros(SHAPE),
            soft_ratio=np.zeros(SHAPE),
            rank=np.full(SHAPE, float(NO_RANK)),
            eta=np.array(
                [[heuristic_desirability(lv + 1, b, features) for b in BRANCH_IDS[lv]] for lv in range(LEVELS)]
            ),
            answer_id=np.zeros(SHAPE, dtype=int),
        )
    answer_ids: Dict[str, Dict[str, int]] = {}
    for row in rows:
        table = tables[row["task_id"]]
        ids = answer_ids.setdefault(row["task_id"], {})
        idx = tuple(int(i) for i in row["path"].split("-"))
        table.answer_id[idx] = ids.setdefault(row["answer"], len(ids))
        table.correct[idx] = row["correct"] == "True"
        table.score[idx] = float(row["score"])
        table.valid[idx] = row["hard_valid"] == "True"
        total = int(row["soft_total"])
        table.soft_ratio[idx] = int(row["soft_satisfied"]) / total if total else 1.0
        if row["slot_rank"] != "":
            table.rank[idx] = float(row["slot_rank"])
    return [tables[t] for t in order]


# --- deposit rules: map the n scores of one iteration to the n deposits Q ---------------------


def deposit_v1(table: TaskTable, paths, scores: np.ndarray) -> np.ndarray:
    """v1: every trace deposits its verifier score, valid or not (finding F7)."""
    return scores


def deposit_valid_only(table: TaskTable, paths, scores: np.ndarray) -> np.ndarray:
    return np.array([table.score[p] if table.valid[p] else 0.0 for p in paths])


def deposit_top_only(table: TaskTable, paths, scores: np.ndarray) -> np.ndarray:
    return np.array([1.0 if table.score[p] == 1.0 else 0.0 for p in paths])


def deposit_best_only(table: TaskTable, paths, scores: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores))
    out[int(np.argmax(scores))] = scores.max()
    return out


def deposit_minus_mean(table: TaskTable, paths, scores: np.ndarray) -> np.ndarray:
    return np.clip(scores - scores.mean(), 0.0, None)


DEPOSIT_RULES: Dict[str, Callable] = {
    "v1_score": deposit_v1,
    "valid_only": deposit_valid_only,
    "top_only": deposit_top_only,
    "best_of_iteration": deposit_best_only,
    "score_minus_mean": deposit_minus_mean,
}


def _selection_key(table: TaskTable, path, path_score: float, agent: int):
    # solver._solve_trace_sort_key: score, hard_valid, soft ratio, -slot_rank, path_score, -agent
    return (
        table.score[path],
        table.valid[path],
        table.soft_ratio[path],
        -table.rank[path],
        path_score,
        -agent,
    )


def simulate_task(
    table: TaskTable,
    *,
    n_agents: int,
    n_iterations: int,
    rng: np.random.Generator,
    alpha: float = 0.0,
    beta: float = 0.0,
    rho: float = DEFAULT_RHO,
    deposit: Callable = deposit_v1,
    epsilon: float = DEFAULT_EPSILON,
    early_stop: bool = False,
) -> Dict[str, object]:
    """One search on one task.

    ``alpha = beta = 0`` gives the uniform ToT-only search (a uniform draw of every branch).
    ``early_stop`` ends the search after the first iteration that has a verifier score-1.0 answer.
    Returns correctness of the answer selected from the last iteration and from all iterations,
    whether any sampled path was correct, the iterations used, and the level-1 probabilities
    per iteration.
    """
    tau = np.ones((LEVELS, max(SHAPE)))
    best_last = best_all = None
    any_correct = False
    iterations_used = 0
    level1_history: List[np.ndarray] = []
    top_per_iteration: List[float] = []
    all_paths: List[tuple] = []
    iteration_bests: List[tuple] = []
    for iteration in range(n_iterations):
        iterations_used += 1
        weights = ((tau + epsilon) ** alpha) * ((table.eta + epsilon) ** beta)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        level1_history.append(probabilities[0].copy())
        cumulative = np.cumsum(probabilities, axis=1)
        draws = rng.random((n_agents, LEVELS))
        paths, keys = [], []
        for agent in range(n_agents):
            path = tuple(
                int(min(np.searchsorted(cumulative[lv], draws[agent, lv], side="left"), SHAPE[lv] - 1))
                for lv in range(LEVELS)
            )
            path_score = float(np.prod([probabilities[lv, path[lv]] for lv in range(LEVELS)]))
            paths.append(path)
            keys.append(_selection_key(table, path, path_score, agent))
            any_correct = any_correct or bool(table.correct[path])
        scores = np.array([table.score[p] for p in paths])
        top_per_iteration.append(float(np.mean(scores == 1.0)))
        all_paths.extend(paths)
        best_index = max(range(n_agents), key=lambda a: keys[a])
        best_last = (keys[best_index], paths[best_index])
        iteration_bests.append(best_last)
        if best_all is None or best_last[0] > best_all[0]:
            best_all = best_last

        new_tau = (1.0 - rho) * tau
        deposits = deposit(table, paths, scores)
        for agent, path in enumerate(paths):
            for lv in range(LEVELS):
                new_tau[lv, path[lv]] += deposits[agent]
        tau = new_tau
        if early_stop and np.any(scores == 1.0):
            break
    def vote(keep, window: int | None) -> bool:
        """The answer that most of the kept traces ended on (a vote counts traces, not distinct answers).

        ``keep`` selects which traces may vote (None: all), ``window`` limits them to the last
        ``window`` traces (None: all). Ties go to the answer seen first. If no trace is kept,
        the plain majority of the window decides.
        """
        if table.answer_id is None or not all_paths:
            return False
        sub = all_paths[-window:] if window else all_paths
        for rule in (keep, None):
            counts: Dict[int, int] = {}
            for p in sub:
                if rule is None or rule(p):
                    answer = int(table.answer_id[p])
                    counts[answer] = counts.get(answer, 0) + 1
            if counts:
                winner = max(counts, key=lambda a: counts[a])  # first seen wins ties
                return any(bool(table.correct[p]) for p in sub if int(table.answer_id[p]) == winner)
        return False

    verified = lambda p: bool(table.score[p] == 1.0)  # noqa: E731
    hard_valid = lambda p: bool(table.valid[p])  # noqa: E731
    majority_correct = vote(None, None)  # plain majority: no verifier involved
    last3, last2 = 3 * n_agents, 2 * n_agents
    def best_of_last(k: int) -> bool:
        window = iteration_bests[-k:]
        return bool(table.correct[max(window, key=lambda kv: kv[0])[1]])

    return {
        "last_correct": bool(table.correct[best_last[1]]),
        "last2_correct": best_of_last(2),
        "last3_correct": best_of_last(3),
        "all_correct": bool(table.correct[best_all[1]]),
        "last_top": bool(table.score[best_last[1]] == 1.0),
        "all_top": bool(table.score[best_all[1]] == 1.0),
        "any_correct": any_correct,
        "iterations_used": iterations_used,
        "level1_probabilities": level1_history,
        "top_share_first": top_per_iteration[0],
        "top_share_last": top_per_iteration[-1],
        "top_share_all": float(np.mean(top_per_iteration)),
        "majority_correct": majority_correct,
        # verified vote: only traces the verifier scores 1.0 vote (window = iterations counted)
        "vote_verified_all_correct": vote(verified, None),
        "vote_verified_last3_correct": vote(verified, last3),
        "vote_verified_last2_correct": vote(verified, last2),
        # hard-valid vote: every hard-valid trace votes
        "vote_valid_all_correct": vote(hard_valid, None),
        "vote_valid_last3_correct": vote(hard_valid, last3),
        "vote_valid_last2_correct": vote(hard_valid, last2),
        "paths": all_paths,
    }


def simulate(
    tables: List[TaskTable],
    *,
    reps: int,
    seed: int = 0,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Run ``reps`` searches per task. Returns one per-task mean array per metric."""
    rng = np.random.default_rng(seed)
    metrics = (
        "last_correct",
        "last2_correct",
        "last3_correct",
        "all_correct",
        "last_top",
        "all_top",
        "any_correct",
        "iterations_used",
        "top_share_first",
        "top_share_last",
        "top_share_all",
        "majority_correct",
        "vote_verified_all_correct",
        "vote_verified_last3_correct",
        "vote_verified_last2_correct",
        "vote_valid_all_correct",
        "vote_valid_last3_correct",
        "vote_valid_last2_correct",
    )
    out = {m: np.zeros(len(tables)) for m in metrics}
    for i, table in enumerate(tables):
        runs = [simulate_task(table, rng=rng, **kwargs) for _ in range(reps)]
        for m in metrics:
            out[m][i] = np.mean([float(r[m]) for r in runs])
    return out


def bootstrap_ci(values: np.ndarray, *, draws: int = 4000, seed: int = 5, level: float = 95.0):
    """Percentile bootstrap interval of the mean over tasks."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    low, high = np.percentile(values[idx].mean(axis=1), [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(low), float(high)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEPOSIT_RULES",
    "TaskTable",
    "bootstrap_ci",
    "load_path_table",
    "simulate",
    "simulate_task",
]
