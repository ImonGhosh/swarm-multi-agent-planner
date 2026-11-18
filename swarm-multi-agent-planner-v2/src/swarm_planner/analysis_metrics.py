"""Measures that show how ACO changed the search, computed from the logs of finished runs.

All functions take ``Run`` objects (``analysis_data``) and return plain rows (lists of dicts) that
``scripts/make_tables.py`` writes as CSV and ``scripts/make_figures.py`` plots. No LLM calls.

Definitions
-----------
* *verified* trace: the verifier scores its answer 1.0.
* Colony probabilities of an iteration: ``p = (tau + eps) / sum(tau + eps)`` per level, from the
  logged ``tau_before`` (all agents of an iteration share the same pheromone).
* *Branch quality* of a task: for each level and branch, the share of that branch's traces that are
  verified, estimated from the **ToT-only** runs (uniform sampling, so the estimate is unbiased
  and independent of anything ACO did). It is the reference ACO's pheromone is compared with.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

from .analysis_data import BRANCHES, LEVELS, Run, TaskLog

EPSILON = 0.001  # ACO epsilon (configs of the main runs)
ALPHA = 1.0


def probs_from_tau(tau: Sequence[float], epsilon: float = EPSILON, alpha: float = ALPHA) -> np.ndarray:
    """(5, 3) branch probabilities from a 15-vector of pheromone values (level-major)."""
    x = (np.asarray(tau, dtype=float).reshape(LEVELS, BRANCHES) + epsilon) ** alpha
    return x / x.sum(axis=1, keepdims=True)


def norm_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy of each row divided by log(3): 1 = uniform, 0 = all mass on one branch."""
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1) / math.log(BRANCHES)


def had_success_before(task: TaskLog, it: int) -> bool:
    return any(t.verified for t in task.traces if t.it < it)


# ----------------------------------------------------------------------------- per iteration
def per_iteration_rows(runs: Sequence[Run]) -> List[Dict[str, Any]]:
    """One row per (run, iteration): means over tasks of verifier-based search statistics."""
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for it in range(int(run.config.get("iterations_per_task", 6))):
            share_all, share_cond, best_q, mean_q, valid, exact = [], [], [], [], [], []
            cum_verified, cum_exact, distinct_paths = [], [], []
            for task in run.tasks.values():
                tr = task.at(it)
                v = [t.verified for t in tr]
                share_all.append(float(np.mean(v)))
                if had_success_before(task, it):
                    share_cond.append(float(np.mean(v)))
                best_q.append(max(t.score for t in tr))
                mean_q.append(float(np.mean([t.score for t in tr])))
                valid.append(float(np.mean([t.hard_valid for t in tr])))
                exact.append(float(np.mean([t.exact for t in tr])))
                upto = [t for t in task.traces if t.it <= it]
                cum_verified.append(float(any(t.verified for t in upto)))
                cum_exact.append(float(any(t.exact for t in upto)))
                distinct_paths.append(len({t.path for t in upto}))
            rows.append(
                {
                    "config": run.name,
                    "seed": run.seed,
                    "it": it + 1,  # 1-based for plots
                    "verified_trace_share": float(np.mean(share_all)),
                    "verified_trace_share_after_first_success": float(np.mean(share_cond)) if share_cond else float("nan"),
                    "tasks_after_first_success": len(share_cond),
                    "best_score": float(np.mean(best_q)),
                    "mean_score": float(np.mean(mean_q)),
                    "hard_valid_trace_share": float(np.mean(valid)),
                    "exact_trace_share": float(np.mean(exact)),
                    "cum_tasks_with_verified": float(np.mean(cum_verified)),
                    "cum_tasks_with_exact": float(np.mean(cum_exact)),
                    "cum_distinct_paths": float(np.mean(distinct_paths)),
                }
            )
    return rows


def first_success_rows(runs: Sequence[Run]) -> List[Dict[str, Any]]:
    """One row per (run, task): iteration (1-based) of the first verified / exact-match trace."""
    rows = []
    for run in runs:
        for task in run.tasks.values():
            fv, fe = task.first_verified_iteration(), task.first_exact_iteration()
            rows.append(
                {
                    "config": run.name,
                    "seed": run.seed,
                    "task_id": task.task_id,
                    "category": task.category,
                    "first_verified_it": "" if fv is None else fv + 1,
                    "first_exact_it": "" if fe is None else fe + 1,
                    "distinct_paths_48": len({t.path for t in task.traces}),
                    "distinct_paths_last2": len({t.path for t in task.traces if t.it >= task.iterations - 2}),
                }
            )
    return rows


def share_difference_rows(tot: Sequence[Run], aco: Sequence[Run], n_boot: int = 2000, seed: int = 0) -> List[Dict[str, Any]]:
    """ACO minus ToT-only verified-trace share, per iteration, mean over seeds with a task bootstrap CI."""
    rng = np.random.default_rng(seed)
    rows = []
    task_ids = list(tot[0].tasks)
    n_it = int(tot[0].config.get("iterations_per_task", 6))
    for it in range(n_it):
        # per task, mean over seeds of the verified share in that iteration
        def per_task(runs):
            return np.array([[np.mean([t.verified for t in r.tasks[tid].at(it)]) for tid in task_ids] for r in runs]).mean(0)

        d = per_task(aco) - per_task(tot)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        rows.append(
            {
                "it": it + 1,
                "tot_only_share": float(per_task(tot).mean()),
                "tot_aco_share": float(per_task(aco).mean()),
                "difference": float(d.mean()),
                "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
            }
        )
    return rows


# ----------------------------------------------------------------------------- pheromone
def pheromone_rows(aco: Sequence[Run]) -> List[Dict[str, Any]]:
    """One row per (seed, iteration, level, branch): mean colony probability over tasks.

    ``subset`` is 'all' (every task) or 'after_success' (tasks that had a verified trace in an
    earlier iteration, i.e. tasks where a deposit has happened). Iteration 7 is the state after
    the last update. Also the normalised entropy of the level (1 = uniform).
    """
    rows = []
    for run in aco:
        names = {lvl: run.branch_order[lvl] for lvl in range(1, LEVELS + 1)}
        n_it = int(run.config.get("iterations_per_task", 6))
        for it in range(n_it + 1):  # it == n_it: after the last update
            for subset in ("all", "after_success"):
                ps, ents = [], []
                for task in run.tasks.values():
                    if subset == "after_success" and not had_success_before(task, it):
                        continue
                    tau = task.tau_before[it] if it < n_it else task.tau_after[n_it - 1]
                    p = probs_from_tau(tau)
                    ps.append(p)
                    ents.append(norm_entropy(p))
                if not ps:
                    continue
                mp, me = np.mean(ps, axis=0), np.mean(ents, axis=0)
                for lvl in range(LEVELS):
                    for b in range(BRANCHES):
                        rows.append(
                            {
                                "seed": run.seed,
                                "it": it + 1,
                                "subset": subset,
                                "tasks": len(ps),
                                "level": lvl + 1,
                                "branch": names[lvl + 1][b],
                                "branch_idx": b,
                                "mean_probability": float(mp[lvl, b]),
                                "mean_norm_entropy": float(me[lvl]),
                            }
                        )
    return rows


def branch_quality(tot: Sequence[Run]) -> Dict[str, np.ndarray]:
    """Per task: (5, 3) verified rate of each branch from the pooled uniform ToT-only traces (nan = unseen)."""
    quality: Dict[str, np.ndarray] = {}
    for tid in tot[0].tasks:
        hits = np.zeros((LEVELS, BRANCHES))
        count = np.zeros((LEVELS, BRANCHES))
        for run in tot:
            for t in run.tasks[tid].traces:
                for lvl, b in enumerate(t.branch_idx):
                    count[lvl, b] += 1
                    hits[lvl, b] += t.verified
        with np.errstate(invalid="ignore", divide="ignore"):
            quality[tid] = np.where(count > 0, hits / count, np.nan)
    return quality


def agreement_rows(aco: Sequence[Run], tot: Sequence[Run], n_boot: int = 2000, seed: int = 0) -> List[Dict[str, Any]]:
    """Does the pheromone point at the branches that really work?

    For each level, over *informative* tasks (the ToT-only estimate has a unique best branch with a
    clearly higher verified rate than the worst) and tasks where ACO deposited at least once:
    the probability mass the final colony state puts on that best branch (1/3 = no learning), the
    share of tasks whose most probable branch is the best branch (chance 1/3) and the same for the
    worst branch. 'iteration' rows give the mass on the best branch over time.
    """
    rng = np.random.default_rng(seed)
    qual = branch_quality(tot)
    rows: List[Dict[str, Any]] = []
    n_it = int(aco[0].config.get("iterations_per_task", 6))
    for lvl in range(LEVELS):
        informative = []
        for tid, q in qual.items():
            qq = q[lvl]
            if np.isnan(qq).any():
                continue
            if qq.max() - qq.min() < 0.05:  # no meaningful spread between branches
                continue
            if (qq == qq.max()).sum() != 1 or (qq == qq.min()).sum() != 1:
                continue
            informative.append((tid, int(qq.argmax()), int(qq.argmin())))
        for run in aco:
            masses_by_it: Dict[int, List[float]] = defaultdict(list)
            top1_best, top1_worst, final_mass, final_worst_mass = [], [], [], []
            for tid, best, worst in informative:
                task = run.tasks[tid]
                if not any(t.verified for t in task.traces):
                    continue  # no deposit ever happened: pheromone cannot have learned
                for it in range(n_it):
                    if had_success_before(task, it):
                        masses_by_it[it].append(float(probs_from_tau(task.tau_before[it])[lvl, best]))
                p = probs_from_tau(task.tau_after[n_it - 1])[lvl]
                final_mass.append(float(p[best]))
                final_worst_mass.append(float(p[worst]))
                am = np.flatnonzero(p == p.max())
                top1_best.append(float(len(am) == 1 and am[0] == best))
                top1_worst.append(float(len(am) == 1 and am[0] == worst))
            if not final_mass:
                continue
            fm = np.array(final_mass)
            boots = np.array([fm[rng.integers(0, len(fm), len(fm))].mean() for _ in range(n_boot)])
            rows.append(
                {
                    "kind": "final",
                    "seed": run.seed,
                    "level": lvl + 1,
                    "tasks": len(fm),
                    "mass_on_best_branch": float(fm.mean()),
                    "mass_ci_low": float(np.quantile(boots, 0.025)),
                    "mass_ci_high": float(np.quantile(boots, 0.975)),
                    "mass_on_worst_branch": float(np.mean(final_worst_mass)),
                    "top1_is_best": float(np.mean(top1_best)),
                    "top1_is_worst": float(np.mean(top1_worst)),
                    "chance": 1 / 3,
                }
            )
            for it, vals in sorted(masses_by_it.items()):
                rows.append(
                    {
                        "kind": "iteration",
                        "seed": run.seed,
                        "level": lvl + 1,
                        "it": it + 1,
                        "tasks": len(vals),
                        "mass_on_best_branch": float(np.mean(vals)),
                    }
                )
    return rows


# ----------------------------------------------------------------------------- diversity of the candidates
def _diversity_row(config: str, seed: int, per_task: List[List[tuple]]) -> Dict[str, Any]:
    """per_task: for each task a list of (answer, verified) for its 48 candidates."""
    n = len(per_task)
    return {
        "config": config,
        "seed": seed,
        "tasks": n,
        "candidates_per_task": float(np.mean([len(c) for c in per_task])),
        "verified_share_per_candidate": float(np.mean([np.mean([v for _, v in c]) for c in per_task])),
        "tasks_with_a_verified_candidate": float(np.mean([any(v for _, v in c) for c in per_task])),
        "mean_distinct_answers_per_task": float(np.mean([len({a for a, _ in c}) for c in per_task])),
        "tasks_with_one_distinct_answer": float(np.mean([len({a for a, _ in c}) == 1 for c in per_task])),
    }


def diversity_rows_from_runs(runs: Sequence[Run]) -> List[Dict[str, Any]]:
    """ToT-only / ToT+ACO: the 48 traces of each task as candidates."""
    return [
        _diversity_row(run.name, run.seed, [[(t.answer, t.verified) for t in task.traces] for task in run.tasks.values()])
        for run in runs
    ]


def diversity_row_from_samples(run_dir, config: str, seed: int) -> Dict[str, Any]:
    """Sampling control: the 48 direct samples of each task (``samples`` events; q >= 1.0 = verified)."""
    from .logging import EVENTS_FILENAME, read_events

    per_task = []
    for e in read_events(f"{run_dir}/{EVENTS_FILENAME}"):
        if e.get("event") == "samples":
            per_task.append([(x["ans"], float(x["q"]) >= 1.0) for x in e["samples"]])
    return _diversity_row(config, seed, per_task)


# ----------------------------------------------------------------------------- cost
def cost_rows_from_events(run_dir, config: str, seed: int) -> Dict[str, Any]:
    """Calls, tokens, nominal cost and latency of one run from its llm_call events."""
    from .logging import EVENTS_FILENAME, read_events

    calls = hits = 0
    tok_in = tok_out = 0
    nominal = 0.0
    seen: Dict[str, float] = {}
    paid_latency: List[float] = []
    providers: Counter = Counter()
    for e in read_events(f"{run_dir}/{EVENTS_FILENAME}"):
        if e.get("event") != "llm_call":
            continue
        calls += 1
        hits += bool(e.get("hit"))
        tok_in += int(e.get("in") or 0)
        tok_out += int(e.get("out") or 0)
        cost = float(e.get("cost") or 0.0)
        nominal += cost
        seen.setdefault(e["req"], cost)
        if not e.get("hit") and e.get("lat"):
            paid_latency.append(float(e["lat"]))
        providers[e.get("prov") or "?"] += 1
    return {
        "config": config,
        "seed": seed,
        "llm_calls": calls,
        "distinct_requests": len(seen),
        "cache_hit_calls": hits,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "usd_all_calls_uncached": round(nominal, 4),
        "usd_distinct_requests": round(sum(seen.values()), 4),
        "mean_latency_paid_call_s": round(float(np.mean(paid_latency)), 3) if paid_latency else "",
        "providers": ";".join(f"{k}:{v}" for k, v in providers.items()),
    }
