"""Tests for the ACO-effect measures (analysis_metrics) on small synthetic runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from swarm_planner.analysis_data import BRANCHES, LEVELS, Run, TaskLog, Trace
from swarm_planner.analysis_metrics import (
    agreement_rows,
    branch_quality,
    diversity_rows_from_runs,
    first_success_rows,
    norm_entropy,
    per_iteration_rows,
    probs_from_tau,
)

BRANCH_ORDER = {lvl: [f"b{lvl}{k}" for k in range(BRANCHES)] for lvl in range(1, LEVELS + 1)}


def make_trace(it, agent, idx, verified):
    path = tuple(f"L{lvl + 1}:{BRANCH_ORDER[lvl + 1][b]}" for lvl, b in enumerate(idx))
    return Trace(it, agent, path, tuple(idx), tuple((1 / 3,) * 3 for _ in range(LEVELS)), "ans", 1.0 if verified else 0.2,
                 verified, verified, verified)


def make_run(name, traces_by_task, tau_after=None):
    tasks = {}
    for tid, traces in traces_by_task.items():
        t = TaskLog(tid, "2p1d", "gold", 2, traces)
        if tau_after is not None:
            for it in range(2):
                t.tau_before[it] = [1.0] * 15
                t.tau_after[it] = tau_after
        tasks[tid] = t
    return Run(name, 0, Path("."), {"iterations_per_task": 2}, [], BRANCH_ORDER, tasks)


def test_probs_from_tau_uniform_and_peaked():
    assert np.allclose(probs_from_tau([1.0] * 15), 1 / 3)
    tau = [1.0] * 15
    tau[0] = 9.0  # level 1, branch 0
    p = probs_from_tau(tau)
    assert p[0, 0] > 0.8 and np.isclose(p.sum(axis=1), 1).all()


def test_norm_entropy_bounds():
    assert np.isclose(norm_entropy(np.array([1 / 3, 1 / 3, 1 / 3])), 1.0)
    assert norm_entropy(np.array([1.0, 0.0, 0.0])) < 1e-6


def test_per_iteration_share_and_cumulative():
    # task A: verified trace in iteration 0 and 1; task B: only in iteration 1
    a = [make_trace(0, 0, (0,) * 5, True), make_trace(0, 1, (1,) * 5, False),
         make_trace(1, 0, (0,) * 5, True), make_trace(1, 1, (0,) * 5, True)]
    b = [make_trace(0, 0, (1,) * 5, False), make_trace(0, 1, (2,) * 5, False),
         make_trace(1, 0, (0,) * 5, True), make_trace(1, 1, (2,) * 5, False)]
    run = make_run("tot_aco", {"A": a, "B": b})
    rows = per_iteration_rows([run])
    assert rows[0]["verified_trace_share"] == (0.5 + 0.0) / 2
    assert rows[1]["verified_trace_share"] == (1.0 + 0.5) / 2
    assert rows[0]["cum_tasks_with_verified"] == 0.5 and rows[1]["cum_tasks_with_verified"] == 1.0
    # only task A had a success before iteration 2 (1-based), and it verified 100% there
    assert rows[1]["tasks_after_first_success"] == 1 and rows[1]["verified_trace_share_after_first_success"] == 1.0
    first = {r["task_id"]: r["first_verified_it"] for r in first_success_rows([run])}
    assert first == {"A": 1, "B": 2}


def test_branch_quality_and_agreement():
    # ToT-only: at level 1 branch 0 always verified, branches 1 and 2 never
    # (informative = unique best and unique worst branch: branch 1 is verified 1 of 3 times)
    tot_traces = [make_trace(0, k, (k % 3,) * 5, k % 3 == 0 or k == 1) for k in range(9)]
    tot = make_run("tot_only", {"A": tot_traces})
    q = branch_quality([tot])["A"]
    assert q[0, 0] == 1.0 and np.isclose(q[0, 1], 1 / 3) and q[0, 2] == 0.0
    # ACO pheromone concentrated on level-1 branch 0
    tau = [1.0] * 15
    tau[0] = 8.0
    aco_traces = [make_trace(0, 0, (0,) * 5, True), make_trace(1, 0, (0,) * 5, True)]
    aco = make_run("tot_aco", {"A": aco_traces}, tau_after=tau)
    rows = [r for r in agreement_rows([aco], [tot]) if r["kind"] == "final" and r["level"] == 1]
    assert len(rows) == 1
    assert rows[0]["mass_on_best_branch"] > 0.7 and rows[0]["top1_is_best"] == 1.0 and rows[0]["top1_is_worst"] == 0.0


def test_diversity_rows():
    # task A: 2 verified of 4 traces, answers "ans" only (1 distinct); task B: no verified trace
    a = [make_trace(0, k, (0,) * 5, k < 2) for k in range(4)]
    b = [make_trace(0, k, (1,) * 5, False) for k in range(4)]
    row = diversity_rows_from_runs([make_run("tot_only", {"A": a, "B": b})])[0]
    assert row["tasks"] == 2 and row["candidates_per_task"] == 4
    assert np.isclose(row["verified_share_per_candidate"], 0.25)
    assert row["tasks_with_a_verified_candidate"] == 0.5
    assert row["mean_distinct_answers_per_task"] == 1 and row["tasks_with_one_distinct_answer"] == 1.0
