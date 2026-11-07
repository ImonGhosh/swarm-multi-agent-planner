"""The offline simulator, checked on a table whose answers are known exactly.

The synthetic table comes from the real enumerator with a fake LLM that answers correctly only
when level 1 chose ``people_first``. One third of the 243 paths are correct, and the correct ones
are exactly the paths that start with branch 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fakes import CANDIDATE, FakeClient
from swarm_planner.path_table import enumerate_dataset
from swarm_planner.simulate import (
    DEPOSIT_RULES,
    SHAPE,
    TaskTable,
    _selection_key,
    bootstrap_ci,
    load_path_table,
    simulate,
    simulate_task,
)
from test_solver import TASK_TEXT, _dataset

WRONG = "Here is the proposed time: Monday, 9:00 - 9:30"


def _responder(messages):
    if "Produce one final answer" in messages[0]["content"]:
        return CANDIDATE if "Extract participants first" in messages[1]["content"] else WRONG
    return "intermediate"


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> list[TaskTable]:
    tmp = tmp_path_factory.mktemp("sim")
    data = _dataset(TASK_TEXT, n_tasks=1)
    data_path = tmp / "dev.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    enumerate_dataset(
        data_path=str(data_path),
        out_dir=str(tmp / "table"),
        model="dummy-model",
        cache_path=None,
        client=FakeClient(_responder),
    )
    return load_path_table(tmp / "table", data_path)


def test_load_path_table_reads_every_path(tables):
    (table,) = tables
    assert table.category == "2p1d" and table.task_id == "t1"
    assert table.correct.shape == SHAPE == (3, 3, 3, 3, 3)
    assert int(table.correct.sum()) == 81
    assert table.correct[1].all() and not table.correct[0].any() and not table.correct[2].any()
    assert (table.score[1] == 1.0).all() and (table.score[0] < 0.5).all()
    assert (table.rank[1] == 0).all() and (table.rank[0] >= 1_000_000).all()
    assert table.eta.shape == (5, 3) and set(np.unique(table.eta)) <= {1.0, 1.2, 1.5}


def test_tot_only_matches_the_closed_form(tables):
    # a correct answer exists in a draw of k paths with probability 1 - (2/3)^k, and the verifier selects it
    last = simulate(tables, reps=3000, seed=1, n_agents=8, n_iterations=6)
    assert last["last_correct"][0] == pytest.approx(1 - (2 / 3) ** 8, abs=0.02)
    assert last["all_correct"][0] == pytest.approx(1 - (2 / 3) ** 48, abs=0.005)
    assert last["any_correct"][0] == pytest.approx(1 - (2 / 3) ** 48, abs=0.005)


def test_single_agent_single_iteration_is_one_random_path(tables):
    r = simulate(tables, reps=4000, seed=2, n_agents=1, n_iterations=1)
    assert r["last_correct"][0] == pytest.approx(1 / 3, abs=0.03)


def test_tot_only_probabilities_stay_uniform(tables):
    run = simulate_task(tables[0], n_agents=4, n_iterations=3, rng=np.random.default_rng(0))
    assert all(np.allclose(p, 1 / 3) for p in run["level1_probabilities"])


def test_informative_deposits_move_the_pheromone_toward_the_right_branch(tables):
    rng = np.random.default_rng(3)

    def final_probability(**kwargs):
        runs = [simulate_task(tables[0], n_agents=8, n_iterations=6, rng=rng, **kwargs) for _ in range(150)]
        return np.mean([r["level1_probabilities"][-1][1] for r in runs])

    uniform_start = 1 / 3
    assert final_probability(alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"]) > 0.6
    assert final_probability(alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["valid_only"]) > 0.6
    # without any pheromone influence nothing moves
    assert final_probability(alpha=0.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"]) == pytest.approx(uniform_start)


def test_deposit_rules(tables):
    table = tables[0]
    paths = [(1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (2, 1, 1, 1, 1)]  # right, wrong, wrong
    scores = np.array([table.score[p] for p in paths])
    assert list(DEPOSIT_RULES["v1_score"](table, paths, scores)) == list(scores)
    assert scores[1] > 0  # v1 lets a wrong answer deposit (F7)
    assert list(DEPOSIT_RULES["valid_only"](table, paths, scores)) == [1.0, 0.0, 0.0]
    assert list(DEPOSIT_RULES["top_only"](table, paths, scores)) == [1.0, 0.0, 0.0]
    assert list(DEPOSIT_RULES["best_of_iteration"](table, paths, scores)) == [1.0, 0.0, 0.0]
    assert DEPOSIT_RULES["score_minus_mean"](table, paths, scores)[0] > 0
    assert (DEPOSIT_RULES["score_minus_mean"](table, paths, scores) >= 0).all()


def test_early_stopping_ends_after_the_first_iteration_with_a_top_answer(tables):
    full = simulate(tables, reps=500, seed=4, n_agents=8, n_iterations=6)
    early = simulate(tables, reps=500, seed=4, n_agents=8, n_iterations=6, early_stop=True)
    # an iteration contains a top answer with probability q = 1 - (2/3)^8, so the mean is about 1/q
    assert early["iterations_used"][0] == pytest.approx(1 / (1 - (2 / 3) ** 8), abs=0.05)
    assert early["all_correct"][0] == pytest.approx(full["all_correct"][0], abs=0.02)
    assert full["iterations_used"][0] == 6


def test_selection_key_matches_the_solvers_order(tables):
    """The simulator's key (score, hard-valid, soft ratio, earliest rank, path score, lowest agent id)
    is the key of selection_rules.TraceRec, which the solver and the offline derivation use."""
    from swarm_planner.selection_rules import TraceRec

    table = tables[0]
    candidates = [(1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (2, 2, 2, 2, 2)]
    for path_score, agent in ((0.01, 3), (0.5, 0), (0.2, 7)):
        for path in candidates:
            ours = _selection_key(table, path, path_score, agent)
            rec = TraceRec(
                it=0, agent=agent, answer="", q=float(table.score[path]), valid=bool(table.valid[path]),
                soft_ratio=float(table.soft_ratio[path]),
                rank=None if table.rank[path] >= 1_000_000 else int(table.rank[path]), ps=path_score,
            )
            theirs = rec.key()  # (..., path score, -iteration, -agent)
            assert [float(x) for x in ours[:5]] == [float(x) for x in theirs[:5]]
            assert float(ours[5]) == float(theirs[6])


def test_aco_visibly_changes_a_full_budget_run_but_not_the_best_of_all_answer(tables):
    """The frozen v2 method (no eta, top-only deposit) against ToT-only, full budget.

    Only the people_first branch is correct, so the verified share of a ToT-only iteration stays at 1/3.
    ACO learns it: the share rises, the last iteration and a majority vote get better, while the best
    answer over all iterations is the same (it is secured at the first success).
    """
    kwargs = dict(reps=1200, seed=6, n_agents=8, n_iterations=6)
    tot = simulate(tables, **kwargs)
    aco = simulate(tables, alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"], **kwargs)

    assert tot["top_share_first"][0] == pytest.approx(1 / 3, abs=0.03)
    assert tot["top_share_last"][0] == pytest.approx(1 / 3, abs=0.03)  # ToT-only never improves
    assert aco["top_share_first"][0] == pytest.approx(1 / 3, abs=0.03)  # uniform until the first deposit
    assert aco["top_share_last"][0] > 0.6
    assert aco["top_share_all"][0] > tot["top_share_all"][0] + 0.1
    assert aco["last_correct"][0] > tot["last_correct"][0]
    # majority vote: the wrong answer wins 2 times out of 3 for ToT-only, and ACO flips that
    assert aco["majority_correct"][0] > tot["majority_correct"][0] + 0.3
    assert aco["all_correct"][0] == pytest.approx(tot["all_correct"][0], abs=0.01)


def test_best_of_the_last_k_iterations_is_monotone_between_last_and_all(tables):
    r = simulate(tables, reps=800, seed=8, n_agents=8, n_iterations=6)
    one, two, three, everything = (r[k][0] for k in ("last_correct", "last2_correct", "last3_correct", "all_correct"))
    assert one <= two + 0.03 and two <= three + 0.03 and three <= everything + 0.03
    # one iteration of 8 agents, then 2 and 3: 1 - (2/3)^(8k)
    assert two == pytest.approx(1 - (2 / 3) ** 16, abs=0.03)
    assert simulate(tables, reps=200, seed=1, n_agents=8, n_iterations=1)["last2_correct"][0] == pytest.approx(
        simulate(tables, reps=200, seed=1, n_agents=8, n_iterations=1)["last_correct"][0]
    )  # with one iteration every window is that iteration


def test_verified_vote_in_the_synthetic_table(tables):
    """Only people_first paths are correct and verified, every other answer is the same wrong one.

    Plain majority: the wrong answer has 2/3 of the votes, so ToT-only loses and ACO (which learns people_first) wins
    more often. Verified vote: only correct answers may vote, so any verified trace in the window decides.
    """
    kwargs = dict(reps=1500, seed=12, n_agents=8, n_iterations=6)
    tot = simulate(tables, **kwargs)
    aco = simulate(tables, alpha=1.0, beta=0.0, deposit=DEPOSIT_RULES["top_only"], **kwargs)
    assert tot["majority_correct"][0] < 0.1
    # a verified vote over a window is right whenever the window has a verified trace: 1 - (2/3)^(8k)
    assert tot["vote_verified_last3_correct"][0] == pytest.approx(1 - (2 / 3) ** 24, abs=0.01)
    assert tot["vote_verified_last2_correct"][0] == pytest.approx(1 - (2 / 3) ** 16, abs=0.02)
    assert tot["vote_verified_all_correct"][0] == pytest.approx(1 - (2 / 3) ** 48, abs=0.005)
    # here the hard-valid traces are exactly the verified ones, so the two votes agree
    assert tot["vote_valid_last3_correct"][0] == pytest.approx(tot["vote_verified_last3_correct"][0], abs=0.001)
    # ACO has more verified traces in the window, so it is at least as good
    assert aco["vote_verified_last2_correct"][0] >= tot["vote_verified_last2_correct"][0] - 0.01


def test_a_vote_counts_traces_not_distinct_answers(tables):
    """Two answers: 'A' on one path, 'B' on two paths. 'B' must win although both are one distinct answer."""
    table = tables[0]
    # monkeypatch-free check through the public result: rebuild a tiny table with chosen answer ids
    import dataclasses

    small = dataclasses.replace(table, answer_id=np.zeros(SHAPE, dtype=int))
    small.answer_id[1] = 1  # every path that starts with branch 1 is answer 1 (and correct); the rest answer 0
    runs = [simulate_task(small, n_agents=8, n_iterations=6, rng=np.random.default_rng(s)) for s in range(200)]
    for r in runs:
        counts = {}
        for p in r["paths"]:
            counts[int(small.answer_id[p])] = counts.get(int(small.answer_id[p]), 0) + 1
        winner = max(counts, key=lambda a: counts[a])
        assert r["majority_correct"] == any(bool(small.correct[p]) for p in r["paths"] if int(small.answer_id[p]) == winner)


def test_simulation_returns_every_sampled_path(tables):
    run = simulate_task(tables[0], n_agents=8, n_iterations=6, rng=np.random.default_rng(0))
    assert len(run["paths"]) == 48 and all(len(p) == 5 for p in run["paths"])


def test_bootstrap_ci():
    constant = np.full(20, 0.4)
    assert bootstrap_ci(constant) == (pytest.approx(0.4), pytest.approx(0.4))
    values = np.random.default_rng(0).random(30)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high
