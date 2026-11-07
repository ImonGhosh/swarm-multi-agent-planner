from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from swarm_planner.metrics import correctness, exact_match, parse_response, solve_rate
from swarm_planner.stats import bootstrap_ci, discordant, mcnemar_exact, mean_sd

OFFICIAL = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"

GOLD = "Here is the proposed time: Monday, 14:30 - 15:30 "
RESPONSES = [
    "Here is the proposed time: Monday, 14:30 - 15:30",      # match
    "Here is the proposed time: Monday, 14:30 - 16:00",      # wrong end
    "Here is the proposed time: Tuesday, 14:30 - 15:30",     # wrong day
    "I think Monday, 14:30 - 15:30 works. Or Tuesday, 9:00 - 9:30",  # first time in the text counts
    "Monday 14:30 to 15:30",                                  # unparseable for the official regex
    "",
    "Here is the proposed time: monday, 14:30 - 15:30",      # day is compared case-sensitively
    "Here is the proposed time: Monday, 14:00 - 15:30",
]


def test_exact_match_equals_the_official_evaluator():
    spec = importlib.util.spec_from_file_location("official_evaluate", OFFICIAL)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    golds = [GOLD] * len(RESPONSES)
    assert solve_rate(RESPONSES, golds) == official.compute_solve_rate(RESPONSES, golds)
    for response in RESPONSES:
        assert official._parse_response(response) == parse_response(response)
    assert correctness(RESPONSES, golds) == [True, False, False, True, False, False, False, False]


def test_unparseable_prediction_never_matches_a_parseable_gold():
    assert not exact_match("no slot here", GOLD)
    assert parse_response("no slot here") == ("", -1, -1)


def test_mean_sd_uses_the_sample_sd():
    assert mean_sd([0.4, 0.5, 0.6]) == pytest.approx((0.5, 0.1))
    assert mean_sd([0.3]) == (0.3, 0.0)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(10, 2) == pytest.approx(mcnemar_exact(2, 10))
    assert 0.0 < mcnemar_exact(10, 2) < 0.05


def test_discordant_pairs():
    a = np.array([1, 1, 0, 0, 1])
    b = np.array([1, 0, 1, 0, 0])
    assert discordant(a, b) == (2, 1)


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_more_tasks():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 2, 40).astype(float)
    large = rng.integers(0, 2, 4000).astype(float)
    for values in (small, large):
        low, high = bootstrap_ci(values)
        assert low <= values.mean() <= high
    assert (bootstrap_ci(large)[1] - bootstrap_ci(large)[0]) < (bootstrap_ci(small)[1] - bootstrap_ci(small)[0])
