from __future__ import annotations

import pytest

from swarm_planner.selection_rules import (
    FALLBACK_PREDICTION,
    HEADLINE_RULE,
    SELECTION_RULES,
    TraceRec,
    iterations_until_first_verified,
    prediction_of,
    rec_from_log,
    rec_from_trace,
    select,
    soft_ratio_of,
)
from swarm_planner.solver import trace_record
from swarm_planner.types import AgentTrace

EARLY = "Here is the proposed time: Monday, 9:30 - 10:00"
MID = "Here is the proposed time: Monday, 10:30 - 11:00"
LATE = "Here is the proposed time: Monday, 15:00 - 15:30"
VALID_BUT_WORSE = "Here is the proposed time: Monday, 12:00 - 12:30"
INVALID = "Here is the proposed time: Monday, 9:00 - 9:30"
RANKS = {EARLY: 0, MID: 1, LATE: 4, VALID_BUT_WORSE: 2, INVALID: None}


def rec(it: int, agent: int, answer: str, ps: float = 0.1) -> TraceRec:
    """A trace with a fixed verifier outcome per answer: EARLY/MID/LATE verified, VALID_BUT_WORSE
    valid with score 0.75, INVALID invalid with score 0.2."""
    if answer in (EARLY, MID, LATE):
        q, valid, soft = 1.0, True, 1.0
    elif answer == VALID_BUT_WORSE:
        q, valid, soft = 0.75, True, 0.5
    else:
        q, valid, soft = 0.2, False, 0.0
    return TraceRec(it=it, agent=agent, answer=answer, q=q, valid=valid, soft_ratio=soft,
                    rank=RANKS[answer], ps=ps)


def traces(*per_iteration):
    """traces([answers of iteration 0], [answers of iteration 1], ...)"""
    return [rec(it, agent, answer) for it, answers in enumerate(per_iteration) for agent, answer in enumerate(answers)]


def test_headline_is_verified_last3_and_all_rules_are_known():
    assert HEADLINE_RULE == "verified_last3"
    assert HEADLINE_RULE in SELECTION_RULES and "last_iteration" in SELECTION_RULES
    assert len(set(SELECTION_RULES)) == len(SELECTION_RULES)


def test_best_of_all_prefers_the_verifier_order():
    recs = traces([LATE, INVALID], [MID, VALID_BUT_WORSE], [EARLY, INVALID])
    assert select(recs, "best_of_all").answer == EARLY
    # the same answer in two iterations: the earlier iteration wins the tie
    recs = traces([EARLY], [EARLY])
    assert select(recs, "best_of_all").it == 0


def test_windows_are_counted_in_iterations():
    recs = traces([EARLY], [INVALID], [MID], [LATE], [VALID_BUT_WORSE], [INVALID])  # 6 iterations
    assert select(recs, "last_iteration").answer == INVALID
    assert select(recs, "best_of_last2").answer == VALID_BUT_WORSE  # iterations 5-6
    assert select(recs, "best_of_last3").answer == LATE  # iterations 4-6 (LATE is verified)
    assert select(recs, "best_of_all").answer == EARLY


def test_verified_last_k_picks_the_earliest_verified_slot_in_the_window():
    recs = traces([EARLY], [VALID_BUT_WORSE], [VALID_BUT_WORSE], [LATE, MID], [LATE], [VALID_BUT_WORSE])
    # window = iterations 4-6, verified there: LATE, MID, LATE -> the earliest slot is MID (EARLY is outside)
    assert select(recs, "verified_last3").answer == MID
    # window = iterations 5-6: only LATE is verified
    assert select(recs, "verified_last2").answer == LATE


def test_verified_vote_picks_the_most_frequent_answer_and_breaks_ties_by_earliest_slot():
    recs = traces([INVALID], [INVALID], [INVALID], [LATE, LATE, MID], [LATE], [INVALID])
    # votes in iterations 4-6: LATE 3, MID 1 -> frequency wins although MID is earlier
    assert select(recs, "verified_vote_last3").answer == LATE
    assert select(recs, "verified_last3").answer == MID  # earliest-slot tiebreak
    # equal votes -> the earlier slot
    recs = traces([INVALID], [INVALID], [INVALID], [LATE, MID], [LATE, MID], [INVALID])
    assert select(recs, "verified_vote_last3").answer == MID


def test_verified_rules_fall_back_to_the_best_trace_of_the_whole_run():
    recs = traces([VALID_BUT_WORSE], [INVALID], [INVALID], [INVALID], [INVALID], [INVALID])
    for rule in ("verified_last3", "verified_last2", "verified_vote_last3", "verified_vote_last2"):
        assert select(recs, rule).answer == VALID_BUT_WORSE, rule
    # a verified answer outside the window is also found by the fallback
    recs = traces([MID], [INVALID], [INVALID], [INVALID], [INVALID], [INVALID])
    assert select(recs, "verified_last3").answer == MID


def test_majority_vote_ignores_the_verifier_and_first_seen_wins_ties():
    recs = traces([INVALID, INVALID, EARLY], [INVALID, EARLY, EARLY], [LATE])
    assert select(recs, "majority_vote").answer == INVALID  # 3 vs 3 vs 1: first seen
    recs = traces([INVALID, EARLY], [INVALID], [INVALID, EARLY])
    assert select(recs, "majority_vote").answer == INVALID  # a wrong answer can win: no verifier


def test_early_stop_uses_the_iterations_up_to_the_first_verified_trace():
    recs = traces([INVALID], [VALID_BUT_WORSE], [LATE], [EARLY], [INVALID], [INVALID])
    assert iterations_until_first_verified(recs) == 3
    assert select(recs, "early_stop").answer == LATE  # EARLY comes after the stop
    none_verified = traces([INVALID], [VALID_BUT_WORSE], [INVALID])
    assert iterations_until_first_verified(none_verified) == 3
    assert select(none_verified, "early_stop").answer == VALID_BUT_WORSE


def test_empty_input_and_unknown_rule():
    assert select([], HEADLINE_RULE) is None
    assert prediction_of(None) == FALLBACK_PREDICTION
    with pytest.raises(ValueError):
        select(traces([EARLY]), "nope")


def test_soft_ratio():
    assert soft_ratio_of(False, 3, 3) == 0.0  # an unparsed answer meets nothing
    assert soft_ratio_of(True, 1, 2) == 0.5
    assert soft_ratio_of(None, 0, 0) == 1.0  # no soft preference at all


def test_live_and_logged_records_are_identical():
    """The solver applies the headline rule to live traces; derive applies every rule to the log.
    Both must see the same records."""
    details = {
        "candidate_parsed": True, "hard_valid": True, "hard_checks_passed": 5,
        "soft_satisfied": 1, "soft_total": 2, "slot_rank": 3,
    }
    trace = AgentTrace(agent_id=4, path=[], thought_results=[], final_answer=LATE,
                       path_score=0.123456789123, quality_score=0.75, quality_details=details)
    live = rec_from_trace(2, trace)
    logged = rec_from_log(2, trace_record(trace))
    assert live == logged
    assert live.key() == logged.key()
