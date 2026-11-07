"""Final-prediction selection rules (decision 11, PHASE4.md D2).

A search leaves up to n x T traces (48 with the default budget). A *rule* picks the one answer
that is submitted. The rule is separate from the quality score: the score drives the pheromone
during the search, the rule only chooses what is reported.

Every rule works on ``TraceRec`` records, which are built either from live ``AgentTrace`` objects
(the solver applies the headline rule) or from the ``iteration`` events of a finished run's log
(``scripts/derive_predictions.py`` applies all rules offline). Both give the same answers.

Ordering of traces (the solver's own key, extended across iterations): verifier score, hard-valid,
share of soft preferences met, earlier slot (lower verifier rank), higher path probability, then
the earlier iteration and the lower agent id.

Rules
-----
``last_iteration``       best trace of the last iteration only (v1; comparable with the paper)
``best_of_all``          best trace of all iterations
``best_of_last2/3``      best trace of the last 2 / 3 iterations
``verified_last3/2``     HEADLINE (last3). Only traces the verifier scores 1.0, from the last 3 / 2
                         iterations, may be chosen. The earliest verified slot wins. If no trace in the
                         window is verified: the best trace of the whole run.
``verified_vote_last3/2`` The same candidates, but the answer most traces ended on wins (a vote counts
                         traces per answer). Ties go to the earlier slot. Same fallback.
``majority_vote``        plain majority over all traces, the verifier is not involved (ties: first seen)
``early_stop``           best of the iterations up to the first one that has a verified trace
                         (what a search that stops at the first success would answer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

FALLBACK_PREDICTION = "Here is the proposed time: Monday, 9:00 - 9:30"
NO_RANK = 1_000_000
VERIFIED = 1.0

HEADLINE_RULE = "verified_last3"
LAST_ITERATION_RULE = "last_iteration"

SELECTION_RULES: Tuple[str, ...] = (
    "verified_last3",
    "verified_last2",
    "verified_vote_last3",
    "verified_vote_last2",
    "best_of_all",
    "best_of_last3",
    "best_of_last2",
    "last_iteration",
    "majority_vote",
    "early_stop",
)

RULE_DESCRIPTIONS: Dict[str, str] = {
    "verified_last3": "verified traces of the last 3 iterations, earliest verified slot (HEADLINE)",
    "verified_last2": "verified traces of the last 2 iterations, earliest verified slot",
    "verified_vote_last3": "verified traces of the last 3 iterations, most frequent answer",
    "verified_vote_last2": "verified traces of the last 2 iterations, most frequent answer",
    "best_of_all": "best trace of all iterations",
    "best_of_last3": "best trace of the last 3 iterations",
    "best_of_last2": "best trace of the last 2 iterations",
    "last_iteration": "best trace of the last iteration (v1 rule)",
    "majority_vote": "plain majority vote over all traces (no verifier)",
    "early_stop": "best trace up to the first iteration with a verified trace",
}


@dataclass(frozen=True)
class TraceRec:
    """One trace with everything the rules need (no prompt or response text)."""

    it: int  # iteration index, 0-based
    agent: int
    answer: str
    q: float  # verifier score
    valid: bool  # hard-valid
    soft_ratio: float
    rank: int | None  # verifier rank of the proposed slot (0 = best), None if not a hard-valid slot
    ps: float  # path probability

    @property
    def verified(self) -> bool:
        return self.q >= VERIFIED

    def key(self) -> tuple:
        rank = self.rank if isinstance(self.rank, int) else NO_RANK
        return (self.q, 1.0 if self.valid else 0.0, self.soft_ratio, -rank, self.ps, -self.it, -self.agent)


def soft_ratio_of(parsed: Any, soft: Any, soft_total: Any) -> float:
    """Share of soft preferences met. 0 for an answer that could not be parsed, 1 if none exist."""
    if parsed is False:
        return 0.0
    total = int(soft_total or 0)
    return (int(soft or 0) / total) if total else 1.0


def rec_from_trace(it: int, trace: Any) -> TraceRec:
    """Build a record from a live ``AgentTrace`` that has been scored by the verifier."""
    d = trace.quality_details if isinstance(trace.quality_details, dict) else {}
    rank = d.get("slot_rank")
    return TraceRec(
        it=it,
        agent=int(trace.agent_id),
        answer=trace.final_answer,
        q=float(trace.quality_score),
        valid=bool(d.get("hard_valid", False)),
        soft_ratio=soft_ratio_of(d.get("candidate_parsed"), d.get("soft_satisfied"), d.get("soft_total")),
        rank=int(rank) if isinstance(rank, int) else None,
        ps=round(float(trace.path_score), 8),
    )


def rec_from_log(it: int, record: Dict[str, Any]) -> TraceRec:
    """Build a record from one trace of a logged ``iteration`` event (see solver.trace_record)."""
    v = record.get("v") or {}
    rank = v.get("rank")
    return TraceRec(
        it=it,
        agent=int(record["agent"]),
        answer=record.get("ans", ""),
        q=float(record.get("q", 0.0)),
        valid=bool(v.get("valid", False)),
        soft_ratio=soft_ratio_of(v.get("parsed"), v.get("soft"), v.get("soft_total")),
        rank=int(rank) if isinstance(rank, int) else None,
        ps=round(float(record.get("ps", 0.0)), 8),
    )


def _best(recs: Iterable[TraceRec]) -> TraceRec | None:
    recs = list(recs)
    return max(recs, key=TraceRec.key) if recs else None


def _window(recs: Sequence[TraceRec], k: int) -> List[TraceRec]:
    """Traces of the last k iterations (counted in iterations, not traces)."""
    if not recs:
        return []
    last = max(r.it for r in recs)
    return [r for r in recs if r.it > last - k]


def _rank_key(rec: TraceRec) -> int:
    return rec.rank if isinstance(rec.rank, int) else NO_RANK


def _verified_pick(recs: Sequence[TraceRec], k: int, *, vote: bool) -> TraceRec | None:
    candidates = [r for r in _window(recs, k) if r.verified]
    if not candidates:
        return _best(recs)  # fallback: the best trace of the whole run
    if not vote:
        return _best(candidates)  # all score 1.0, so the key picks the earliest slot
    counts: Dict[str, int] = {}
    for r in candidates:
        counts[r.answer] = counts.get(r.answer, 0) + 1
    top = max(counts.values())
    leaders = [r for r in candidates if counts[r.answer] == top]
    # ties between answers with the same number of votes go to the earlier slot (lower rank)
    earliest = min(_rank_key(r) for r in leaders)
    return _best(r for r in leaders if _rank_key(r) == earliest)


def _majority_pick(recs: Sequence[TraceRec]) -> TraceRec | None:
    if not recs:
        return None
    counts: Dict[str, int] = {}
    first: Dict[str, TraceRec] = {}
    for r in sorted(recs, key=lambda r: (r.it, r.agent)):
        counts[r.answer] = counts.get(r.answer, 0) + 1
        first.setdefault(r.answer, r)
    # first seen wins ties (dicts keep insertion order, max returns the first maximum)
    winner = max(counts, key=lambda a: counts[a])
    return first[winner]


def iterations_until_first_verified(recs: Sequence[TraceRec]) -> int:
    """Iterations an early-stopping search would have used (all of them if none is verified)."""
    if not recs:
        return 0
    last = max(r.it for r in recs)
    for it in range(last + 1):
        if any(r.verified for r in recs if r.it == it):
            return it + 1
    return last + 1


def select(recs: Sequence[TraceRec], rule: str) -> TraceRec | None:
    """The chosen trace under ``rule`` (None only if there are no traces at all)."""
    recs = list(recs)
    if not recs:
        return None
    if rule == "last_iteration":
        return _best(_window(recs, 1))
    if rule == "best_of_all":
        return _best(recs)
    if rule in ("best_of_last2", "best_of_last3"):
        return _best(_window(recs, int(rule[-1])))
    if rule in ("verified_last2", "verified_last3"):
        return _verified_pick(recs, int(rule[-1]), vote=False)
    if rule in ("verified_vote_last2", "verified_vote_last3"):
        return _verified_pick(recs, int(rule[-1]), vote=True)
    if rule == "majority_vote":
        return _majority_pick(recs)
    if rule == "early_stop":
        used = iterations_until_first_verified(recs)
        return _best(r for r in recs if r.it < used)
    raise ValueError(f"Unknown selection rule {rule!r}; expected one of {SELECTION_RULES}")


def prediction_of(rec: TraceRec | None) -> str:
    return rec.answer if rec is not None and rec.answer else FALLBACK_PREDICTION
