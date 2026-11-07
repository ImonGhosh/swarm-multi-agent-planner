"""Deterministic task-based quality scoring for calendar scheduling.

A candidate answer is judged in strict priority order:

1. hard constraints (allowed day, duration, work window, nobody busy, no "can not meet");
2. soft preferences ("would like to avoid", "would rather not", "do not want to");
3. "earliest availability", judged among the slots that are best on the soft preferences.

Score ranges: an answer that fails a hard check scores at most 0.49 * 4/5 = 0.392, a
hard-valid answer scores 0.5 to 1.0. The two ranges never overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .parser import (
    DAY_TO_INDEX,
    SLOT_STEP_MINUTES,
    Preference,
    minutes_to_time,
    parse_proposed_time,
    parse_task,
    time_to_minutes,
)

Slot = Tuple[str, int, int]

# Deposit rules (see deposit_quality). "top_only" is the frozen v2 rule (aco/pheromone.py).
DEPOSIT_MODES = ("v1_score", "valid_only", "top_only")


@dataclass(frozen=True)
class DeterministicQualityResult:
    score: float
    candidate_parsed: bool
    hard_valid: bool
    hard_checks_passed: int
    hard_checks_total: int
    soft_satisfied: int
    soft_total: int
    soft_satisfaction_ratio: float
    # Position of the candidate in the verifier's total order of hard-valid slots
    # (0 = best slot: most soft preferences met, then earliest). None if not a hard-valid slot.
    slot_rank: int | None
    best_slot: str | None
    candidate_slot: str | None

    def to_dict(self) -> Dict[str, object]:
        return {
            "score": self.score,
            "candidate_parsed": self.candidate_parsed,
            "hard_valid": self.hard_valid,
            "hard_checks_passed": self.hard_checks_passed,
            "hard_checks_total": self.hard_checks_total,
            "soft_satisfied": self.soft_satisfied,
            "soft_total": self.soft_total,
            "soft_satisfaction_ratio": self.soft_satisfaction_ratio,
            "slot_rank": self.slot_rank,
            "best_slot": self.best_slot,
            "candidate_slot": self.candidate_slot,
        }


def deposit_quality(result: DeterministicQualityResult, mode: str = "valid_only") -> float:
    """Map a verifier result to the quality Q a trace deposits as pheromone.

    * ``v1_score``: deposit the raw score. Invalid answers still deposit up to 0.392, so
      pheromone also flows to paths that produced wrong answers (finding F7).
    * ``valid_only``: deposit the score only for hard-valid answers and 0 otherwise.
    * ``top_only``: deposit 1 for an answer scoring 1.0 and 0 otherwise. This is the v2 rule
      that ``aco.pheromone.apply_pheromone_update`` applies; the other two modes are the
      alternatives compared offline in Phase 3.
    """
    if mode == "v1_score":
        return float(result.score)
    if mode == "valid_only":
        return float(result.score) if result.hard_valid else 0.0
    if mode == "top_only":
        return 1.0 if float(result.score) >= 1.0 else 0.0
    raise ValueError(f"Unknown deposit mode {mode!r}; expected one of {DEPOSIT_MODES}")


def _overlaps(interval: Tuple[int, int], blocked: Tuple[int, int]) -> bool:
    return interval[0] < blocked[1] and interval[1] > blocked[0]


def _violates_preference(preference: Preference, day: str, start: int, end: int) -> bool:
    if day not in preference.days:
        return False
    if preference.bound is None or preference.bound_minutes is None:
        return True
    if preference.bound == "before":
        return start < preference.bound_minutes
    if preference.bound == "after":
        return end > preference.bound_minutes
    return False


def _format_slot(day: str, start: int, end: int) -> str:
    return f"{day}, {minutes_to_time(start)} - {minutes_to_time(end)}"


def _dedupe_and_sort_slots(slots: Iterable[Slot]) -> List[Slot]:
    unique = {(day, start, end) for day, start, end in slots}
    return sorted(
        unique,
        key=lambda slot: (DAY_TO_INDEX.get(slot[0], 999), slot[1], slot[2]),
    )


class DeterministicTaskQualityScorer:
    """Scores calendar predictions without golden labels or LLM calls."""

    def __init__(self, task_text: str) -> None:
        self.task = parse_task(task_text)
        # Chronological list of every slot that passes all hard checks.
        self.valid_slots = self._enumerate_hard_valid_slots()
        # Same slots, best first: most soft preferences met, then earliest.
        self.ranked_slots = sorted(
            self.valid_slots,
            key=lambda slot: (-self._soft_met(slot), DAY_TO_INDEX.get(slot[0], 999), slot[1], slot[2]),
        )
        self._rank_of: Dict[Slot, int] = {slot: i for i, slot in enumerate(self.ranked_slots)}

    def _soft_met(self, slot: Slot) -> int:
        return sum(
            1 for pref in self.task.soft_preferences if not _violates_preference(pref, *slot)
        )

    def _passes_hard_preferences(self, day: str, start: int, end: int) -> bool:
        return not any(
            _violates_preference(pref, day, start, end)
            for pref in self.task.hard_preferences
        )

    def _passes_busy(self, day: str, start: int, end: int) -> bool:
        meeting = (start, end)
        for person in self.task.participants:
            for blocked in self.task.busy_by_participant_by_day.get(person, {}).get(day, []):
                if _overlaps(meeting, blocked):
                    return False
        return True

    def _enumerate_hard_valid_slots(self) -> List[Slot]:
        slots: List[Slot] = []
        for day in self.task.allowed_days:
            start = self.task.work_start_minutes
            last_start = self.task.work_end_minutes - self.task.duration_minutes
            while start <= last_start:
                end = start + self.task.duration_minutes
                if self._passes_busy(day, start, end) and self._passes_hard_preferences(day, start, end):
                    slots.append((day, start, end))
                start += SLOT_STEP_MINUTES
        return _dedupe_and_sort_slots(slots)

    @property
    def best_slot(self) -> Slot | None:
        """The slot the verifier ranks first (what 'verifier-as-solver' answers)."""
        return self.ranked_slots[0] if self.ranked_slots else None

    def top_slots(self) -> List[Slot]:
        """Every hard-valid slot that attains the highest score, in chronological order.

        More than one means the task has several equally good answers. When the task asks for
        the earliest availability this is a single slot. Natural Plan's exact-match metric
        still accepts only the gold one of them.
        """
        scored = [(slot, self.score_prediction(_format_answer(slot)).score) for slot in self.valid_slots]
        if not scored:
            return []
        top = max(score for _, score in scored)
        return [slot for slot, score in scored if score == top]

    def score_prediction(self, prediction: str) -> DeterministicQualityResult:
        best_slot = _format_slot(*self.best_slot) if self.best_slot else None
        parsed = parse_proposed_time(prediction)
        if not parsed:
            return DeterministicQualityResult(
                score=0.0,
                candidate_parsed=False,
                hard_valid=False,
                hard_checks_passed=0,
                hard_checks_total=5,
                soft_satisfied=0,
                soft_total=0,
                soft_satisfaction_ratio=0.0,
                slot_rank=None,
                best_slot=best_slot,
                candidate_slot=None,
            )

        day, start_text, end_text = parsed
        day = day.capitalize()
        start = time_to_minutes(start_text)
        end = time_to_minutes(end_text)
        candidate_slot = _format_slot(day, start, end)

        check_allowed_day = day in self.task.allowed_days
        check_duration = (end - start) == self.task.duration_minutes
        check_work_window = (
            start >= self.task.work_start_minutes and end <= self.task.work_end_minutes
        )
        check_busy = self._passes_busy(day, start, end)
        check_hard_prefs = self._passes_hard_preferences(day, start, end)

        hard_checks = [
            check_allowed_day,
            check_duration,
            check_work_window,
            check_busy,
            check_hard_prefs,
        ]
        hard_checks_passed = sum(1 for check in hard_checks if check)
        hard_checks_total = len(hard_checks)
        hard_valid = hard_checks_passed == hard_checks_total

        slot_rank = self._rank_of.get((day, start, end))

        soft_total = len(self.task.soft_preferences)
        soft_satisfied = self._soft_met((day, start, end))

        if self.task.earliest_requested:
            soft_total += 1
            if slot_rank == 0:
                soft_satisfied += 1

        soft_ratio = (soft_satisfied / float(soft_total)) if soft_total else 1.0
        if hard_valid:
            score = 0.5 + (0.5 * soft_ratio)
        else:
            hard_ratio = hard_checks_passed / float(hard_checks_total)
            score = 0.49 * hard_ratio

        return DeterministicQualityResult(
            score=round(float(score), 6),
            candidate_parsed=True,
            hard_valid=hard_valid,
            hard_checks_passed=hard_checks_passed,
            hard_checks_total=hard_checks_total,
            soft_satisfied=soft_satisfied,
            soft_total=soft_total,
            soft_satisfaction_ratio=round(float(soft_ratio), 6),
            slot_rank=slot_rank,
            best_slot=best_slot,
            candidate_slot=candidate_slot,
        )


def _format_answer(slot: Slot) -> str:
    return f"Here is the proposed time: {_format_slot(*slot)}"


def build_task_quality_scorer(task_text: str) -> DeterministicTaskQualityScorer:
    return DeterministicTaskQualityScorer(task_text)
