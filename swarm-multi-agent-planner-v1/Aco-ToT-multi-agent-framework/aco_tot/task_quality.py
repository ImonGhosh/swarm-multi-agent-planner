"""Deterministic task-based quality scoring for calendar scheduling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .prompt_io import parse_proposed_time


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_TO_INDEX = {day: idx for idx, day in enumerate(DAY_ORDER)}
DAY_RE = r"(Monday|Tuesday|Wednesday|Thursday|Friday)"
SLOT_STEP_MINUTES = 30


@dataclass(frozen=True)
class Preference:
    subject: str
    kind: str
    days: Tuple[str, ...]
    bound: str | None
    bound_minutes: int | None
    hard: bool


@dataclass(frozen=True)
class ParsedTask:
    participants: Tuple[str, ...]
    duration_minutes: int
    work_start_minutes: int
    work_end_minutes: int
    allowed_days: Tuple[str, ...]
    busy_by_participant_by_day: Dict[str, Dict[str, List[Tuple[int, int]]]]
    soft_preferences: Tuple[Preference, ...]
    hard_preferences: Tuple[Preference, ...]
    earliest_requested: bool


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
    slot_rank: int | None
    earliest_slot: str | None
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
            "earliest_slot": self.earliest_slot,
            "candidate_slot": self.candidate_slot,
        }


def _normalize_text(task_text: str) -> str:
    text = task_text or ""
    text = re.sub(r"\bearlist\b", "earliest", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Z][a-z]+)has\b", r"\1 has", text)
    text = re.sub(r"\b([A-Z][a-z]+)is\b", r"\1 is", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def _time_to_minutes(time_text: str) -> int:
    hour_text, minute_text = time_text.split(":")
    return int(hour_text) * 60 + int(minute_text)


def _minutes_to_time(minutes: int) -> str:
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour}:{minute:02d}"


def _extract_task_header(task_text: str) -> tuple[Tuple[str, ...], int, int, int, Tuple[str, ...]]:
    header_re = re.compile(
        r"TASK:\s*You need to schedule a meeting for (?P<participants>.+?) "
        r"for (?P<duration>half an hour|one hour) "
        r"between the work hours of (?P<work_start>[0-9]{1,2}:[0-9]{2}) to (?P<work_end>[0-9]{1,2}:[0-9]{2}) "
        r"on (?P<days>.+?)\.",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = header_re.search(task_text)
    if not match:
        raise ValueError("Unable to parse task header.")

    participants_raw = re.sub(r"\s+and\s+", ", ", match.group("participants").strip())
    participants = tuple(
        token.strip()
        for token in participants_raw.split(",")
        if token and token.strip()
    )

    duration_text = match.group("duration").lower().strip()
    duration_minutes = 30 if "half" in duration_text else 60
    work_start = _time_to_minutes(match.group("work_start"))
    work_end = _time_to_minutes(match.group("work_end"))

    days_phrase = match.group("days").strip()
    days_phrase = re.sub(r"^either\s+", "", days_phrase, flags=re.IGNORECASE)
    days_phrase = re.sub(r"\s+or\s+", ", ", days_phrase, flags=re.IGNORECASE)
    allowed_days = tuple(
        day
        for day in DAY_ORDER
        if re.search(rf"\b{day}\b", days_phrase, flags=re.IGNORECASE)
    )
    if not allowed_days:
        allowed_days = ("Monday",)

    return participants, duration_minutes, work_start, work_end, allowed_days


def _extract_schedule_section(task_text: str) -> str:
    marker_match = re.search(
        r"Here are the existing schedules for everyone during the day(?:s)?:",
        task_text,
        flags=re.IGNORECASE,
    )
    if not marker_match:
        return ""
    tail = task_text[marker_match.end() :]
    stop = re.search(
        r"Find a time that works for everyone's schedule and constraints\.",
        tail,
        flags=re.IGNORECASE,
    )
    if stop:
        return tail[: stop.start()].strip()
    return tail.strip()


def _extract_busy_statement(statement: str) -> tuple[str, str] | None:
    cleaned = statement.strip()
    if not cleaned:
        return None
    busy_match = re.match(
        rf"^([A-Z][a-zA-Z']+)\s+(has meetings|is busy|has blocked their calendar)\s+on\s+(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if busy_match:
        person = busy_match.group(1).replace("'s", "")
        return person, busy_match.group(3).strip()
    open_match = re.match(
        rf"^([A-Z][a-zA-Z']+)(?:'s calendar is wide open the entire day| is free the entire day| has no meetings the whole day)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if open_match:
        person = open_match.group(1).replace("'s", "")
        return person, ""
    return None


def _parse_day_segments(schedule_tail: str) -> Dict[str, List[Tuple[int, int]]]:
    if not schedule_tail:
        return {}
    day_marker = re.compile(rf"{DAY_RE}\s+during\s+", flags=re.IGNORECASE)
    matches = list(day_marker.finditer(schedule_tail))
    if not matches:
        return {}

    by_day: Dict[str, List[Tuple[int, int]]] = {}
    for idx, match in enumerate(matches):
        day = match.group(1).capitalize()
        segment_start = match.end()
        segment_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(schedule_tail)
        segment = schedule_tail[segment_start:segment_end]
        intervals: List[Tuple[int, int]] = []
        for start, end in re.findall(r"([0-9]{1,2}:[0-9]{2})\s+to\s+([0-9]{1,2}:[0-9]{2})", segment):
            start_min = _time_to_minutes(start)
            end_min = _time_to_minutes(end)
            if end_min > start_min:
                intervals.append((start_min, end_min))
        by_day.setdefault(day, []).extend(intervals)
    return by_day


def _parse_busy_schedules(
    schedule_section: str,
    participants: Iterable[str],
) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    by_person_day: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        person: {day: [] for day in DAY_ORDER}
        for person in participants
    }
    statements = re.split(r"[;\n]+", schedule_section)
    for statement in statements:
        parsed = _extract_busy_statement(statement)
        if parsed is None:
            continue
        person, schedule_tail = parsed
        if person not in by_person_day:
            continue
        day_segments = _parse_day_segments(schedule_tail)
        for day, intervals in day_segments.items():
            if day not in by_person_day[person]:
                by_person_day[person][day] = []
            by_person_day[person][day].extend(intervals)
    return by_person_day


def _parse_days_from_text(text: str) -> Tuple[str, ...]:
    seen = {match.capitalize() for match in re.findall(DAY_RE, text, flags=re.IGNORECASE)}
    return tuple(day for day in DAY_ORDER if day in seen)


def _parse_preferences(task_text: str, participants: Iterable[str]) -> tuple[Tuple[Preference, ...], Tuple[Preference, ...], bool]:
    participant_set = set(participants)
    text = _normalize_text(task_text)
    earliest_requested = bool(
        re.search(
            r"(The group|You)\s+would\s+like\s+to\s+(?:schedule\s+the\s+meeting\s+at\s+their|meet\s+at\s+their)\s+earliest\s+availability",
            text,
            flags=re.IGNORECASE,
        )
    )

    clause_re = re.compile(
        rf"(?P<subject>[A-Z][a-zA-Z']+)\s+"
        r"(?P<kind>would like to avoid more meetings|would rather not meet|do not want to meet|can not meet)\s+"
        r"on\s+(?P<rest>.*?)"
        rf"(?=(?:\b[A-Z][a-zA-Z']+\s+(?:would like to avoid more meetings|would rather not meet|do not want to meet|can not meet)\s+on\b)|"
        r"(?:\b(?:The group|You)\s+would\s+like\s+to\b)|"
        r"(?:\bFind a time that works for everyone's schedule and constraints\b)|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    soft: List[Preference] = []
    hard: List[Preference] = []
    for match in clause_re.finditer(text):
        subject = match.group("subject").replace("'s", "")
        if subject not in participant_set:
            continue
        kind = match.group("kind").lower().strip()
        rest = match.group("rest")
        days = _parse_days_from_text(rest)
        if not days:
            continue
        bound_match = re.search(r"\b(before|after)\s+([0-9]{1,2}:[0-9]{2})", rest, flags=re.IGNORECASE)
        bound = None
        bound_minutes = None
        if bound_match:
            bound = bound_match.group(1).lower()
            bound_minutes = _time_to_minutes(bound_match.group(2))
        pref = Preference(
            subject=subject,
            kind=kind,
            days=days,
            bound=bound,
            bound_minutes=bound_minutes,
            hard=(kind == "can not meet"),
        )
        if pref.hard:
            hard.append(pref)
        else:
            soft.append(pref)
    return tuple(soft), tuple(hard), earliest_requested


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
    return f"{day}, {_minutes_to_time(start)} - {_minutes_to_time(end)}"


def _dedupe_and_sort_slots(slots: Iterable[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    unique = {(day, start, end) for day, start, end in slots}
    return sorted(
        unique,
        key=lambda slot: (DAY_TO_INDEX.get(slot[0], 999), slot[1], slot[2]),
    )


class DeterministicTaskQualityScorer:
    """Scores calendar predictions without golden labels or LLM calls."""

    def __init__(self, task_text: str) -> None:
        normalized = _normalize_text(task_text)
        (
            participants,
            duration_minutes,
            work_start_minutes,
            work_end_minutes,
            allowed_days,
        ) = _extract_task_header(normalized)
        busy = _parse_busy_schedules(
            _extract_schedule_section(normalized),
            participants,
        )
        soft_prefs, hard_prefs, earliest_requested = _parse_preferences(
            normalized,
            participants,
        )
        self.task = ParsedTask(
            participants=participants,
            duration_minutes=duration_minutes,
            work_start_minutes=work_start_minutes,
            work_end_minutes=work_end_minutes,
            allowed_days=allowed_days,
            busy_by_participant_by_day=busy,
            soft_preferences=soft_prefs,
            hard_preferences=hard_prefs,
            earliest_requested=earliest_requested,
        )
        self.valid_slots = self._enumerate_hard_valid_slots()

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

    def _enumerate_hard_valid_slots(self) -> List[Tuple[str, int, int]]:
        slots: List[Tuple[str, int, int]] = []
        for day in self.task.allowed_days:
            start = self.task.work_start_minutes
            last_start = self.task.work_end_minutes - self.task.duration_minutes
            while start <= last_start:
                end = start + self.task.duration_minutes
                if self._passes_busy(day, start, end) and self._passes_hard_preferences(day, start, end):
                    slots.append((day, start, end))
                start += SLOT_STEP_MINUTES
        return _dedupe_and_sort_slots(slots)

    def score_prediction(self, prediction: str) -> DeterministicQualityResult:
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
                earliest_slot=_format_slot(*self.valid_slots[0]) if self.valid_slots else None,
                candidate_slot=None,
            )

        day, start_text, end_text = parsed
        day = day.capitalize()
        start = _time_to_minutes(start_text)
        end = _time_to_minutes(end_text)
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

        earliest_slot = _format_slot(*self.valid_slots[0]) if self.valid_slots else None
        slot_rank = None
        for idx, slot in enumerate(self.valid_slots):
            if slot == (day, start, end):
                slot_rank = idx
                break

        soft_satisfied = 0
        soft_total = 0
        for preference in self.task.soft_preferences:
            soft_total += 1
            if not _violates_preference(preference, day, start, end):
                soft_satisfied += 1

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
            earliest_slot=earliest_slot,
            candidate_slot=candidate_slot,
        )


def build_task_quality_scorer(task_text: str) -> DeterministicTaskQualityScorer:
    return DeterministicTaskQualityScorer(task_text)
