"""Parse calendar-scheduling task text and candidate answers into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


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
    # Anything the parser could not explain (a dropped calendar, an unknown
    # preference subject, ...). Empty for every task in the Natural Plan data.
    warnings: Tuple[str, ...] = ()


# The dataset glues a name to the next word in some tasks ("Ethanhas meetings").
# The split is applied only where a schedule verb follows, so names that end in
# "is" or "has" (Doris, Alexis, Dennis) are never split.
_GLUED_HAS = re.compile(r"\b([A-Z][a-z]+)has\b(?=\s+(?:meetings|blocked|no meetings))")
_GLUED_IS = re.compile(r"\b([A-Z][a-z]+)is\b(?=\s+(?:busy|free))")

_PREF_KINDS = (
    "would like to avoid more meetings|would rather not meet|do not want to meet|can not meet"
)
_PREF_STATEMENT_RE = re.compile(
    rf"(?:{_PREF_KINDS})\s+on\b|\bwould\s+like\s+to\b", flags=re.IGNORECASE
)


def _normalize_text(task_text: str) -> str:
    text = task_text or ""
    text = re.sub(r"\bearlist\b", "earliest", text, flags=re.IGNORECASE)
    text = _GLUED_HAS.sub(r"\1 has", text)
    text = _GLUED_IS.sub(r"\1 is", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def time_to_minutes(time_text: str) -> int:
    hour_text, minute_text = time_text.split(":")
    return int(hour_text) * 60 + int(minute_text)


def minutes_to_time(minutes: int) -> str:
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
    work_start = time_to_minutes(match.group("work_start"))
    work_end = time_to_minutes(match.group("work_end"))

    days_phrase = match.group("days").strip()
    days_phrase = re.sub(r"^either\s+", "", days_phrase, flags=re.IGNORECASE)
    days_phrase = re.sub(r"\s+or\s+", ", ", days_phrase, flags=re.IGNORECASE)
    allowed_days = tuple(
        day
        for day in DAY_ORDER
        if re.search(rf"\b{day}\b", days_phrase, flags=re.IGNORECASE)
    )
    if not allowed_days:
        # v1 silently defaulted to Monday here, which would score against the wrong days.
        raise ValueError("Unable to parse the allowed days from the task header.")

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
    cleaned = statement.strip().rstrip(".").strip()
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
        r"^([A-Z][a-zA-Z']+)(?:'s calendar is wide open the entire (?:day|week)"
        r"| is free the entire (?:day|week)"
        r"| has no meetings the whole (?:day|week))$",
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
            start_min = time_to_minutes(start)
            end_min = time_to_minutes(end)
            if end_min > start_min:
                intervals.append((start_min, end_min))
        by_day.setdefault(day, []).extend(intervals)
    return by_day


def _parse_busy_schedules(
    schedule_section: str,
    participants: Iterable[str],
) -> tuple[Dict[str, Dict[str, List[Tuple[int, int]]]], List[str]]:
    """Return (busy intervals per participant per day, parser warnings)."""
    participants = tuple(participants)
    warnings: List[str] = []
    by_person_day: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        person: {day: [] for day in DAY_ORDER}
        for person in participants
    }
    seen: set[str] = set()
    statements = re.split(r"[;\n]+", schedule_section)
    for statement in statements:
        if not statement.strip():
            continue
        parsed = _extract_busy_statement(statement)
        if parsed is None:
            # Preference sentences share this section; everything else is unexpected.
            if not _PREF_STATEMENT_RE.search(statement):
                warnings.append(f"unparsed schedule statement: {statement.strip()[:80]!r}")
            continue
        person, schedule_tail = parsed
        if person not in by_person_day:
            warnings.append(f"schedule statement for unknown participant {person!r}")
            continue
        seen.add(person)
        day_segments = _parse_day_segments(schedule_tail)
        if schedule_tail and not any(day_segments.values()):
            warnings.append(f"busy statement for {person!r} has no parsable intervals")
        for day, intervals in day_segments.items():
            if day not in by_person_day[person]:
                by_person_day[person][day] = []
            by_person_day[person][day].extend(intervals)
    for person in participants:
        if person not in seen:
            warnings.append(f"no schedule statement found for participant {person!r}")
    # Every "H:MM to H:MM" interval in the text must end up in a parsed calendar.
    in_text = len(re.findall(r"[0-9]{1,2}:[0-9]{2}\s+to\s+[0-9]{1,2}:[0-9]{2}", schedule_section))
    parsed_count = sum(
        len(intervals) for days in by_person_day.values() for intervals in days.values()
    )
    if in_text != parsed_count:
        warnings.append(f"{in_text} busy intervals in the text but {parsed_count} parsed")
    return by_person_day, warnings


def _parse_days_from_text(text: str) -> Tuple[str, ...]:
    seen = {match.capitalize() for match in re.findall(DAY_RE, text, flags=re.IGNORECASE)}
    return tuple(day for day in DAY_ORDER if day in seen)


def _parse_preferences(
    task_text: str, participants: Iterable[str]
) -> tuple[Tuple[Preference, ...], Tuple[Preference, ...], bool, List[str]]:
    """Return (soft preferences, hard preferences, earliest_requested, parser warnings).

    The dataset sometimes splits one preference over several sentences, e.g.
    "X do not want to meet on Monday. Tuesday before 11:00." Each sentence is parsed on
    its own: a bare day means the whole day, and a time bound applies only to the days
    named in its own sentence. Gold answers confirm this reading on all 1000 tasks.
    """
    participant_set = set(participants)
    warnings: List[str] = []
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
    clauses = 0
    for match in clause_re.finditer(text):
        clauses += 1
        subject = match.group("subject").replace("'s", "")
        if subject not in participant_set:
            warnings.append(f"preference for unknown participant {subject!r}")
            continue
        kind = match.group("kind").lower().strip()
        parsed_any = False
        for sentence in re.split(r"\.\s*", match.group("rest")):
            days = _parse_days_from_text(sentence)
            if not days:
                continue
            parsed_any = True
            bound_match = re.search(
                r"\b(before|after)\s+([0-9]{1,2}:[0-9]{2})", sentence, flags=re.IGNORECASE
            )
            bound = None
            bound_minutes = None
            if bound_match:
                bound = bound_match.group(1).lower()
                bound_minutes = time_to_minutes(bound_match.group(2))
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
        if not parsed_any:
            warnings.append(f"preference clause for {subject!r} names no day")

    mentions = len(re.findall(rf"(?:{_PREF_KINDS})\s+on\b", text, flags=re.IGNORECASE))
    if mentions != clauses:
        warnings.append(f"{mentions} preference phrases but {clauses} clauses parsed")
    if re.search(r"\bearliest\b", text, flags=re.IGNORECASE) and not earliest_requested:
        warnings.append("'earliest' is mentioned but no earliest-availability request was parsed")
    return tuple(soft), tuple(hard), earliest_requested, warnings


def parse_task(task_text: str) -> ParsedTask:
    """Parse a task prompt into its constraints. Raises ValueError if unparseable."""
    normalized = _normalize_text(task_text)
    (
        participants,
        duration_minutes,
        work_start_minutes,
        work_end_minutes,
        allowed_days,
    ) = _extract_task_header(normalized)
    busy, busy_warnings = _parse_busy_schedules(
        _extract_schedule_section(normalized),
        participants,
    )
    soft_prefs, hard_prefs, earliest_requested, pref_warnings = _parse_preferences(
        normalized,
        participants,
    )
    return ParsedTask(
        participants=participants,
        duration_minutes=duration_minutes,
        work_start_minutes=work_start_minutes,
        work_end_minutes=work_end_minutes,
        allowed_days=allowed_days,
        busy_by_participant_by_day=busy,
        soft_preferences=soft_prefs,
        hard_preferences=hard_prefs,
        earliest_requested=earliest_requested,
        warnings=tuple(busy_warnings + pref_warnings),
    )


def parse_proposed_time(text: str) -> Tuple[str, str, str] | None:
    """Parse 'Day, HH:MM - HH:MM' from a free-form answer."""
    if not text:
        return None
    match = re.search(
        r"([A-Za-z]+)\s*,\s*([0-9]{1,2}:[0-9]{2})\s*-\s*([0-9]{1,2}:[0-9]{2})",
        text,
    )
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def answers_match(prediction: str, gold: str) -> bool:
    """Exact match of day, start and end, as the Natural Plan evaluator compares them."""
    pred = parse_proposed_time(prediction)
    ref = parse_proposed_time(gold)
    if pred is None or ref is None:
        return False
    try:
        return (
            pred[0].lower() == ref[0].lower()
            and time_to_minutes(pred[1]) == time_to_minutes(ref[1])
            and time_to_minutes(pred[2]) == time_to_minutes(ref[2])
        )
    except ValueError:
        return False


def canonicalize_final_answer(text: str) -> str:
    """Ensure final answer follows evaluator-friendly canonical format."""
    parsed = parse_proposed_time(text or "")
    if not parsed:
        fallback = (text or "").strip()
        if fallback:
            return fallback
        return "Here is the proposed time: Monday, 9:00 - 9:30"
    day, start, end = parsed
    return f"Here is the proposed time: {day}, {start} - {end}"
