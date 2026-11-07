"""Compressed JSONL event log for solve runs.

One file per run, ``events.jsonl.gz``, one JSON object per line with an ``event`` field:

``run_start``   resolved config, git commit, installed packages, branch and edge order
``iteration``   per task and iteration: pheromone before/after, every trace with its branch
                probabilities, request hashes, answer and verifier result
``llm_call``    one per LLM call: request hash, tokens, cost, latency, provider, cache hit
``task_result`` per task: prediction, per-iteration summary, call/cost totals
``task_error``  a task that raised (the run stops after logging it)
``run_end``     totals for the whole run

Prompts and responses are not written here. Events carry a 16-character request hash, and
the full request and answer are stored once in the LLM cache (``LLMCache.find_by_prefix``).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENTS_FILENAME = "events.jsonl.gz"
# Events between forced flushes. A flush makes everything written so far readable even if the
# process is killed, at a small cost in compression ratio.
FLUSH_EVERY = 500


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class EventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._gz = gzip.GzipFile(self.path, "wb", compresslevel=6)
        self._since_flush = 0
        self.events_written = 0

    def log(self, event: str, payload: dict[str, Any] | None = None, *, flush: bool = False) -> None:
        row = {"event": event, **_to_jsonable(payload or {})}
        if event not in ("llm_call",):
            row["ts"] = utc_now_iso()
        self._write(row, flush=flush)

    def log_row(self, row: dict[str, Any]) -> None:
        """Write an already-built event row (used for llm_call events)."""
        self._write(_to_jsonable(row), flush=False)

    def _write(self, row: dict[str, Any], *, flush: bool) -> None:
        self._gz.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        self.events_written += 1
        self._since_flush += 1
        if flush or self._since_flush >= FLUSH_EVERY:
            self._gz.flush()
            self._since_flush = 0

    def close(self) -> None:
        if not self._gz.closed:
            self._gz.close()


def read_events(path: str | Path):
    """Yield the events of a (possibly truncated) events.jsonl.gz file."""
    with gzip.open(path, "rb") as handle:
        try:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        except (EOFError, gzip.BadGzipFile):
            return  # a run that was killed mid-write: keep what was flushed
