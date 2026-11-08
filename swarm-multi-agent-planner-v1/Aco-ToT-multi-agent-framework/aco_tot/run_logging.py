"""Structured JSONL logging for train/infer runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_log_path(
    *,
    mode: str,
    base_dir: str | Path,
    log_out_path: str | None = None,
) -> Path:
    if log_out_path:
        return Path(log_out_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(base_dir) / "logs" / f"{mode}_run_{ts}.jsonl"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class JsonlRunLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp_utc": utc_now_iso(),
            "event": event,
            **_to_jsonable(payload),
        }
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
