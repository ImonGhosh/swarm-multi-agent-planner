"""SQLite response cache for LLM calls.

Every call is keyed by a hash of (model, generation parameters, messages, nonce). A cached
response is returned instead of calling the provider. This has three effects:

* Runs are repeatable. Temperature 0 on OpenRouter is not bit-deterministic (measured in
  Phase 2: 3 distinct answers in 15 identical calls), so the cache fixes one realisation.
* ToT-only and ToT+ACO answer identical requests from the same cache, so they are compared on
  the same LLM outputs.
* The shipped cache file lets anyone replay a run without API calls, and the cache doubles as
  the prompt/response store that the run logs refer to by hash.

Calls with temperature > 0 must pass a ``nonce`` (for example the sample index) so that
repeated samples of the same prompt do not collapse into one cached answer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SHORT_HASH_LENGTH = 16


@dataclass(frozen=True)
class CachedResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    provider: str
    latency_s: float
    finish_reason: str


def request_payload(
    *,
    model: str,
    params: Dict[str, Any],
    messages: List[Dict[str, str]],
    nonce: Optional[str] = None,
) -> Dict[str, Any]:
    return {"model": model, "params": params, "messages": messages, "nonce": nonce}


def request_key(payload: Dict[str, Any]) -> str:
    """Stable sha256 of a request payload (full 64 hex characters)."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compress(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8"), 6)


def short_hash(key: str) -> str:
    """The prefix logged in run events; look it up in the cache with LLMCache.find_by_prefix."""
    return key[:SHORT_HASH_LENGTH]


# The request (which contains the 5-shot prompt) is stored zlib-compressed: about 3-4x smaller
# per row, which matters once the cache holds 100k+ entries.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key               TEXT PRIMARY KEY,
    model             TEXT NOT NULL,
    request_z         BLOB NOT NULL,
    text              TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    provider          TEXT NOT NULL,
    latency_s         REAL NOT NULL,
    finish_reason     TEXT NOT NULL,
    created_at        REAL NOT NULL
)
"""


class LLMCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # timeout: several runs may share one cache file.
        self._db = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._migrate_uncompressed_requests()
        self._db.execute(_SCHEMA)
        self._db.commit()

    def _migrate_uncompressed_requests(self) -> None:
        """Convert a cache written with a plain-text request_json column."""
        columns = [row[1] for row in self._db.execute("PRAGMA table_info(responses)")]
        if "request_json" not in columns:
            return
        self._db.execute("ALTER TABLE responses RENAME TO responses_old")
        self._db.execute(_SCHEMA)
        rows = self._db.execute(
            "SELECT key, model, request_json, text, prompt_tokens, completion_tokens, cost_usd, "
            "provider, latency_s, finish_reason, created_at FROM responses_old"
        ).fetchall()
        self._db.executemany(
            "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(r[0], r[1], _compress(r[2]), *r[3:]) for r in rows],
        )
        self._db.execute("DROP TABLE responses_old")
        self._db.commit()
        self._db.execute("VACUUM")

    def get(self, key: str) -> Optional[CachedResponse]:
        row = self._db.execute(
            "SELECT text, prompt_tokens, completion_tokens, cost_usd, provider, latency_s, "
            "finish_reason FROM responses WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return CachedResponse(*row)

    def put(self, key: str, payload: Dict[str, Any], response: CachedResponse) -> None:
        """Store a response. The first answer for a key wins (INSERT OR IGNORE)."""
        self._db.execute(
            "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                payload["model"],
                _compress(json.dumps(payload, ensure_ascii=False)),
                response.text,
                response.prompt_tokens,
                response.completion_tokens,
                response.cost_usd,
                response.provider,
                response.latency_s,
                response.finish_reason,
                time.time(),
            ),
        )
        self._db.commit()

    def find_by_prefix(self, prefix: str) -> Optional[Dict[str, Any]]:
        """Return the stored request and response for a logged short hash."""
        row = self._db.execute(
            "SELECT key, request_z, text FROM responses WHERE key LIKE ? LIMIT 2",
            (prefix + "%",),
        ).fetchall()
        if len(row) != 1:
            return None
        key, request_z, text = row[0]
        return {"key": key, "request": json.loads(zlib.decompress(request_z)), "text": text}

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        self._db.close()
