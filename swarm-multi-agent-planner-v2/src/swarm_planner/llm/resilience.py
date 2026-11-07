"""Helpers for resilient OpenRouter invocation."""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, NotFoundError, RateLimitError


_T = TypeVar("_T")
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The provider every experiment is pinned to. Chosen in Phase 2 from the OpenRouter endpoint
# list for google/gemma-3-4b-it (DeepInfra: 100% uptime, bf16; Parasail was degraded).
DEFAULT_PINNED_PROVIDER = "DeepInfra"


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def provider_preferences_from_env() -> Dict[str, Any]:
    """
    Build OpenRouter provider routing preferences from env vars.

    By default the run is pinned to one provider with no fallback, so a provider outage
    stops the run instead of silently switching to another backend.

    Supported env vars:
      - OPENROUTER_PROVIDER_ONLY: comma-separated provider names (default: DeepInfra).
        Set to ``any`` to remove the pin.
      - OPENROUTER_ALLOW_FALLBACKS: true|false (default: false when pinned, true otherwise)
      - OPENROUTER_PROVIDER_SORT: price|throughput|latency (only used when not pinned)
      - OPENROUTER_PROVIDER_IGNORE: comma-separated provider names
      - OPENROUTER_PROVIDER_DATA_COLLECTION: allow|deny
    """
    raw_only = os.getenv("OPENROUTER_PROVIDER_ONLY")
    if raw_only is None or not raw_only.strip():
        only = [DEFAULT_PINNED_PROVIDER]
    elif raw_only.strip().lower() == "any":
        only = []
    else:
        only = _csv_env("OPENROUTER_PROVIDER_ONLY")

    raw_fallbacks = os.getenv("OPENROUTER_ALLOW_FALLBACKS", "").strip().lower()
    if raw_fallbacks in {"true", "1", "yes"}:
        allow_fallbacks = True
    elif raw_fallbacks in {"false", "0", "no"}:
        allow_fallbacks = False
    else:
        allow_fallbacks = not only
    provider: Dict[str, Any] = {"allow_fallbacks": allow_fallbacks}
    if only:
        provider["only"] = only

    sort = os.getenv("OPENROUTER_PROVIDER_SORT", "").strip().lower()
    if not only and sort in {"price", "throughput", "latency"}:
        provider["sort"] = sort

    ignore = _csv_env("OPENROUTER_PROVIDER_IGNORE")
    if ignore:
        provider["ignore"] = ignore

    data_collection = os.getenv("OPENROUTER_PROVIDER_DATA_COLLECTION", "").strip().lower()
    if data_collection in {"allow", "deny"}:
        provider["data_collection"] = data_collection

    return provider


def is_openrouter_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    return "openrouter.ai" in base_url.lower()


def is_retryable_openai_error(err: BaseException) -> bool:
    if isinstance(err, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(err, APIStatusError):
        status_code = getattr(err, "status_code", None)
        if status_code in _RETRYABLE_STATUS_CODES:
            return True

    message = str(err).lower()
    if "temporarily rate-limited" in message:
        return True
    if "provider returned error" in message and "429" in message:
        return True
    return False


def _extract_retry_after_seconds(err: BaseException) -> float | None:
    if not isinstance(err, APIStatusError):
        return None
    response = getattr(err, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Retry-After can be either delta-seconds or an HTTP date.
    try:
        seconds = float(text)
        if seconds >= 0:
            return seconds
    except ValueError:
        pass

    try:
        http_date = datetime.strptime(text, "%a, %d %b %Y %H:%M:%S %Z")
        if http_date.tzinfo is None:
            http_date = http_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (http_date - now).total_seconds())
    except ValueError:
        return None


async def call_with_backoff(
    call: Callable[[], Awaitable[_T]],
    *,
    model_label: str,
    on_not_found: Callable[[str, NotFoundError], Any] | None = None,
) -> _T:
    """
    Await ``call()`` with retry/backoff for transient provider failures.
    """
    max_retries = _env_int(
        "OPENROUTER_MAX_RETRIES",
        default=12,
        minimum=0,
        maximum=40,
    )
    base_delay = _env_float(
        "OPENROUTER_RETRY_BASE_SECONDS",
        default=2.0,
        minimum=0.1,
        maximum=60.0,
    )
    max_delay = _env_float(
        "OPENROUTER_RETRY_MAX_SECONDS",
        default=90.0,
        minimum=base_delay,
        maximum=300.0,
    )

    for attempt in range(max_retries + 1):
        try:
            return await call()
        except NotFoundError as err:
            if on_not_found is not None:
                on_not_found(model_label, err)
            raise
        except Exception as err:
            retryable = is_retryable_openai_error(err)
            if not retryable:
                raise
            if attempt >= max_retries:
                raise RuntimeError(
                    "OpenRouter request for model "
                    f"'{model_label}' failed after {max_retries + 1} attempts "
                    "due to repeated transient provider errors/rate limits. "
                    "Try BYOK, a different provider/model, or increase retry env vars."
                ) from err
            retry_after = _extract_retry_after_seconds(err)
            if retry_after is not None:
                delay = min(max_delay, max(base_delay, retry_after))
            else:
                delay = min(max_delay, base_delay * float(2**attempt))
            jitter = random.uniform(0.0, min(1.0, delay * 0.25))
            await asyncio.sleep(delay + jitter)
