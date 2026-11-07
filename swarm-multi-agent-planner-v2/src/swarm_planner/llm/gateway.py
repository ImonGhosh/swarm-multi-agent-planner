"""The single entry point for every LLM call: cache, provider call, usage accounting.

``LLMGateway.chat`` returns the answer together with its request hash, token counts, cost,
provider and latency, and emits one ``llm_call`` event per call. The provider is called with
the raw OpenAI SDK because LangChain drops the provider name and the cost.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional

from .cache import (
    CachedResponse,
    LLMCache,
    request_key,
    request_payload,
    short_hash,
)
from .client import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    create_client,
    raise_model_unavailable,
    uses_openrouter,
)
from .resilience import call_with_backoff, provider_preferences_from_env


class CostCapExceeded(RuntimeError):
    """The run has paid for as much as ``max_cost_usd`` allows; no further provider call is made."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    request_hash: str  # short hash, as logged
    cache_hit: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    provider: str
    latency_s: float
    finish_reason: str


@dataclass
class CallStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Cost of the calls that were actually paid for (cache misses).
    cost_usd_paid: float = 0.0
    # Cost the same calls would have had without the cache (misses + hits).
    cost_usd_equivalent: float = 0.0
    latency_s_paid: float = 0.0

    def add(self, result: LLMResult) -> None:
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.cost_usd_equivalent += result.cost_usd
        if result.cache_hit:
            self.cache_hits += 1
        else:
            self.cost_usd_paid += result.cost_usd
            self.latency_s_paid += result.latency_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd_paid": round(self.cost_usd_paid, 8),
            "cost_usd_uncached_equivalent": round(self.cost_usd_equivalent, 8),
            "latency_s_paid": round(self.latency_s_paid, 3),
        }


def _field(obj: Any, name: str) -> Any:
    """Read a field that the SDK may expose as an attribute or as an extra field."""
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return value


class LLMGateway:
    def __init__(
        self,
        *,
        model: str,
        cache: Optional[LLMCache] = None,
        client: Any = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        top_p: float = LLM_TOP_P,
        provider_preferences: Optional[Dict[str, Any]] = None,
        max_concurrent_requests: int = 8,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_cost_usd: Optional[float] = None,
    ) -> None:
        self.model = model
        # Stop making provider calls once this much has been paid in this run (cache hits are
        # free). The calls already in flight finish, so the total can overshoot by a few cents.
        self.max_cost_usd = max_cost_usd
        self.cache = cache
        self._client = client
        self.params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        self._provider_preferences = provider_preferences
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent_requests)))
        self.event_sink = event_sink
        self.stats = CallStats()
        self._task_stats: Dict[str, CallStats] = {}
        self._inflight: Dict[str, asyncio.Future] = {}

    # -- configuration -----------------------------------------------------------------

    def provider_preferences(self) -> Dict[str, Any]:
        if self._provider_preferences is None:
            self._provider_preferences = provider_preferences_from_env() if uses_openrouter() else {}
        return self._provider_preferences

    def stats_for(self, task_id: str) -> CallStats:
        return self._task_stats.setdefault(task_id, CallStats())

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_client()
        return self._client

    # -- the call ------------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        nonce: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        """Answer one chat request from the cache, or from the provider on a miss.

        ``nonce`` must differ between samples when temperature > 0. ``meta`` (task id,
        iteration, agent, step) is copied into the llm_call event.
        """
        payload = request_payload(
            model=self.model, params=self.params, messages=messages, nonce=nonce
        )
        key = request_key(payload)
        cached = self.cache.get(key) if self.cache is not None else None
        if cached is not None:
            result = self._result(key, cached, cache_hit=True)
        elif key in self._inflight:
            # Another agent is already asking the provider the identical question.
            shared = await self._inflight[key]
            result = replace(shared, cache_hit=True, latency_s=0.0)
        else:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future
            try:
                result = await self._call_provider(key, payload, messages)
                future.set_result(result)
            except BaseException as err:
                future.set_exception(err)
                future.exception()  # mark as retrieved when nobody else is waiting
                raise
            finally:
                self._inflight.pop(key, None)
        self._record(result, meta or {})
        return result

    async def _call_provider(
        self, key: str, payload: Dict[str, Any], messages: List[Dict[str, str]]
    ) -> LLMResult:
        if self.max_cost_usd is not None and self.stats.cost_usd_paid >= self.max_cost_usd:
            raise CostCapExceeded(
                f"cost cap reached: paid ${self.stats.cost_usd_paid:.4f} >= "
                f"max_cost_usd ${self.max_cost_usd:.4f}. Everything finished so far is in the "
                "cache, so re-running the same command continues without paying again."
            )
        client = self._get_client()
        extra_body: Dict[str, Any] = {}
        provider = self.provider_preferences()
        if provider:
            extra_body["provider"] = provider

        async def _create() -> Any:
            return await client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body=extra_body or None,
                **self.params,
            )

        async with self._semaphore:
            started = time.perf_counter()
            response = await call_with_backoff(
                _create, model_label=self.model, on_not_found=raise_model_unavailable
            )
            latency = time.perf_counter() - started

        choice = response.choices[0]
        usage = _field(response, "usage")
        cost = _field(usage, "cost")
        stored = CachedResponse(
            text=(choice.message.content or ""),
            prompt_tokens=int(_field(usage, "prompt_tokens") or 0),
            completion_tokens=int(_field(usage, "completion_tokens") or 0),
            cost_usd=float(cost) if cost is not None else 0.0,
            provider=str(_field(response, "provider") or ""),
            latency_s=latency,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
        )
        if self.cache is not None:
            self.cache.put(key, payload, stored)
            # Return what is stored, so that a concurrent writer's answer wins consistently.
            winner = self.cache.get(key)
            if winner is not None:
                stored = winner
        return self._result(key, stored, cache_hit=False)

    @staticmethod
    def _result(key: str, stored: CachedResponse, *, cache_hit: bool) -> LLMResult:
        return LLMResult(
            text=stored.text,
            request_hash=short_hash(key),
            cache_hit=cache_hit,
            prompt_tokens=stored.prompt_tokens,
            completion_tokens=stored.completion_tokens,
            cost_usd=stored.cost_usd,
            provider=stored.provider,
            latency_s=0.0 if cache_hit else stored.latency_s,
            finish_reason=stored.finish_reason,
        )

    def _record(self, result: LLMResult, meta: Dict[str, Any]) -> None:
        self.stats.add(result)
        task_id = meta.get("task_id")
        if task_id is not None:
            self.stats_for(str(task_id)).add(result)
        if self.event_sink is not None:
            self.event_sink(
                {
                    "event": "llm_call",
                    **meta,
                    "req": result.request_hash,
                    "hit": result.cache_hit,
                    "in": result.prompt_tokens,
                    "out": result.completion_tokens,
                    "lat": round(result.latency_s, 3),
                    "cost": round(result.cost_usd, 8),
                    "prov": result.provider,
                    "fin": result.finish_reason,
                }
            )
