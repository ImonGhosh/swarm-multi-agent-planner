from __future__ import annotations

import asyncio

import pytest

from fakes import FakeClient
from swarm_planner.llm.cache import CachedResponse, LLMCache, request_key, request_payload
from swarm_planner.llm.gateway import LLMGateway
from swarm_planner.llm.resilience import provider_preferences_from_env

MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hello"},
]
PARAMS = {"temperature": 0.0, "max_tokens": 400, "top_p": 1.0}


def _key(**overrides):
    args = {"model": "m", "params": PARAMS, "messages": MESSAGES, "nonce": None}
    args.update(overrides)
    return request_key(request_payload(**args))


def _stored(text="answer"):
    return CachedResponse(text, 100, 10, 0.0002, "FakeProvider", 0.5, "stop")


# --- cache --------------------------------------------------------------------------


def test_request_key_depends_on_everything_that_changes_the_answer():
    base = _key()
    assert _key() == base
    assert len(base) == 64
    assert _key(model="other") != base
    assert _key(params={**PARAMS, "temperature": 0.7}) != base
    assert _key(messages=MESSAGES[:1]) != base
    assert _key(nonce="sample-1") != base
    assert _key(nonce="sample-1") != _key(nonce="sample-2")


def test_cache_roundtrip_persists_and_first_answer_wins(tmp_path):
    path = tmp_path / "c.sqlite"
    key = _key()
    payload = request_payload(model="m", params=PARAMS, messages=MESSAGES)
    cache = LLMCache(path)
    assert cache.get(key) is None
    cache.put(key, payload, _stored("first"))
    cache.put(key, payload, _stored("second"))
    assert cache.get(key).text == "first"
    assert len(cache) == 1
    cache.close()

    reopened = LLMCache(path)
    assert reopened.get(key).text == "first"
    found = reopened.find_by_prefix(key[:16])
    assert found["text"] == "first"
    assert found["request"]["messages"] == MESSAGES
    assert reopened.find_by_prefix("0000000000000000") is None
    reopened.close()


def test_cache_migrates_a_plain_text_cache_and_stays_readable(tmp_path):
    import json
    import sqlite3

    path = tmp_path / "old.sqlite"
    key = _key()
    payload = request_payload(model="m", params=PARAMS, messages=MESSAGES)
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE responses (key TEXT PRIMARY KEY, model TEXT NOT NULL, request_json TEXT NOT NULL,"
        " text TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,"
        " cost_usd REAL NOT NULL, provider TEXT NOT NULL, latency_s REAL NOT NULL,"
        " finish_reason TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    db.execute(
        "INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (key, "m", json.dumps(payload), "old answer", 100, 10, 0.0002, "P", 0.5, "stop", 1.0),
    )
    db.commit()
    db.close()

    cache = LLMCache(path)
    assert cache.get(key).text == "old answer"
    assert cache.find_by_prefix(key[:16])["request"]["messages"] == MESSAGES
    cache.close()
    columns = [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(responses)")]
    assert "request_z" in columns and "request_json" not in columns


def test_stored_requests_are_compressed(tmp_path):
    import sqlite3

    cache = LLMCache(tmp_path / "c.sqlite")
    long_messages = [{"role": "user", "content": "five shot example " * 400}]
    payload = request_payload(model="m", params=PARAMS, messages=long_messages)
    cache.put(request_key(payload), payload, _stored())
    cache.close()
    size = sqlite3.connect(tmp_path / "c.sqlite").execute(
        "SELECT length(request_z) FROM responses"
    ).fetchone()[0]
    assert size < len(long_messages[0]["content"]) / 10


# --- gateway ------------------------------------------------------------------------


def _gateway(tmp_path, client, **kwargs):
    return LLMGateway(model="m", cache=LLMCache(tmp_path / "c.sqlite"), client=client, **kwargs)


def test_second_identical_call_is_served_from_the_cache(tmp_path):
    client = FakeClient(cost=0.001)
    gateway = _gateway(tmp_path, client)

    async def go():
        a = await gateway.chat(MESSAGES, meta={"task_id": "t1"})
        b = await gateway.chat(MESSAGES, meta={"task_id": "t1"})
        return a, b

    first, second = asyncio.run(go())
    assert len(client.calls) == 1
    assert first.cache_hit is False and second.cache_hit is True
    assert first.text == second.text and first.request_hash == second.request_hash
    assert second.latency_s == 0.0
    stats = gateway.stats
    assert (stats.calls, stats.cache_hits) == (2, 1)
    assert stats.cost_usd_paid == pytest.approx(0.001)
    assert stats.cost_usd_equivalent == pytest.approx(0.002)
    assert gateway.stats_for("t1").calls == 2


def test_nonce_keeps_stochastic_samples_apart(tmp_path):
    counter = iter(range(100))
    client = FakeClient(lambda messages: f"sample-{next(counter)}")
    gateway = _gateway(tmp_path, client, temperature=0.7)

    async def go():
        a = await gateway.chat(MESSAGES, nonce="0")
        b = await gateway.chat(MESSAGES, nonce="1")
        a_again = await gateway.chat(MESSAGES, nonce="0")
        return a, b, a_again

    a, b, a_again = asyncio.run(go())
    assert a.text != b.text
    assert a_again.text == a.text and a_again.cache_hit
    assert len(client.calls) == 2


def test_identical_concurrent_requests_make_one_provider_call(tmp_path):
    client = FakeClient(delay=0.05)
    gateway = _gateway(tmp_path, client)

    async def go():
        return await asyncio.gather(*[gateway.chat(MESSAGES) for _ in range(6)])

    results = asyncio.run(go())
    assert len(client.calls) == 1
    assert len({r.text for r in results}) == 1
    assert sum(1 for r in results if not r.cache_hit) == 1
    assert gateway.stats.cost_usd_paid == pytest.approx(0.0001)


def test_llm_call_event_carries_usage_provider_and_meta(tmp_path):
    events = []
    gateway = _gateway(tmp_path, FakeClient(provider="DeepInfra", cost=0.0003), event_sink=events.append)
    meta = {"task_id": "t1", "it": 2, "agent": 5, "step": "L3"}
    asyncio.run(gateway.chat(MESSAGES, meta=meta))
    asyncio.run(gateway.chat(MESSAGES, meta=meta))
    assert [e["hit"] for e in events] == [False, True]
    first = events[0]
    assert first["event"] == "llm_call"
    assert {k: first[k] for k in ("task_id", "it", "agent", "step")} == meta
    assert (first["in"], first["out"], first["prov"], first["fin"]) == (100, 10, "DeepInfra", "stop")
    assert first["cost"] == pytest.approx(0.0003) and len(first["req"]) == 16


def test_failed_call_does_not_poison_the_next_attempt(tmp_path):
    state = {"fail": True}

    def responder(messages):
        if state["fail"]:
            raise ValueError("boom")
        return "ok"

    client = FakeClient(responder)
    gateway = _gateway(tmp_path, client)
    with pytest.raises(ValueError):
        asyncio.run(gateway.chat(MESSAGES))
    state["fail"] = False
    assert asyncio.run(gateway.chat(MESSAGES)).text == "ok"
    assert len(gateway.cache) == 1  # the failure was not cached


def test_without_a_cache_every_call_reaches_the_provider():
    client = FakeClient()
    gateway = LLMGateway(model="m", cache=None, client=client)

    async def go():
        await gateway.chat(MESSAGES)
        await gateway.chat(MESSAGES)

    asyncio.run(go())
    assert len(client.calls) == 2


# --- provider pinning ---------------------------------------------------------------


def test_provider_is_pinned_without_fallback_by_default(monkeypatch):
    for name in (
        "OPENROUTER_PROVIDER_ONLY",
        "OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_PROVIDER_SORT",
        "LLM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    prefs = provider_preferences_from_env()
    assert prefs == {"allow_fallbacks": False, "only": ["DeepInfra"]}

    client = FakeClient()
    asyncio.run(LLMGateway(model="m", client=client).chat(MESSAGES))
    assert client.calls[0]["extra_body"]["provider"] == prefs


def test_provider_pin_can_be_changed_or_removed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "Parasail")
    assert provider_preferences_from_env()["only"] == ["Parasail"]
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "any")
    monkeypatch.setenv("OPENROUTER_PROVIDER_SORT", "price")
    assert provider_preferences_from_env() == {"allow_fallbacks": True, "sort": "price"}
