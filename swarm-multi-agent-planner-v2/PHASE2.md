# Phase 2 — Logging, response cache and cost tracking

**Status:** done. About $0.11 of LLM calls were spent on live checks and a smoke test (6 dev tasks at the full 8 × 6 budget).

**Goal:** a run leaves everything needed to audit it (Req. 2), to show ACO's effect (Req. 4) and to analyse it later (Ours 3): per-iteration pheromone and probabilities, every trace with its verifier result, and per-call usage, cost and provider. Responses are cached so runs can be replayed without API calls.

---

## 1. Summary

| Task in PLAN.md | Result |
|---|---|
| One compressed event log per run | Done. `events.jsonl.gz` with `run_start`, `iteration`, `llm_call`, `task_result`, `task_error`, `run_end`. About 9 KB per task, so roughly 4 MB for 400 tasks. |
| Store each prompt once, refer by hash | Done. Events carry a 16-character request hash. The full request and answer are in the LLM cache. |
| LLM response cache | Done. SQLite, zlib-compressed requests, optional nonce for stochastic calls, concurrent identical requests share one provider call. |
| Token, cost, latency and provider per call | Done, through the raw OpenAI SDK. |
| Pin the provider | Done. DeepInfra, no fallback. |
| Run directory layout | Done. `predictions.json`, `events.jsonl.gz`, `summary.json`, `config.resolved.json`. |

Added beyond the plan: task-level concurrency and a global request cap (a sequential run was too slow), and a cache migration so the cache can shrink without losing entries.

**Main findings**
- **Temperature 0 is not deterministic** on the provider: 3 distinct answers in 15 identical calls. The plan's "caching does not change results" was wrong. The cache fixes one realisation and makes runs repeatable.
- **A fresh 400-task run costs about $5–6**, much less than expected, because agents that share a path prefix share calls (64% of calls in a single run).
- **F8 is visible in the new logs:** on dev task 40 the correct answer appears in iterations 1–4, but the final iteration had no valid answer, so the prediction was wrong.

74 tests pass (33 from Phase 0, 15 from Phase 1, 26 new).

---

## 2. What was done

### 2.1 The LLM gateway (`llm/gateway.py`)

Every LLM call now goes through `LLMGateway.chat(messages, nonce=None, meta=None)`:

1. Build a request hash from the model, generation parameters, messages and nonce.
2. Return the cached answer if there is one.
3. If an identical request is already in flight, wait for it (agents with the same path prefix ask the same question at the same moment).
4. Otherwise call the provider (with retry and backoff), store the answer, and return it.

Each call returns `LLMResult` (text, short request hash, cache hit, tokens, cost, provider, latency, finish reason), updates run and per-task counters, and emits an `llm_call` event. A global semaphore (`--max_concurrent_requests`) caps provider load.

LangChain's `ChatOpenAI` was replaced by the raw `AsyncOpenAI` client, because LangChain drops the provider name and the cost. LangGraph still runs the five levels and the finalize step. `langchain-openai` (and two transitive packages) left `pyproject.toml` and `requirements.lock`.

### 2.2 The cache (`llm/cache.py`)

- One SQLite file, WAL mode, `INSERT OR IGNORE` (the first answer to a request wins).
- The key is the sha256 of `{model, params, messages, nonce}`. The provider is not part of the key, but it is stored with the answer.
- **Nonce:** at temperature 0 the nonce is empty. A stochastic run (the Phase 4 sampling control at temperature 0.7) must pass a per-sample nonce, otherwise its 48 samples would collapse into one cached answer.
- The stored request is zlib-compressed, which makes entries about 3× smaller (6.8 KB to 2.1 KB measured). An older cache with plain-text requests is migrated automatically when opened.
- `LLMCache.find_by_prefix(hash)` returns the stored request and answer for a hash seen in the logs.

Why the cache matters:
- **Repeatability.** Re-running a run replays finished calls: 1728 of 1728 calls hit the cache, $0, and the predictions were identical.
- **Fair comparison.** ToT-only and ToT+ACO answer identical requests from the same cache, so they differ only in which paths they take.
- **Resume.** A crashed run is simply re-run and replays what it finished.
- **Reproduction.** Shipping the cache lets the professor replay every run without API calls.

### 2.3 The event log (`logging.py`)

`EventLog` writes one JSON object per line to `events.jsonl.gz`. The file is flushed every 500 events and at every task boundary, so a killed run leaves a readable file (`read_events` also tolerates a truncated file).

| Event | Fields |
|---|---|
| `run_start` | `config` (all resolved settings, dataset sha256, generation parameters, provider preferences), `branch_order` (branch ids per level), `edge_order` (the 15 edges), `environment` (git commit, dirty flag, Python, platform, installed packages), `ts` |
| `iteration` | `task_id`, `it` (0-based), `tau_before` and `tau_after` (15 floats in `edge_order`, or `null` for ToT-only), `best_agent`, and `traces`, one per agent: `agent`, `path`, `p` (5 × 3 branch probabilities in `branch_order`), `req` (the 6 request hashes), `ans`, `ps` (path score), `q` (verifier score), `v` (`valid`, `hard`, `soft`, `soft_total`, `rank`) |
| `llm_call` | `task_id`, `it`, `agent`, `step` (`L1`–`L5` or `final`), `req`, `hit`, `in`, `out`, `lat`, `cost`, `prov`, `fin` |
| `task_result` | `task_id`, `position`, `prediction`, `metadata` (including `per_iteration`, which is 1-based), `llm` totals, `wall_s` |
| `task_error` | `task_id`, `error`. The run stops after logging it. |
| `run_end` | `tasks_processed`, `final_mean_quality`, `pheromone_updates_applied`, `llm` totals, `cache_entries`, `wall_s` |

No prompt or response text is written to the log. Per task, a log of 8 traces × 6 iterations is about 9 KB compressed.

Cost fields: `llm` totals report both `cost_usd_paid` (calls that reached the provider) and `cost_usd_uncached_equivalent` (what all calls would have cost without the cache). The second one is what to compare between configurations, because a later run is free once the cache is warm.

### 2.4 Changes to the solver, graph and types

- `tot/graph.py`: the graph takes the gateway. Each level records the **full branch probability distribution** (uniform 1/3 for ToT-only) and the request hash, so ACO's influence can be shown from the log. Graph state and prompts are unchanged.
- `types.py`: `ThoughtResult` gained `probabilities` and `request_hash`. The large prompt fields (`accumulated_prompt`, `system_prompt`, `user_prompt`) were removed from traces. `AgentTrace` gained `final_request_hash`.
- `solver.py`:
  - `solve_task` takes a gateway and an optional event log and emits an `iteration` event per iteration.
  - `solve_dataset(data_path, run_dir, ...)` replaces `out_path` and `log_out_path`. It writes the four run-directory files, supports `limit`, and stops with a logged `task_error` if a task fails.
  - Tasks can run concurrently (`task_concurrency`). Tasks are independent and every random draw is derived from `(seed, task, iteration, agent, level)`, so concurrency does not change results. A test checks this.
  - All agents of a task now run in parallel by default (the cap moved to the gateway).
- `scripts/run_experiment.py`: new flags `--run_dir`, `--cache_path`, `--limit`, `--task_concurrency`, `--max_concurrent_requests`. `--out_path` and `--log_out_path` are gone.
- `runinfo.py`: git commit, installed packages (like `pip freeze`, without a subprocess), dataset hash.

### 2.5 Provider pinning (`llm/resilience.py`)

By default requests go to DeepInfra only, with `allow_fallbacks: false`, so a provider outage stops the run instead of silently switching backend. `OPENROUTER_PROVIDER_ONLY` changes the provider, or `any` removes the pin. The provider that answered is recorded with every call. `.env.example` documents the new defaults.

---

## 3. Live checks (2026-11-05)

**Providers for `google/gemma-3-4b-it`** (OpenRouter endpoint list):

| Provider | Quantization | Input / output per 1M tokens | Uptime (30 min) |
|---|---|---|---|
| DeepInfra | bf16 | $0.05 / $0.10 | 100% |
| Parasail | bf16 | $0.04 / $0.05 | 66.7%, status degraded |

DeepInfra was pinned: it is slightly dearer but stable, and cost is not the constraint.

**Determinism at temperature 0** (one real prompt of 4.4k characters, DeepInfra, 10 sequential and 5 concurrent identical calls): 3 distinct answers in 15 calls. Two calls differed from the other thirteen, including the first. The latency was 0.5 to 3.4 s, and one call cost about $0.0001.

**Available response fields:** `provider`, `usage.prompt_tokens`, `usage.completion_tokens` and `usage.cost` all come back through the raw SDK.

---

## 4. Smoke test

Dev tasks 40–45 (`data/dev_100.json`), the full 8 agents × 6 iterations, seed 0. Outputs are kept in `results/smoke_logging/`.

| Run | Tasks | LLM calls | Cache hits | Paid | Uncached equivalent | Wall time |
|---|---|---|---|---|---|---|
| ToT+ACO (`tot_aco`) | 2 | 576 | 366 (64%) | $0.0228 | $0.0618 | 96 s |
| Same again (`replay`) | 2 | 576 | 576 | $0 | $0.0618 | 1.4 s |
| ToT-only (`tot_only`) | 2 | 576 | 362 | $0.0230 | $0.0605 | not measured |
| ToT+ACO, 6 tasks (`scale`) | 6 | 1728 | 1206 | $0.0575 | $0.1851 | 141 s |

- The replay gave identical predictions, also after the cache migration.
- In the 6-task run, 522 calls were paid. The average paid call took 2.7 s and about 10 calls ran at once (`--task_concurrency 6 --max_concurrent_requests 32`). All 522 were answered by DeepInfra, and 8 of them (1.5%) stopped at `max_tokens=400`.
- Observed prompt sizes: up to 3.2k tokens, with an average answer of 48 tokens.
- The quality of these 6 tasks is not meaningful (mean 0.61), only the pipeline is.

**What the logs already show (F8).** On task 40, valid answers (including the gold 14:00 – 15:00) appear in iterations 1 to 4, but iteration 5 had none, so the final prediction scored 0.392 (invalid). The ToT-only run found it in its last iteration by chance. Which iteration happens to be last decides the answer, which is why Phase 3 compares best-of-last with best-of-all.

---

## 5. Cost and time projections

| Item | Estimate |
|---|---|
| Fresh 400-task run (ToT or ToT+ACO) | about 105 distinct provider calls per task, so $5–6 and roughly 3 hours at the measured parallelism |
| The same run with no caching at all | about $12 |
| Dev path table (100 tasks × up to 606 calls) | about $7, a few hours |
| Event log per 400-task run | about 4 MB |
| Cache size | about 2 KB per entry. 100–250k entries is 200–500 MB |

These are extrapolations from 6 tasks. Time depends on provider speed and on how far `--max_concurrent_requests` can be raised before rate limits appear, which was not stress-tested.

---

## 6. Decisions and differences from the plan

- **Logs hold hashes, the cache holds text.** This keeps the main log small, as you requested. The log is not standalone: analysis that needs intermediate answers joins to the cache by hash (`find_by_prefix`).
- **`config.resolved.json` instead of `.yaml`.** Avoids a new dependency. The same config is also in `run_start`. PLAN.md's layout was updated.
- **The cache fixes one realisation of each answer.** Seeds therefore vary only the branch choices, not the LLM's own noise. This is a limitation to state in the paper.
- **Concurrency defaults:** one task at a time, all agents of a task at once, at most 8 requests in flight. Raise `--task_concurrency` and `--max_concurrent_requests` for real runs.

---

## 7. Verification

| Check | Result |
|---|---|
| `pytest` | 74 passed |
| Cache: key sensitivity, round trip, first answer wins, compression, migration of an old cache | tested |
| Gateway: hit and miss, nonce, concurrent identical requests share one call, event content, errors don't poison the next call, no-cache mode, provider pin | tested |
| Event log: structure, pheromone and probabilities, hashes resolve in the cache, no prompt text in the log, replay with 0 provider calls, ToT-only shares the cache, task concurrency gives identical results, `--limit`, failed task, crash-safe and truncated files | tested |
| Live: 15-call determinism test, 6-task smoke test, replay, ToT-only, cache migration on real data | done, see Sections 3 and 4 |

---

## 8. Known issues carried to later phases

| Item | Phase |
|---|---|
| The baseline is not built yet. With the cache, a baseline repeated over 3 seeds would return identical answers (SD exactly 0). Decide there whether to pass the seed as the nonce so each seed is an independent provider call. | 4 |
| Several processes sharing one cache file rely on SQLite's WAL mode and a 60 s timeout. This was not tested with real parallel processes. | 4 |
| The solver still returns the best trace of the last iteration (F8). The log has everything needed to compute the other selection policies. | 3, 4 |
| The deposit mapping and the v2 update rule (F7) are still not wired in. | 3 |
| 1.5% of calls stop at `max_tokens=400`. They are cached as they are. Check in Phase 3 whether they hurt. | 3 |
| A run that fails writes no `predictions.json`. Re-run it: finished calls replay from the cache. | — |

---

## 9. How to use

From the v2 root (`.venv/bin/python` on Linux or macOS):

```bash
# a run: everything goes to --run_dir, responses are cached
.venv/Scripts/python.exe scripts/run_experiment.py --data_path data/dev_100.json --limit 2 \
    --run_dir results/runs/demo/seed_0 --seed 0 --aco true --task_concurrency 2

# repeat it: 0 API calls, identical predictions
.venv/Scripts/python.exe scripts/run_experiment.py --data_path data/dev_100.json --limit 2 \
    --run_dir results/runs/demo_replay/seed_0 --seed 0 --aco true

# look up a request hash from the log
.venv/Scripts/python.exe -c "from swarm_planner.llm.cache import LLMCache; \
print(LLMCache('results/cache/llm_cache.sqlite').find_by_prefix('<16-char hash>'))"

# read a log
.venv/Scripts/python.exe -c "from swarm_planner.logging import read_events; \
print(next(read_events('results/runs/demo/seed_0/events.jsonl.gz'))['event'])"
```

The API key goes into `.env` (see `.env.example`). `results/cache/` is git-ignored because it grows large. It is published as a release archive instead (Phase 6).
