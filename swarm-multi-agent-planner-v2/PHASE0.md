# Phase 0 — v2 repo setup and fixes

**Status:** done. No LLM calls were made in this phase.

**Goal:** a v2 repo that installs and runs solve mode on a clean checkout (fixes F1), has trustworthy datasets (the dataset part of F2), and removes the seed collision (F9). The verifier, logging and ACO rule are deliberately unchanged. They belong to Phases 1–3.

---

## 1. Summary

| Task in PLAN.md | Result |
|---|---|
| Port v1 code into the new layout | Done. Package `src/swarm_planner/` with `tot/`, `aco/`, `verifier/`, `llm/` and the top-level modules `solver.py`, `logging.py`, `seeding.py`, `types.py`. |
| Remove `train`/`infer`, the `quality` import, `store.py`, dead code | Done. The solve path no longer imports anything that is missing. |
| Make the existing tests pass | Done. Applicable v1 tests ported, new tests added: **33 pass**. |
| `scripts/build_datasets.py` | Done. Builds `curated_400.json` and `dev_100.json`. The curated set matches v1's exactly. |
| `seeding.py` (F9) | Done. |
| Pin dependencies | Done. `pyproject.toml` and `requirements.lock`. |

---

## 2. What was done

### 2.1 Package layout (ported from v1 `aco_tot/`)

| v2 module | Origin | Changes |
|---|---|---|
| `tot/levels.py` | `config.py` | Levels, branches, prompts and edge keys only. The ACO defaults moved to `aco/__init__.py`. |
| `tot/context.py` | `context_accumulator.py` + `prompt_io.py` | Prompt accumulation, 5-shot task extraction and task features, merged. |
| `tot/graph.py` | `llm_nodes.py` | The LangGraph builder. The client code moved to `llm/client.py`. Randomness now comes from `seeding.py`. |
| `aco/selection.py` | `aco.py` | Uses numpy generators instead of `random.Random`. |
| `aco/pheromone.py` | `pheromone.py` + part of `store.py` | The update rule is unchanged. `create_initial_pheromone_state` moved here. |
| `aco/heuristics.py` | `heuristics.py` | Unchanged apart from imports. |
| `verifier/parser.py` | `task_quality.py` + `prompt_io.py` | The parsing half, plus `parse_proposed_time` and `canonicalize_final_answer`. A new `parse_task()` wraps the parsing steps. |
| `verifier/scorer.py` | `task_quality.py` | The scoring half. The logic is unchanged. |
| `llm/client.py`, `llm/resilience.py` | `llm_nodes.py`, `openrouter_resilience.py` | Same behaviour. The resilience file is unchanged. |
| `solver.py` | `engine.py` | The solve path only. `process_task` became `solve_task`. |
| `logging.py`, `types.py` | `run_logging.py`, `types.py` | Same behaviour. |

Scripts:
- `scripts/run_experiment.py` replaces the v1 CLI `infer_aco_tot_calendar.py`. It has no `--mode` flag.
- `scripts/evaluate.py` is the official Natural Plan evaluator, copied verbatim with its Apache-2.0 header.

### 2.2 Dead code removed

- The `train` and `infer` modes and their metrics and pheromone-file handling.
- `store.py`, `invoke.py`, and the `quality` import that made solve mode fail on a clean checkout (F1). `quality.py` was not recreated.
- `invoke.py`'s single-task role is now covered by the async `solve_task()`, exported from the package.

### 2.3 Datasets (`scripts/build_datasets.py`)

- Source: `natural-plan/data/calendar_scheduling.json` (1000 tasks, sha256 recorded in `data/manifest.json`).
- Rule: group by (num_people, num_days), in source-file order. The first 40 per category are the **test set** (`curated_400.json`). The next 10 per category are the **dev set** (`dev_100.json`).
- All `pred_*` fields are dropped. The script fails if any unexpected field remains, so the leftover Qwen predictions cannot come back.
- **Check against v1:** the rebuilt test set has the same 400 task ids, in the same order, with the same content (ignoring `pred_*`). The script enforces this when the v1 file is present, and a test covers it too.
- 10 categories: 2 people × 1–5 days, plus 3–7 people × 1 day. Each has 40 test and 10 dev tasks. Test and dev do not overlap.

### 2.4 Seeding (F9) (`seeding.py`)

- v1: `seed + task_offset`, then `+ iteration * 10_000_000`, `+ agent_id * 100003`, `+ level * 7919`. Streams of different seeds collided.
- v2: `numpy.random.SeedSequence([seed, task_key(task_id), iteration, agent, level])`. `task_key` is a sha256-based integer, so it is stable across processes.
- `solver.py` now passes the run seed and the task id unchanged. It no longer adds the task position to the seed.
- Tests: reproducibility, every coordinate changes the stream, no collisions over the full grid (3 seeds × 3 tasks × 6 iterations × 8 agents × 5 levels), and seed 43 on task *i* ≠ seed 42 on task *i+1*.

### 2.5 Dependencies

- `pyproject.toml`: setuptools, `src` layout, Python ≥ 3.10. Runtime dependencies are numpy, openai, langchain-core, langchain-openai, langgraph, python-dotenv and absl-py. Dev: pytest.
- `requirements.lock` is generated with `uv pip compile --universal`. It matches the versions the tests ran against.
- v1 had no requirements file, so the dependency list was inferred from its imports.
- `.env.example` documents every environment variable the code reads.

---

## 3. Verification

| Check | Result |
|---|---|
| `pytest` | 33 passed |
| Clean import of `swarm_planner.solver`, `run_experiment.py --help`, `evaluate.py --help` | OK |
| Rebuilt test set vs v1 curated set | Identical (ids, order, content) |
| v2 verifier vs v1 verifier | Identical parsed tasks and valid-slot lists on all 400 tasks, and identical score output on **80,800** candidate answers (every half-hour slot in every day, 30 and 60 min durations, the gold answer, and an unparseable string) |
| End-to-end solve with a fake LLM | Runs through the real LangGraph and the real verifier, writes an evaluator-compatible output and a run log |

The verifier comparison was a one-off script and is not part of the repo. The same behaviour is now locked in by `tests/test_verifier.py`.

---

## 4. Decisions and differences from v1

- **CLI defaults** follow the fixed setup in PLAN.md Section 3: `google/gemma-3-4b-it`, 8 agents, 6 iterations, seed 0, `data/curated_400.json`. v1 defaulted to Qwen, 3 iterations and seed 42.
- **Provider pinning** is unchanged (`allow_fallbacks: True`, as in v1). The plan puts provider pinning and per-call provider recording in Phase 2.
- **The v2 output format** keeps v1's field names (`pred_5shot_pro`, `aco_tot_*`) so the official evaluator works on it.
- **Module names** `types.py` and `logging.py` follow the plan. They shadow stdlib names only if Python is started from inside `src/swarm_planner/`. Run from the repo root or use the installed package.

---

## 5. Known issues carried over to later phases

These are present in v2 on purpose, to keep Phase 0 a faithful port:

| Item | Phase |
|---|---|
| F5: `_normalize_text` splits names such as Doris → "Dor is" | 1 |
| F6: "earliest" ranked among all hard-valid slots, not among those meeting soft preferences | 1 |
| F7: pheromone deposit has no baseline and no bounds | 1, 3 |
| F8: the answer is the best trace of the last iteration only | 3, 4 |
| F10 / F2: logs are still the v1 format, with no pheromone snapshots, token counts or cache | 2 |
| Baseline and sampling-control runners, `configs/*.yaml`, `README.md` | 4, 6, 7 |

`configs/` and `results/` exist but are empty.

---

## 6. How to reproduce

From the v2 root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/build_datasets.py                # needs ../../natural-plan/
```

On Linux or macOS the interpreter path is `.venv/bin/python`.
