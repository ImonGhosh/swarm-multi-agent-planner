# v2 Plan — Swarm Multi-Agent Planner

This plan covers the v2 revision of the project. It answers the professor's feedback (`suggestions-v1.md`) and our own follow-up points. v2 starts from v1 (`../swarm-multi-agent-planner-v1/`), reuses its code, and refines it.

**Status:** Phases 0, 1 and 2 done (see [PHASE0.md](PHASE0.md), [PHASE1.md](PHASE1.md), [PHASE2.md](PHASE2.md)). Phase 3 done: analysis on a 20-task dev sample, and the v2 method is frozen ([PHASE3.md](PHASE3.md)). The main runs use the full budget with no early stopping, so that ACO acts (decisions 7 and 9, finding F15). The v1-rule row was removed. **Phase 4: code done, live runs not yet started.** Decisions 10–17 were made on 2026-11-07 (Section 6): top-only deposit, headline selection rule `verified_last3` (earliest verified slot), about $30–35 with per-run cost caps, seed as baseline nonce. The code and the evaluation pipeline are implemented and tested (140 tests; a full-budget ToT run replayed from the cache at $0 reproduces the Phase 3 simulation). The user runs the live steps: commands and output locations are in [RUN_DETAILS.md](RUN_DETAILS.md), details in [PHASE4.md](PHASE4.md). Phases 5–8 not started. Decision 5 (quality signal) is settled: option 1, the fixed deterministic verifier (see Section 6).

**Convention:** when a phase is finished, add a short **Status** line under its heading below and write the details in `PHASE<n>.md`.

---

## 1. Goals

| Source | Goal | Where it is addressed |
|---|---|---|
| Prof. Req. 1 | Complete, runnable code, with environment, configs, prompts and reproduction instructions | Phases 0, 6, 7 |
| Prof. Req. 2 | Ship the original outputs and execution logs behind every reported number | Phases 2, 4, 6 |
| Prof. Req. 3 / Ours 2 | Run baseline, ToT-only and ToT+ACO with at least 3 documented seeds and report mean ± SD | Phase 4 |
| Prof. Req. 4 / Ours 4 | Show concrete evidence that ACO influences the search (pheromone values, branch probabilities, …) | Phases 2, 5 |
| Prof. Req. 5 / Ours 11 | Revise the paper so the conclusions follow the evidence, and discuss it explicitly if ACO ≈ ToT-only | Phase 8 |
| Ours 1 | Keep `google/gemma-3-4b-it`, rely on the existing leaderboard, re-run only the 3 modes | Phase 4 |
| Ours 3 | Log every run, pheromone state and agent trace for later analysis | Phase 2 |
| Ours 5 | Analyse whether the 5-level ToT is the bottleneck that limits ACO | Phase 3 |
| Ours 6 | Analyse and, if possible, improve the heuristic desirability (η) | Phase 3 (done: η analysed, found useless, dropped) |
| Ours 7 | Analyse and improve the deterministic verifier | Phase 1 |
| Ours 8 | Analyse whether the overall per-instance n × T design is reasonable | Phase 3 |
| Ours 9 | Clean, intuitive v2 repo structure | Phase 6 |
| Ours 10 | Updated README | Phase 7 |

---

## 2. What we found in v1

The diagnostics were run offline on the 400-task curated set with v1's verifier code. No LLM calls were made.

| # | Finding | Why it matters |
|---|---|---|
| F1 | `aco_tot/engine.py:32` imports `aco_tot/quality.py`, which was deleted from git. That module is only used by the legacy `train` mode, but the top-level import makes solve mode fail with `ModuleNotFoundError` on a clean checkout. `data/calendar_scheduling_input_full.json` is also not valid JSON. | Req. 1: the shipped repo doesn't run. The fix is to remove the train/infer modes and the import. `quality.py` does **not** need to be recreated. |
| F2 | The repo only ships the outputs of the ToT+ACO run (`*_output_curated.json`, Gemma, 40.0% exact match). There are no outputs or logs for the paper's baseline (0.2475) or ToT-only (0.3950) numbers. The input file also still has leftover `pred_5shot_pro` / `pred_model` fields from an earlier Qwen-2.5-7B run, which could be mistaken for our results. | Req. 2. |
| F3 | `LLM_TEMPERATURE = 0.0` (`llm_nodes.py`). Each of the 3⁵ = 243 branch paths is (near-)deterministic per task. Seeds only control branch sampling. | A seed-replicated baseline at temperature 0 has SD ≈ 0. ToT/ACO is a search over at most 243 fixed candidates per task. Identical LLM calls can be cached. **Phase 2 correction:** temperature 0 is only *nearly* deterministic on the provider (3 distinct answers in 15 identical DeepInfra calls), so a cache freezes one realisation, and a baseline SD of exactly 0 should not be assumed. |
| F4 | The verifier enumerates every hard-valid slot. Used on its own as a solver (pick the earliest top-scoring slot), it reaches **85.8%** exact match on the 400 tasks, versus 40.0% for v1 ToT+ACO. After the Phase 1 fixes it reaches 92.75% on the test set (371/400) and 93.4% on all 1000 tasks. | The paper must frame the system as *verifier-guided generation*. We must also be ready to answer "why not just output the verifier's answer?" |
| F5 | Verifier bug: `_normalize_text` rewrites `Xis` → `X is`, which splits names: **Doris → "Dor is"**, Alexis, Dennis. That participant's calendar is silently dropped. This affects **15 / 400** tasks. | In those tasks, invalid answers get high scores and pheromone goes to the wrong paths. |
| F6 | The verifier counts "earliest" as rank 0 among *all* hard-valid slots, not among slots that also meet the soft preferences (e.g. `calendar_scheduling_example_602`). The gold answer scores < 1.0 on 14 tasks. On another 43 tasks there are several top-scoring slots and gold is not the earliest of them. | The verifier ranks gold highest on only about 86% of tasks. This caps the achievable accuracy. |
| F7 | Every valid answer scores 0.5–1.0 and invalid ones up to ~0.39. All 8 agents deposit pheromone, with no baseline subtraction and no bounds on pheromone values. | The pheromone mostly reflects how often a branch was visited, not how good its answers were, so it reinforces the heuristic's initial bias. This is the likely cause of the +0.5 pp (2 tasks) ACO gain. |
| F8 | The final answer is the best trace of the *last* iteration only. For ToT-only, 40 of the 48 traversals are discarded. | The comparison is effectively best-of-8 random vs best-of-8 ACO-biased. It must be analysed alongside best-of-all-48 selection. |
| F9 | The per-task seed is `seed + task_offset`, so seed 43 on task *i* equals seed 42 on task *i+1*. | Consecutive seeds are not independent replicates. |
| F10 | Logs keep only the last iteration's traces and the chosen branch's probability. There are no per-iteration pheromone snapshots, full probability distributions, token usage, latency, cost, provider, git commit or environment. | Requirements 2 and 4 can't be met from v1 logs. |
| F11 | Found in Phase 1: Natural Plan does **not** guarantee a single solution. Over the 1000 tasks, 166 have several equally good answers (all constraints met, no earliest request that would single one out). In 66 of them the gold answer is not the earliest of those answers (29 in the test set). The official evaluator is exact match, so any other valid answer counts as wrong. | No deterministic verifier can reach 100% on these tasks. Verifier-as-solver tops out at 93.4%, and the paper must say that exact match undercounts valid answers on these tasks. |
| F12 | Found in Phase 3 (20 dev tasks): the answer is selected from the last iteration only, so most of the search is thrown away. ToT-only with best-of-all selection reaches 48.5% against 38.3% for last-iteration selection. | Any ToT-versus-ACO comparison depends on the selection policy. The v1 ACO "gain" is an artefact of it (ACO v1 is −2.4 points under best-of-all). Both policies must be reported. |
| F13 | Found in Phase 3: the v1 deposit rule gives invalid answers a pheromone deposit and the heuristic η favours the weakest L1 branch, so the pheromone does not learn (probability on the best L1 branch 0.292 → 0.295). With a top-only deposit it reaches 0.545. | The v2 rule must not deposit for invalid answers, and η should be dropped or reported as an ablation. |
| F14 | Found in Phase 3: only L1 (and weakly L2) change correctness. L3–L5 branch choices are indistinguishable from chance, and 10 of 20 tasks have no correct path among 243 (7 never reach even a hard-valid answer). | The limit is the 4B model's slot arithmetic, not the tree or the search budget. ToT-only already nearly reaches the oracle ceiling, so ACO has little room to improve accuracy. |
| F15 | Found at the end of Phase 3: with the chosen defaults (top-only deposit, best-of-all selection, early stopping) ACO is inert. The pheromone only changes after a verified answer, and that answer ends the search. In simulation ToT+ACO v2 and ToT-only are identical (48.6% accuracy, 3.73 of 6 iterations). | Resolved by decision 9: the main runs use the full budget with no early stopping. ACO then changes the verified share per iteration, last-iteration and majority-vote accuracy and the pheromone, but not the best-of-all accuracy. The paper says that ACO adds no best-of-all accuracy at equal budget under an exact verifier. |

v1 accuracy by category (baseline from the paper, ToT+ACO from the shipped output): the gains are concentrated in categories with few valid slots (4–7 people, about 1.2–1.6 valid slots on average). The weakest categories are 2 people with 4–5 days (about 12 valid slots on average).

---

## 3. Fixed experimental setup

| Item | Value |
|---|---|
| Model | `google/gemma-3-4b-it` via OpenRouter, pinned to the provider **DeepInfra** (`allow_fallbacks: false`, bf16, 100% uptime when checked), provider recorded per call. Generation: temperature 0, top_p 1, max_tokens 400. |
| Test set | Curated 400 (first 40 tasks of each of the 10 Natural Plan categories), rebuilt with a script from `natural-plan/data/calendar_scheduling.json` and checked to match v1's set exactly |
| Dev set | 100 tasks (tasks 41–50 of each category). No overlap with the test set. **All analysis and tuning in Phase 3 uses only the dev set.** |
| Budget | n = 8 agents, T = 6 iterations (48 traversals per task, 6 LLM calls each) |
| ACO (v2 rule, frozen in Phase 3) | **No heuristic η.** Branch probability p_ij = (τ_ij + ε)^α / Σ_k (τ_ik + ε)^α with α = 1.0, ε = 0.001, τ₀ = 1.0 (uniform at the start). Update τ ← (1 − ρ)·τ + Σ Q with ρ = 0.1, where a trace deposits Q = 1 on each of its edges if the verifier scores its answer 1.0 and Q = 0 otherwise. Answer = chosen by a final selection rule that is still open (decision 11); the logs keep every trace, so any rule can be derived. **The full budget is always used: n = 8 agents × T = 6 iterations, no early stopping** (otherwise the pheromone would never act, F15). Early-stopped answers and call counts are derived offline from the logs. The v1 rule of the published paper (with η and a score deposit) is not run again. |
| Seeds | 0, 1, 2. Per-call randomness derived via `numpy.random.SeedSequence` from (seed, task_id, iteration, agent, level). |
| Metric | Natural Plan exact match (official evaluator, unchanged) |
| Quality signal (Q) | Fixed deterministic verifier with the Phase 1 fixes (decision 5, option 1). Frozen before any test-set run. |
| Training set | Not used. The reward-model option was dropped (decision 5). |

---

## 4. Phases

### Phase 0 — Set up the v2 repo and fix what's broken (no LLM cost)
**Status: done.** v1 code ported into `src/swarm_planner/`, `train`/`infer`/`quality` code removed (F1), datasets rebuilt and verified identical to v1's curated set, per-call seeding added (F9), dependencies pinned, 33 tests passing. Details: [PHASE0.md](PHASE0.md).

- Port the v1 code into the new layout (Section 5).
- Remove the `train`/`infer` modes, the `quality` import, `store.py` and other dead code.
- Make the existing tests pass.
- `scripts/build_datasets.py`: build `curated_400.json` and `dev_100.json` from the Natural Plan source, without leftover prediction fields, and check that the curated set matches v1 exactly.
- `seeding.py`: per-call seed derivation (fixes F9).
- Pin dependencies: `pyproject.toml` plus a lock file.

### Phase 1 — Audit and fix the verifier (our point 7, no LLM cost)
**Status: done.** Gold now scores 1.0 and is a top-scoring slot on all 1000 tasks. 934/1000 (93.4%) of the verifier's own answers match gold, 92.75% on the test set. The plan's assumption that Natural Plan has a single solution per task was wrong (F11), so 66 tasks are documented exceptions. The deposit mapping exists as a function but is not wired into the pheromone update yet, which is Phase 3. 48 tests pass. Details: [PHASE1.md](PHASE1.md).

The fixed verifier is the quality signal for ACO (decision 5, option 1), so its correctness is central to every result.
- Fix the name-splitting normalisation (F5). **Done.**
- Score in strict priority order: hard constraints → soft preferences → earliest, where "earliest" is judged among slots that meet the soft preferences (F6). **Done.**
- **Check the verifier against all 1000 gold answers** (`scripts/audit_verifier.py`). Original target: gold scores 1.0 and is the only top-scoring slot. **Corrected target:** gold scores 1.0 and is a top-scoring slot, and is the only one whenever the task asks for the earliest availability. Tasks with several equally good answers are classified separately (F11).
- Test: every gold answer in the 1000 tasks is a top-scoring slot, and the tasks where gold is not the verifier's first-ranked top slot match the list in `data/verifier_known_exceptions.json` (`tests/test_verifier_gold.py`). **Done.**
- Revisit how verifier scores map to pheromone deposits (Q), so that invalid answers deposit nothing or a clearly separated amount (feeds F7). **Partly done:** `deposit_quality(result, mode)` offers `v1_score` and `valid_only`. Choosing and wiring one in is part of the Phase 3 update rule.
- Report the verifier's precision and recall against gold, and its "verifier-as-solver" accuracy as a reference row. **Done**, see [PHASE1.md](PHASE1.md).

### Phase 2 — Logging, response cache and cost tracking (our point 3)
**Status: done.** One gateway now makes every LLM call (cache, raw SDK, cost and provider recorded), and each run writes `events.jsonl.gz`, `predictions.json`, `summary.json` and `config.resolved.json` to its run directory. A full-budget live smoke test (6 dev tasks) and a cache replay worked: the replay made 0 API calls and gave identical predictions. Temperature 0 turned out **not** to be deterministic, so the cache freezes one realisation (see F3). 74 tests pass. Details: [PHASE2.md](PHASE2.md).

- Write one compressed JSONL file per run (`events.jsonl.gz`) with these events. **Done.**
  - `run_start`: the full resolved config, git commit, installed packages (like `pip freeze`), timestamp, dataset hash, branch and edge order.
  - `iteration`: pheromone values before and after the update, the full branch probability distribution per agent and level, and **all** traces with verifier details.
  - `llm_call`: request hash, tokens in/out, cost, latency, provider, cache hit.
  - `task_result`, `task_error` and `run_end`.
- Store each prompt once and reference it by hash. **Done, in the cache:** events hold a 16-character request hash, and the LLM cache stores the full request and answer once. The log stays small (about 8 KB per task, so 3–4 MB per 400-task run).
- Add an **LLM response cache** (SQLite, keyed by model, generation parameters, messages and an optional nonce). **Done.**
  - Temperature 0 on the provider is not bit-deterministic (3 distinct answers in 15 identical calls). The cache therefore does not leave results unchanged: it makes them repeatable by fixing the first answer to each request. Stochastic runs (temperature > 0) pass a per-sample nonce so samples don't collapse.
  - It cuts cost: in the smoke test 64% of the calls of a single run were already shared (agents with the same path prefix, and concurrent identical requests share one provider call).
  - ToT-only and ToT+ACO answer identical requests from the same cache, so they are compared on the same LLM outputs (fairer comparison).
  - Shipping the cache lets the professor replay every run without API calls (Req. 2).
- Pin the provider and record it per call. **Done:** `DeepInfra`, `allow_fallbacks: false`.
- Task-level concurrency (`--task_concurrency`) and a global request cap (`--max_concurrent_requests`). **Added**, because a sequential run was too slow. They change speed only.

### Phase 3 — Path-table analysis on the dev set (our points 5, 6, 8)
**Status: done. The analysis is complete and the v2 method is frozen (decisions 6–8 in Section 6); the main runs use the full budget with no early stopping so that ACO acts (decision 9, finding F15).** The path table was run on 20 dev tasks (2 per category, 4,860 paths, 12,120 calls, $1.59, 24 minutes) because the full 100-task table was over budget. The analyses and simulations are in [PHASE3.md](PHASE3.md) and `results/path_table/dev_20/analysis/`. With 20 tasks (10 of them solvable) the conclusions are a preview with wide intervals, and Phase 4 is the real check. Decisions made: temperature stays 0, no 3-level ToT ablation, 20-task dev run with no further dev tests, heuristic η dropped, full budget with no early stopping in the main runs, v1-rule row removed; the top-only deposit and the final selection rule were reopened for decision before Phase 4 (decisions 10 and 11). 98 tests pass. The code changes that implement the frozen method (removing η, best-of-all selection, the top-only deposit, the baseline and sampling-control runners) are Phase 4 preparation and are not made yet. Early stopping needs no solver change: it is derived offline from the logs.

**Headline findings** (details in PHASE3.md):
- Generation is the limit: 10 of 20 tasks have no correct path among the 243, and ToT-only with the full budget (48.6%) already almost reaches the oracle ceiling (50%).
- Last-iteration selection discards most of the search: selecting over all iterations raises ToT-only by **+9.8 points [+3.5, +16.7]** at no cost. The v1 ACO "gain" disappears under best-of-all selection (ACO v1: −2.4 [−6.6, −0.1]).
- Only L1 (and weakly L2) matter; L3–L5 are indistinguishable from chance. η gives no benefit and favours the worst L1 branch in 7 of 10 informative tasks.
- With the v1 deposit the pheromone does not learn (0.292 → 0.295 on the best L1 branch); with a top-only deposit it does (→ 0.545), but an exact verifier plus best-of-all selection leaves ACO no room to beat ToT-only.
- Early stopping at the first verified answer keeps accuracy and uses 62% of the budget.

- `scripts/enumerate_paths.py`: run **all 243 paths for each dev task**, round-robin over the categories (606 calls per task, cached). **Done on 20 tasks** (scope reduced from 100). This gives a table of verifier scores and correctness for every path of every task. It writes `path_table.csv.gz` (one row per path), `nodes.jsonl.gz` (the answer text of every call) and a full event log.
- `scripts/simulate_aco.py` (core in `src/swarm_planner/simulate.py`): re-run any ToT/ACO setting offline from that table, over hundreds of seeds, at no LLM cost. **Done.** Checked against the real solver replayed from the cache (`scripts/validate_simulator.py`): 75.3% vs 76.9% predicted for ToT-only, 77.7% vs 77.0% for ACO v1. `scripts/analyze_path_table.py` produces the analysis report.
- **Is the ToT structure the bottleneck? (point 5)**
  - Split tasks into three groups: no path correct / every path correct / some paths correct. Only the last group can ACO improve at all. **Result:** 10 / 0 / 10 of 20 tasks. Seven of the 10 unsolved tasks never produce even a hard-valid answer (the 2-person 3–5 day tasks are all unsolved).
  - Keep "correct" (exact match to gold) apart from "verifier score 1.0". On the 166 tasks with several top-scoring answers (F11) a path can score 1.0 and still be marked wrong, and the verifier cannot tell which answer is the gold.
  - Measure each level's effect on correctness (accuracy per branch, and interactions between levels).
  - Identify levels that make no difference (L5 and the extra finalize step are likely). **Result:** L3, L4 and L5 branch choices are indistinguishable from chance (p = 0.33, 0.34, 0.58); L1 matters (constraint-first +6.6, preference-first −4.5 points) and L2 weakly. The finalize call is not redundant (it changes the L5 slot in about 23% of paths).
  - Find where errors first appear, using the intermediate outputs (likely the slot arithmetic in L3). **Result:** the model often does not carry out the computation: 19% of L3 answers already announce a final slot, and the slot stated at L3, L4 and L5 equals gold in only 9%, 11% and 12% of paths. The limit is the 4B model's slot arithmetic, not the tree.
- **Heuristic η (point 6)**
  - Check whether η matches the actual branch success rate per task type. **Result:** it does not. An η-guided agent is 12.9% accurate against 13.1% for a uniform one, and at L1 η's favourite is the worst branch in 7 of 10 informative tasks.
  - Ablations: β = 0 (pheromone only) and α = 0 (heuristic only). **Result:** α = 0 (η only) is no better than ToT-only; β = 0 is the best ACO variant (+4.1 with the v1 deposit and +9.6 with a top-only deposit, under last-iteration selection).
  - Note that the L5 heuristic is the same for every task.
  - Option: estimate η from dev-set branch success rates. **Open:** with 20 tasks only level 1 can be estimated, so the evidence-based η would apply there only (1.0 elsewhere), or η is dropped.
  - Option checked and not recommended: a verifier look-ahead on intermediate answers (pruning and early credit). The signal is strong (a verified L3 slot ends correct in 74% of paths), but pruning gains only +0.4 to +0.6 points at equal calls and wrongly drops 13–22% of correct paths, and the early credit gives +0.0 to +0.1 points. See [PHASE3.md](PHASE3.md) Section 5b.
- **Overall approach (point 8)**
  - Final-iteration best vs best over all iterations. **Result:** +9.8 points [+3.5, +16.7] for ToT-only in favour of best-of-all.
  - Stopping early once a trace scores 1.0. **Result:** same accuracy (+0.4 [0.0, +1.1]) with 62% of the budget.
  - How the n × T budget is split between agents and iterations. **Result:** irrelevant under best-of-all selection (48.8% for every split); under last-iteration selection ToT-only collapses to 13% at 1 × 48 while ACO with an informative deposit stays near 48%.
  - Accuracy and pass@k as the budget grows. **Result:** best-of-all accuracy 38.5% at 8 traversals, 45.0% at 16, 46.9% at 24, 48.9% at 48 (ceiling 50%).
  - Reusing pheromone learned on dev tasks as the starting point for test tasks.
- **Quality signal:** fixed (decision 5, option 1). The path table scores every path with the Phase 1 verifier. No signal comparison is run.
- **Decision:** choose the v2 ACO update rule and freeze it, together with the verifier, before any test-set run. Update-rule candidates:
  - deposit by the iteration's best trace only, or by quality minus the iteration mean;
  - Max-Min Ant System bounds [τ_min, τ_max] (not simulated: with 20 tasks it cannot be separated from the others).
  - **Status: decided.** No heuristic η (decision 6), top-only deposit assumed and to be confirmed (decision 10), the final selection rule open (decision 11), full budget with no early stopping in the main runs (decisions 7 and 9). The frozen specification is in the fixed-setup table (Section 3) and PHASE3.md Section 5. Not simulated or skipped: Max-Min bounds (not separable from the others with 20 tasks), and reusing pheromone learned on dev tasks as a prior (dev-only, 20 tasks, and moot without η).
- ~~Optional exploration: a 3-level ToT variant with verifier scoring after L3.~~ **Dropped** (decision 3). The 5-level tree stays as it is, and Phase 3 only analyses it.

### Phase 4 — Main experiments (our points 1, 2)
**Status: code done (decisions 10–17 made), live runs not yet started.** Implemented: η removed from the live search, top-only deposit, selection rules (`selection_rules.py`), baseline / sampling-control / verifier-as-solver runners, a hard per-run cost cap, offline derivation of every rule from the logs (`derive.py`), aggregation with SD over seeds, bootstrap CIs and exact McNemar (`scripts/evaluate_runs.py`), and a one-command driver (`scripts/run_experiments.ps1`). 140 tests pass. Verifier-as-solver is already run on the test set (92.75%). How to run everything, where results go and what to expect: [RUN_DETAILS.md](RUN_DETAILS.md). Implementation record and the decisions with their reasoning: [PHASE4.md](PHASE4.md).

Run all configurations on the 400 test tasks with seeds {0, 1, 2}:

| Configuration | Role |
|---|---|
| Single-SLM baseline | Required (Req. 3) |
| ToT-only (`aco=false`) | Required (Req. 3) |
| ToT+ACO (frozen v2 method) | The main ToT+ACO configuration (Section 3). **Required (Req. 3).** The v1 rule of the published paper is no longer run. |
| Equal-budget sampling control: 48 direct samples at temperature 0.7, best one picked by the verifier | Separates the effect of the ToT structure from "sampling + verifier" |
| Verifier-as-solver | Reference row (F4). 92.75% on the test set after Phase 1. |

- ToT-only, ToT+ACO and the sampling control all select their answer with the same verifier, so the comparison stays fair. The verifier-as-solver row shows how much of the result the verifier alone explains.
- **Final selection rule: open (decision 11).** Simulated on the dev sample (ToT-only / ToT+ACO): best of all iterations 48.6 / 48.6 (equal: the best answer is secured at the first success); best of the last 3 iterations 46.6 / 48.3; last 2 44.5 / 48.1; last iteration 38.2 / 47.7; verified vote over the last 3 / 2 iterations 46.6 / 48.3 and 44.5 / 48.1 (the user's current preference); plain majority vote 13.3 / 36.3. The same rule must be applied to both modes, and several rules are reported side by side with the headline named openly.
- **Where ACO is expected to differ** (dev simulation, 20 tasks, wide intervals), and what Phase 4 reports next to the headline:
  - the share of verified traces per iteration (13% → 32% by iteration 6 for ACO, flat at 13% for ToT-only);
  - last-iteration accuracy (+9.4 points) and majority-vote accuracy over the 48 traces, which needs no verifier at selection time (13% → 36%);
  - the pheromone itself (probability on the best level-1 branch 0.333 → 0.66).
- Other selection and stopping policies are computed offline from the same logged runs and reported as comparison rows: final-iteration best, majority vote, early stopping (answers and the calls it would use), and oracle pass@k.
- **Seeds and paired comparison.** ToT-only and ToT+ACO are compared on the same tasks and seeds. A paired test per seed is reported for each measure.
- Statistics: mean ± SD over seeds, 95% bootstrap CI over tasks, paired tests between configurations (McNemar per seed). Report per-category tables.
- Order the runs so they reuse the cache as much as possible.

### Phase 5 — Analysis and figures showing ACO's effect (Req. 4, our point 4)
- How pheromone values change per level over iterations: mean over tasks with confidence bands, plus heatmaps.
- How branch-selection probabilities and their entropy move away from uniform (1/3).
- **Key figure:** whether the pheromone ranking matches the true branch quality from the path table, i.e. whether ACO moves toward branches that actually work.
- Best and mean score per iteration, ACO vs ToT-only.
- The share of verified (score 1.0) traces per iteration, ACO vs ToT-only, and the accuracy of last-iteration and majority-vote selection. These are where ACO is expected to differ (Phase 3 simulation).
- The iteration where the first correct answer appears.
- How many distinct paths are explored.
- Accuracy per category.
- Cost table: LLM calls, tokens, latency and USD per configuration. The baseline uses 1 call per task and the swarm about 288.
- Every figure and table is produced by `scripts/make_figures.py` / `scripts/make_tables.py` from the shipped logs.

### Phase 6 — v2 repo structure (our point 9)
```
swarm-multi-agent-planner-v2/
  README.md  PLAN.md  pyproject.toml  requirements.lock  .env.example
  configs/        baseline.yaml  tot.yaml  tot_aco.yaml  tot_aco_v2.yaml  sampling_control.yaml
  data/           curated_400.json  dev_100.json          (built by scripts/build_datasets.py)
  src/swarm_planner/
    tot/          levels.py  graph.py  context.py
    aco/          selection.py  pheromone.py  heuristics.py
    verifier/     parser.py  scorer.py
    llm/          client.py  gateway.py  cache.py  resilience.py
    baseline.py  solver.py  logging.py  runinfo.py  seeding.py  types.py
  scripts/        run_experiment.py  run_all.ps1  run_all.sh  evaluate.py
                  build_datasets.py  enumerate_paths.py  simulate_aco.py
                  analyze.py  make_figures.py  make_tables.py
  tests/          (adds tests: verifier on all 1000 gold answers, seeding, cache, logging)
  results/
    runs/<config>/seed_<s>/   predictions.json  events.jsonl.gz  summary.json  config.resolved.json
    aggregate/                tables/*.csv  figures/*.pdf
  paper/          LaTeX source, figures, compiled PDF
```
- Do not carry over: `__pycache__`, train/infer code, `store.py`, stale test datasets, leftover prediction fields in input files.
- Keep the official Natural Plan evaluator with its Apache-2.0 header.
- The logs and the cache are large. Publish them as a GitHub Release or Zenodo archive with checksums and link them from the README.

### Phase 7 — README (our point 10)
- Quickstart: install, `.env`, smoke test.
- One command per configuration and seed, plus `run_all` for the full reproduction.
- A **cache-replay demo** that reproduces the main table in about a minute without API calls, for the reassessment meeting.
- Updated explanations of the method, verifier, ACO rules and logging format.
- Expected outputs with checksums.

### Phase 8 — Paper revision (our point 11, Req. 5)
- Methods: name the model, all hyperparameters, temperature, seeds, provider and run dates. Frame the system as verifier-guided generation and include the verifier validation.
- Results: the new mean ± SD table, ACO evidence figures, cost table and per-category results.
- Discussion: if ACO does not consistently beat ToT-only, say so explicitly and explain why, using the path-table analysis.
- Limitations: mention that Natural Plan's exact-match metric rejects valid answers on tasks with several equally good answers (F11), which caps every system below 100%. This is a benchmark property, but the counts are our measurements, so word it as "we observe" (see [PHASE1.md](PHASE1.md) Section 3.6). Report the valid-answer rate (verifier score 1.0) next to exact match, and read the Natural Plan paper on how gold is generated first. Also state that the fixed verifier is a near-perfect signal and effectively a solver (F4), so it is a weak test of ACO's credit assignment. Name the learned reward model and the feasibility-only checker as future work.
- Fix the existing text issues:
  - duplicated sentences (Introduction, TravelPlanner);
  - "Guerice" → "Guericke";
  - "contributions of this paper are :";
  - mixed British/American spelling;
  - Related Work written as a numbered list.
- Add peer-reviewed references:
  - Stützle & Hoos 2000 (Max-Min Ant System);
  - Dorigo & Stützle 2004 (*Ant Colony Optimization*);
  - Yao et al., NeurIPS 2023 (Tree of Thoughts);
  - Wang et al., ICLR 2023 (self-consistency).

---

## 5. Cost estimate

| Item | LLM calls (uncached) |
|---|---|
| One ToT / ToT+ACO run (400 tasks × 48 traversals × 6 calls) | ~115k |
| Dev path table (100 tasks × ≤606 distinct calls) | ~60k |
| Baseline, per seed | 400 |
| Sampling control, per seed | ~19k |

With the shared cache, the 3 seeds × 3 ToT-based configurations should cost roughly as much as a few uncached runs.

**Measured in Phase 2** (DeepInfra, dev tasks 40–45, full 8 × 6 budget): DeepInfra charges $0.05 per million input and $0.10 per million output tokens. A fresh ToT+ACO run needs about 105 distinct provider calls per task, which cost about $0.012–0.014 per task, so roughly **$5–6 for a 400-task run**. The same calls without any caching would cost about $0.031 per task (about $12 per run). The average paid call took 2.7 s, and about 10 calls ran in parallel at `--max_concurrent_requests 32`, which puts a fresh 400-task run at roughly 3 hours. The dev path table costs about **$0.086 per task** (606 calls; my first estimate of $0.067 came from one cheap task): about $9 for all 100 dev tasks, so it was reduced to 20 tasks (actual: $1.59, 24 minutes). **Phase 4 is the big cost.** With the full budget and no early stopping, one 400-task ToT run costs about $9 (about $0.022 per task for the distinct requests of 48 traversals). The six ToT-based runs (ToT-only and ToT+ACO, 3 seeds each) share the cache and cost about **$25** together (simulated from the dev table's real per-request costs, `scripts/simulate_aco.py` Section 7). The temperature-0.7 sampling control (48 samples per task, 3 seeds, about 58k single calls) adds roughly $5–7, and the single-call baseline about $0.1–0.3. **Total about $30–35**, a rough estimate to confirm before Phase 4 (see [PHASE3.md](PHASE3.md) Section 7).

---

## 6. Open decisions

1. **Temperature. DECIDED: keep 0, as in v1.** Seeds then control branch sampling only, the baseline SD is close to 0 (the provider is not perfectly deterministic at 0, see F3), and the temperature-0.7 sampling control shows the effect of stochastic sampling. Alternative: use a higher temperature everywhere (costs more and can't be cached).
2. **ACO v2 variant in the main table. DECIDED:** v2 is the main ToT+ACO row. The v1-rule row is removed (decision 9).
3. **ToT redesign scope. DECIDED: no redesign ablation.** The 5-level tree stays as the main setup, and Phase 3 only analyses it.
4. **Paper source.** Is the LaTeX source available? If not, rebuild it from the PDF text in the SCITEPRESS-style template.
5. **Quality signal for ACO. DECIDED: option 1, fixed deterministic verifier.** v1's approach with the Phase 1 fixes. The paper frames the system as verifier-guided generation, and ACO's contribution is finding a verifier-approved answer faster at equal budget.
   - **Why:** the professor gave no specific feedback on the verifier. A reward model is not feasible in the time left. A feasibility-only checker would likely score below v1, which the professor may see as a regression.
   - **Dropped:** option 2 (feasibility checker + agreement) and option 3 (learned reward model, ~300k extra calls and a training step).
   - **Known limitation, to state in the paper:** the verifier enumerates every valid slot, so it is effectively a solver (F4, 85.8% alone) and a near-perfect signal. That makes it a weak test of ACO's credit assignment, and the paper must say so under Limitations and report the verifier-as-solver row.
6. **Heuristic desirability η. DECIDED (Phase 3): dropped altogether.** The branch-selection formula becomes p_ij = (τ_ij + ε)^α / Σ_k (τ_ik + ε)^α, with no β. Reason: on the 20-task dev sample η gave no benefit (12.9% vs 13.1% for a uniform agent) and at level 1 it favoured the worst branch in 7 of 10 informative tasks. The η ablation is reported in the paper from the dev simulation; no test-set η variant is run. The heuristic code, its tests and the β parameter are removed in Phase 4 preparation.
7. **Early stopping. DECIDED (Phase 3, revised after F15): not used in the main runs**, because it stops the search before ACO can act. It stays as an offline analysis derived from the logs (it keeps accuracy and would use about 62% of the budget). The **final selection rule is a separate, open decision (11)**. Earlier in Phase 3 best-of-all was adopted as the headline (+9.8 points over last-iteration selection for ToT-only), and that was reopened because best-of-all gives ACO no effect on the headline.
8. **More dev tests. DECIDED: none.** The Phase 3 conclusions stand on the 20-task dev sample, with the uncertainty stated in PHASE3.md.
9. **Phase 4 run design for ACO. DECIDED: full budget with no early stopping in the main runs, and the v1-rule row is removed.** The reasoning (finding F15): With a top-only deposit, early stopping and best-of-all selection, ACO cannot act: the pheromone only changes after a verified answer, which ends the search. In simulation ToT+ACO v2 and ToT-only give identical accuracy (48.6%) and cost (3.73 of 6 iterations). Req. 4 (evidence that ACO influences the search) then has no pheromone movement to show. Options: (a) run ToT+ACO with the full budget and no real early stopping, derive the early-stopped answers and call counts offline from the logs, and report best-of-all as the headline (recommended; costs more than real early stopping); (b) real early stopping for all runs and show the ACO evidence only from the dev simulation (cheaper, but a weak answer to Req. 4); (c) an informative deposit that fires before success, for example valid-only (simulated at −1.7 points under best-of-all). **Decided: option (a).** Because the v1-rule row is gone, η needs no legacy option in the code. With the full budget ACO changes the runs in measurable ways (the verified share per iteration, last-iteration accuracy, majority-vote accuracy, the pheromone), but not the best-of-all accuracy, which stays equal to ToT-only. Still to confirm before Phase 4: the top-only deposit (assumed, my recommendation).

10. **Top-only deposit. DECIDED (2026-11-07): yes.** A path deposits pheromone only if its answer scores 1.0. *Yes (recommended):* the pheromone learns from perfect answers only, the simplest rule, and it starts moving after the first success. *No:* valid-only (earlier signal, accuracy within noise, −1.7 under best-of-all), the v1 score (no learning, −2.4), or best-of-iteration and score-minus-mean (no better). Details: [PHASE4.md](PHASE4.md) D1.
11. **Final prediction selection rule. DECIDED (2026-11-07): headline `verified_last3`** = the earliest slot among the traces the verifier scores 1.0 in the last 3 iterations, falling back to the best trace of the whole run; the window is fixed before the test runs; the true frequency vote (ties to the earliest slot), best-of-all, best-of-last 3 / 2, the v1 last-iteration rule, plain majority vote and early stopping are reported next to it. With the earliest-slot tiebreak this equals best-of-last-k restricted to verified traces, and the paper says so. *Original discussion, for the record:* the user's preferred approach was the verified vote** (only traces the verifier scores 1.0 vote, over the last 2 or 3 iterations; a vote counts traces per answer; not yet final, see the caveats in PHASE4.md D2a–D2c). Choose the headline and what is reported next to it, with the same rule for both modes. Simulated ToT-only / ToT+ACO on dev: best of all 48.6 / 48.6 (fair, ACO no effect on this number); best of the last iteration 38.2 / 47.7 (ACO +9.4, ToT-only handicapped); best of the last 2 or 3 iterations 44.5 / 48.1 and 46.6 / 48.3 (compromise); **verified vote** over the last 3 iterations 46.6 / 48.3 (ACO +1.7) and over the last 2 iterations 44.5 / 48.1 (ACO +3.7), both modes far above the baseline (the preferred approach); plain majority vote 13.3 / 36.3 (verifier-free, ACO +23, but ToT-only falls below the single-agent baseline, so only a secondary row). Caveats of the verified vote: it equals best-of-last-k on this sample and differs only on the 17.5% of tasks with several verified answers, where picking by frequency instead of the earliest slot could cost up to about 3 points; the window should be fixed before the test runs. Details: [PHASE4.md](PHASE4.md) D2. How to justify the choice to the professor, and why both modes should stay above the baseline: [PHASE4.md](PHASE4.md) Section 3b.
12. **Phase 4 budget. DECIDED: about $30–35 accepted, with per-run caps (`--max_cost_usd`; 12 / 3.5 / 0.5 dollars for ToT / sampling / baseline runs).** About $30–35 in total (six ToT-based runs about $25, sampling control $5–7, baseline under $0.5). Options: a smaller test subset (for example 200 tasks, about half) or fewer sampling-control seeds. Details: [PHASE4.md](PHASE4.md) D3.
13. **Baseline seed handling. DECIDED: the seed is the cache nonce.** Use the seed as a cache nonce (each seed a real call, SD reflects provider noise, recommended) or accept identical replays (SD exactly 0). [PHASE4.md](PHASE4.md) D4.
14. **Acceptance criterion against the single-agent baseline. DECIDED: both ToT modes above the baseline under the headline rule, mean over seeds.** "Both modes score above the baseline" is an empirical outcome, so define the criterion (which selection rule, per mode, mean over seeds). Optional check of the baseline on the 20 dev tasks for about a cent. [PHASE4.md](PHASE4.md) D5.
15. **Sampling control comparison. DECIDED: compared with best-of-all (and the headline), stated openly.** The control picks the best of 48 samples by the verifier (like best-of-all). Decide how it is compared if another rule is the headline. [PHASE4.md](PHASE4.md) D6.
16. **Valid-answer rate as a secondary column. DECIDED: yes** (`verified_answer_rate`). It shows how much of any miss comes from tasks with several valid answers (F11). [PHASE4.md](PHASE4.md) D7.
17. **Git commit and tag before the first test-set run. DECIDED: yes** (see RUN_DETAILS.md Section 6). v2 is currently untracked. [PHASE4.md](PHASE4.md) D8.

Not blockers for Phase 4: parallel processes on one cache file and rate limits at 400 tasks are untested (run one configuration at a time, start with a smoke test); reading the Natural Plan paper on how gold answers are generated and the LaTeX source question (decision 4) matter at the paper stage.

---

## 7. Execution order

Phases 0 → 1 → 2 need no LLM calls and can start immediately. Phase 3's enumeration needed decision 1 (settled: temperature 0). Phase 4 needs the frozen v2 ACO rule from Phase 3 and the frozen Phase 1 verifier. Phases 5–8 follow from the logged results.
