# Running Phase 4 — commands, outputs, what to expect

Everything below is run from the **v2 root** (`swarm-multi-agent-planner/swarm-multi-agent-planner-v2/`) in **PowerShell**. The code is implemented and tested (140 tests pass, see [PHASE4.md](PHASE4.md)). No live LLM call has been made for Phase 4 yet. You run the live steps; afterwards you tell me and we start Phase 5 from the logs.

## 0. Before the first run

1. `.env` (git-ignored) holds `OPENROUTER_API_KEY`. The provider is pinned to DeepInfra (no fallback), model `google/gemma-3-4b-it`.
2. Check the code once: `.venv\Scripts\python.exe -m pytest -q` → **140 passed**.
3. **Run one step at a time.** Two processes writing to the same cache file are untested.
4. Do not delete `results\cache\llm_cache.sqlite`. It holds every finished LLM call, which makes every run resumable and makes ToT-only and ToT+ACO share identical LLM answers.
5. Optional but recommended: tag the code before the first test-set run (decision D8, see Section 6).

## 1. The method that is run (frozen)

| | |
|---|---|
| Budget | 8 agents × 6 iterations = 48 traversals per task, 6 LLM calls each. **Full budget, no early stopping** (otherwise the pheromone could never act, F15). |
| ACO | no heuristic η; p ∝ (τ + 0.001)^1; τ ← 0.9·τ + deposits; a trace deposits 1 on each of its edges **only if the verifier scores its answer 1.0** (top-only, D1). |
| Reported answer (**headline rule `verified_last3`**) | among the traces of the **last 3 iterations** that the verifier scores 1.0, the **earliest slot**; if none is verified there, the best trace of the whole run (D2). |
| Everything else | derived offline from the logs by `derive_predictions.py`: best-of-all, best-of-last 3 / 2, last iteration (the v1 rule), the true frequency vote over verified traces (last 3 / 2), plain majority vote, early stopping, oracle pass@k. |
| Baseline | 1 call per task, the 5-shot prompt as the only message, temperature 0. The seed is a cache nonce, so each seed is a real provider call (D4). |
| Sampling control | 48 direct calls per task (same prompt), temperature 0.7, one nonce per sample, best by the verifier (compared with `best_of_all`, D6). |
| Verifier-as-solver | the verifier's own best slot, no LLM. Reference row: **92.75 %** on the test set (already run and checked). |
| Seeds | 0, 1, 2 |
| Test set | `data\curated_400.json` |

## 2. The three required settings

### 2.1 Single-SLM baseline (seeds 0, 1, 2) — about $0.15, a few minutes

```powershell
.\scripts\run_phase4.ps1 -Step baseline
# or one seed by hand:
.\.venv\Scripts\python.exe scripts\run_baseline.py --seed 0 --task_concurrency 8 --max_concurrent_requests 16 --max_cost_usd 0.5
```
Output: `results\runs\baseline\seed_0\` (and `seed_1`, `seed_2`).

### 2.2 ToT-only (seeds 0, 1, 2) — first fresh run about $9, all six ToT runs about $25 together

```powershell
.\scripts\run_phase4.ps1 -Step tot_only
# or one seed by hand:
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco false --seed 0 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
```
Output: `results\runs\tot_only\seed_0\` (and `seed_1`, `seed_2`).

### 2.3 ToT+ACO (seeds 0, 1, 2) — shares the cache with ToT-only, so it costs less

```powershell
.\scripts\run_phase4.ps1 -Step tot_aco
# or one seed by hand:
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco true --seed 0 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
```
Output: `results\runs\tot_aco\seed_0\` (and `seed_1`, `seed_2`).

### 2.4 The two extra rows

```powershell
.\scripts\run_phase4.ps1 -Step sampling     # sampling control, 3 seeds, about $6 (about 19,200 calls per seed)
.\scripts\run_phase4.ps1 -Step verifier     # verifier-as-solver, free (already done: 92.75 %)
```
Output: `results\runs\sampling_control\seed_<s>\` and `results\runs\verifier_solver\`.

### 2.5 Recommended order, and the whole thing in one command

1. `-Step smoke` (about $0.02, see Section 3), then look at its output.
2. `-Step baseline` → `-Step tot_only` → `-Step tot_aco` → `-Step sampling` → `-Step evaluate`.

`.\scripts\run_phase4.ps1 -Step all` runs all of it in that order (it stops at the first error). I would rather run the steps one by one for the first time.

**Total cost** ≈ **$30–35** (estimate extrapolated from the 20 dev tasks, so a real run can differ by some tens of percent). The caps are *per run*: `-TotCap 12`, `-SamplingCap 3.5`, `-BaselineCap 0.5` (change with e.g. `-TotCap 15`). **Time:** the first fresh ToT run is the slow one. My estimate is 1–3 hours (about 10 parallel calls at 2.7 s per call were measured on dev; `--task_concurrency 8 --max_concurrent_requests 32` should be faster, but rate limits at 400 tasks are untested). Later runs reuse the cache and are faster.

## 3. Smoke test (do this first)

```powershell
.\scripts\run_phase4.ps1 -Step smoke
```
Runs every mode on 2 dev tasks, seed 0, into `results\smoke_phase4\` (kept apart from the real runs), then evaluates them. Expected cost about **$0.02** (the ToT runs replay cached dev answers; the baseline and 2 × 48 samples are the only paid calls). Check:
- every command ends with `Run directory: ...` and no error;
- the progress lines show `paid $...` well below the caps;
- `results\smoke_phase4\aggregate\report.txt` exists and lists all five configurations.

## 4. While a run is going

A progress line is printed after each task:

```
[solve] 37/400 elapsed 41.2m eta 7.4h paid $1.234/$12.00 calls 10120 (hits 6700) task_id=... quality=1.000 verified_traces=14
```
- `paid $x/$cap` is what this run has paid the provider so far. When it reaches the cap the run stops with `CostCapExceeded`. The calls in flight finish, so the total can overshoot by a few cents.
- `(hits n)` are calls answered from the cache (free).
- **Stopped or crashed?** (cost cap, Ctrl+C, network, rate limit, power.) Run the **same command again**. Finished calls are in the cache, so the run replays them in minutes and continues. `predictions.json` and `summary.json` are only written when a run completes; the log (`events.jsonl.gz`) is flushed after every task, so partial progress is never lost.
- **Many 429 / rate-limit messages?** The client backs off and retries by itself. If it is slow or fails repeatedly, lower the load: `--task_concurrency 4 --max_concurrent_requests 16`. This changes speed only, never the results.
- To cap total spend differently for one run, pass `--max_cost_usd <dollars>` to the script.

## 5. Where everything is stored

```
results/
  cache/llm_cache.sqlite              every LLM request and answer (git-ignored; ship it with the release archive)
  runs/
    baseline/seed_<s>/                predictions.json  events.jsonl.gz  summary.json  config.resolved.json
    tot_only/seed_<s>/                (same files)
    tot_aco/seed_<s>/                 (same files)
    sampling_control/seed_<s>/        (same files; events hold one `samples` event per task)
    verifier_solver/                  predictions.json  summary.json  config.resolved.json
    <config>/seed_<s>/derived/        written by evaluate / derive_predictions.py:
        predictions_<rule>.json         one file per selection rule, readable by scripts\evaluate.py
        per_task.csv                    exact match and verified flag per task and rule, oracle pass@k, early-stop iterations
        selection_report.json           the same, aggregated, plus per-iteration verified share and the early-stop budget
  aggregate/phase4/
    table_main.csv                    one row per (config, rule): mean exact match ± SD over seeds, 95 % bootstrap CI, per-seed values, valid-answer rate
    per_category.csv                  exact match per (config, rule, category)
    paired_tests.csv                  paired comparisons per seed (McNemar exact) and pooled (mean over seeds, bootstrap CI)
    report.txt                        the same, readable
  smoke_phase4/                       the smoke test (safe to delete)
```
- `predictions.json` of a swarm run already uses the **headline rule**, so `python scripts\evaluate.py --data_path results\runs\tot_aco\seed_0\predictions.json` gives the headline number with the official evaluator. `derived\predictions_last_iteration.json` is the v1 rule (comparable with the paper's v1 numbers).
- `events.jsonl.gz` holds, per task and iteration: the pheromone before and after, every trace with its branch probabilities, request hashes, answer and verifier result; plus one `llm_call` event per call (tokens, cost, latency, provider, cache hit). This is what Phase 5 reads. Prompts and answers are in the cache, referenced by the 16-character request hash.

## 6. After the runs

```powershell
.\scripts\run_phase4.ps1 -Step evaluate          # derive every rule + aggregate (free, a minute)
type results\aggregate\phase4\report.txt
```
Per-rule scores with the **official** evaluator, if you want to check one by hand:
```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --data_path results\runs\tot_aco\seed_0\derived\predictions_best_of_all.json
```
**Acceptance check (D5):** `tot_only` and `tot_aco` must score above `baseline` under the headline rule `verified_last3` (mean over seeds). The expectation from the dev simulation is about 47–48 % against a baseline of about 25 %, but that is an expectation; report whatever comes out.

**What to send me for Phase 5:** just say the runs are done. The `results\runs\...` folders and `results\aggregate\phase4\` are all I need; I read the logs from disk. If a run failed, paste the last 10 lines of the console.

**Code freeze (D8).** The code is committed and tagged `phase4-freeze` in the repository root `swarm-multi-agent-planner/` (branch `project-refinement-v1`, local only, nothing pushed). Every run log records the git commit and `git_dirty` (true only if a tracked file was modified), so a run with `git_dirty: false` and this commit is reproducible from the tag. **Do not edit code between the tag and the last run.** If something must change, tell me, and we re-tag and say so in the paper. The cache, `.env`, the virtual environment and `resultsuns` are not part of the commit.
