# Running the experiments (compact)

Run everything from `swarm-multi-agent-planner-v2/` in PowerShell, **one command at a time**. Wait for `Run directory: ...` before the next one. Model: `google/gemma-3-4b-it` (default). Test set: `data\curated_400.json`. Budget: 8 agents × 6 iterations (same as v1), seeds 0, 1, 2. More detail: [RUN_DETAILS.md](RUN_DETAILS.md).

Before the first run: `.env` holds `OPENROUTER_API_KEY`; `.\.venv\Scripts\python.exe -m pytest -q` shows 140 passed; optional smoke test `.\scripts\run_experiments.ps1 -Step smoke` (about $0.02, writes to `results\smoke_test\`).

## 1. Baseline (1 call per task) — about $0.15

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --seed 0 --task_concurrency 8 --max_concurrent_requests 16 --max_cost_usd 0.5
.\.venv\Scripts\python.exe scripts\run_baseline.py --seed 1 --task_concurrency 8 --max_concurrent_requests 16 --max_cost_usd 0.5
.\.venv\Scripts\python.exe scripts\run_baseline.py --seed 2 --task_concurrency 8 --max_concurrent_requests 16 --max_cost_usd 0.5
```

## 2. ToT-only — first run about $9, all six ToT runs about $25 together (run this before ToT+ACO)

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco false --agents_per_task 8 --iterations_per_task 6 --seed 0 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco false --agents_per_task 8 --iterations_per_task 6 --seed 1 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco false --agents_per_task 8 --iterations_per_task 6 --seed 2 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
```

## 3. ToT+ACO — reuses the cache of step 2, so it costs less

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco true --agents_per_task 8 --iterations_per_task 6 --seed 0 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco true --agents_per_task 8 --iterations_per_task 6 --seed 1 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
.\.venv\Scripts\python.exe scripts\run_experiment.py --aco true --agents_per_task 8 --iterations_per_task 6 --seed 2 --task_concurrency 8 --max_concurrent_requests 32 --max_cost_usd 12
```

## 4. Evaluate (free)

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_runs.py
type results\aggregate\main\report.txt
```
One run with the official evaluator: `.\.venv\Scripts\python.exe scripts\evaluate.py --data_path results\runs\tot_aco\seed_0\predictions.json`

## 5. Optional: sampling control (not required; about $6, 30–60 min per seed)

48 direct calls per task at temperature 0.7, best one picked by the verifier. Tests whether the ToT structure matters or "many samples + verifier" is enough. Run after steps 1–3, then re-run step 4.

```powershell
.\.venv\Scripts\python.exe scripts\run_sampling_control.py --seed 0 --task_concurrency 4 --max_concurrent_requests 32 --max_cost_usd 3.5
.\.venv\Scripts\python.exe scripts\run_sampling_control.py --seed 1 --task_concurrency 4 --max_concurrent_requests 32 --max_cost_usd 3.5
.\.venv\Scripts\python.exe scripts\run_sampling_control.py --seed 2 --task_concurrency 4 --max_concurrent_requests 32 --max_cost_usd 3.5
```

## Where the results go

```
results/runs/
  baseline/seed_<s>/          predictions.json  events.jsonl.gz  summary.json  config.resolved.json
  tot_only/seed_<s>/          (same files; predictions.json already uses the headline rule verified_last3)
  tot_aco/seed_<s>/           (same files)
  sampling_control/seed_<s>/  (same files; optional)
  verifier_solver/            reference row, already done (92.75 %)
  <config>/seed_<s>/derived/  created by evaluate: predictions_<rule>.json (one per selection rule), per_task.csv, selection_report.json
results/aggregate/main/     table_main.csv  per_category.csv  paired_tests.csv  report.txt
results/cache/llm_cache.sqlite  every LLM call (do not delete; ship with the results)
```
`events.jsonl.gz` holds the pheromone per iteration, every trace with its branch probabilities and verifier result, and every LLM call (tokens, cost, latency, provider). It is the input for Phase 5.

## If something goes wrong

- **A run stops** (cost cap, crash, rate limit, Ctrl+C): run the **same command again**. Finished calls are in the cache, so it resumes quickly.
- **Many rate-limit messages or very slow:** use `--task_concurrency 4 --max_concurrent_requests 16`. This changes speed only, not results.
- **Cost cap hit** (`CostCapExceeded`): raise `--max_cost_usd` and re-run the same command.
- **Do not** run two commands at the same time (one process on the cache file), and do not edit code between the `experiments-freeze` tag and the last run.
- **Seeds:** at temperature 0 with the cache, ToT seeds vary only the branch choices, not the model's own noise. Baseline seeds are real separate calls.
