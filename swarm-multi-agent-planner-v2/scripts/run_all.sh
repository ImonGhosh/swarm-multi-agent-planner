#!/usr/bin/env bash
# Bash twin of scripts/run_experiments.ps1: runs one step (or all) on the 400 test tasks.
#
#   scripts/run_all.sh <step>        step = smoke | verifier | baseline | tot_only | tot_aco | sampling | evaluate | all
#
# Every run reads its settings from configs/<name>.yaml (--config). Only what you set explicitly
# overrides the file, through environment variables:
#   SEEDS="0 1 2"  TASK_CONCURRENCY=  MAX_REQUESTS=  TOT_CAP=  SAMPLING_CAP=  BASELINE_CAP=
#   DRY_RUN=1      print the commands without running anything
# Example: SEEDS="0" TOT_CAP=15 scripts/run_all.sh tot_aco
#
# Every run is resumable (finished LLM calls are in results/cache/llm_cache.sqlite): after a crash or a
# cost-cap stop, run the same command again. Run ONE step at a time (one process on the cache file).
set -euo pipefail

STEP="${1:-all}"
SEEDS="${SEEDS:-0 1 2}"
cd "$(dirname "$0")/.."

if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY=".venv/Scripts/python.exe"; fi
if [ -z "${DRY_RUN:-}" ] && [ ! -x "$PY" ]; then
  echo "Missing $PY - create the virtual environment first (see PHASE0.md)." >&2; exit 1
fi

run_py() {
  if [ -n "${DRY_RUN:-}" ]; then echo "python $*"; else "$PY" "$@"; fi
}

# flags that override the config file, only for variables the caller set
overrides() {
  local out=()
  [ -n "${TASK_CONCURRENCY:-}" ] && out+=(--task_concurrency "$TASK_CONCURRENCY")
  [ -n "${MAX_REQUESTS:-}" ] && [ "$1" != "baseline" ] && out+=(--max_concurrent_requests "$MAX_REQUESTS")
  case "$1" in
    baseline) [ -n "${BASELINE_CAP:-}" ] && out+=(--max_cost_usd "$BASELINE_CAP") ;;
    sampling_control) [ -n "${SAMPLING_CAP:-}" ] && out+=(--max_cost_usd "$SAMPLING_CAP") ;;
    tot_only|tot_aco) [ -n "${TOT_CAP:-}" ] && out+=(--max_cost_usd "$TOT_CAP") ;;
  esac
  echo "${out[@]:-}"
}

run_verifier() { run_py scripts/run_verifier_solver.py --config configs/verifier_solver.yaml; }
run_baseline() {
  for s in $SEEDS; do run_py scripts/run_baseline.py --config configs/baseline.yaml --seed "$s" $(overrides baseline); done
}
run_tot() {  # $1 = tot_only | tot_aco
  for s in $SEEDS; do run_py scripts/run_experiment.py --config "configs/$1.yaml" --seed "$s" $(overrides "$1"); done
}
run_sampling() {
  for s in $SEEDS; do
    run_py scripts/run_sampling_control.py --config configs/sampling_control.yaml --seed "$s" $(overrides sampling_control)
  done
}
run_evaluate() { run_py scripts/evaluate_runs.py; }
run_smoke() {  # 2 dev tasks, every mode, seed 0, results/smoke_test/ (never mixed with the real runs)
  local out="results/smoke_test/runs" data="data/dev_100.json"
  run_py scripts/run_verifier_solver.py --config configs/verifier_solver.yaml --data_path "$data" --limit 2 --run_dir "$out/verifier_solver"
  run_py scripts/run_baseline.py --config configs/baseline.yaml --data_path "$data" --limit 2 --seed 0 --run_dir "$out/baseline/seed_0" --max_cost_usd 0.2
  run_py scripts/run_experiment.py --config configs/tot_only.yaml --data_path "$data" --limit 2 --seed 0 --run_dir "$out/tot_only/seed_0" --task_concurrency 2 --max_concurrent_requests 16 --max_cost_usd 1
  run_py scripts/run_experiment.py --config configs/tot_aco.yaml --data_path "$data" --limit 2 --seed 0 --run_dir "$out/tot_aco/seed_0" --task_concurrency 2 --max_concurrent_requests 16 --max_cost_usd 1
  run_py scripts/run_sampling_control.py --config configs/sampling_control.yaml --data_path "$data" --limit 2 --seed 0 --run_dir "$out/sampling_control/seed_0" --max_cost_usd 0.5
  run_py scripts/evaluate_runs.py --runs_root "$out" --out_dir results/smoke_test/aggregate
}

case "$STEP" in
  smoke) run_smoke ;;
  verifier) run_verifier ;;
  baseline) run_baseline ;;
  tot_only) run_tot tot_only ;;
  tot_aco) run_tot tot_aco ;;
  sampling) run_sampling ;;
  evaluate) run_evaluate ;;
  all) run_verifier; run_baseline; run_tot tot_only; run_tot tot_aco; run_sampling; run_evaluate ;;
  *) echo "Unknown step '$STEP'. Use: smoke | verifier | baseline | tot_only | tot_aco | sampling | evaluate | all" >&2; exit 2 ;;
esac
