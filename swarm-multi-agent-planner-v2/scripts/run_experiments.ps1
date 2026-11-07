# Phase 4 driver: runs one configuration (or all) on the 400 test tasks, one seed after the other.
#
#   .\scripts\run_experiments.ps1 -Step smoke          # tiny live test of every mode (about $0.02)
#   .\scripts\run_experiments.ps1 -Step verifier       # verifier-as-solver reference row (free)
#   .\scripts\run_experiments.ps1 -Step baseline       # single-call baseline, seeds 0 1 2  (about $0.15)
#   .\scripts\run_experiments.ps1 -Step tot_only       # ToT-only, seeds 0 1 2
#   .\scripts\run_experiments.ps1 -Step tot_aco        # ToT+ACO,  seeds 0 1 2
#   .\scripts\run_experiments.ps1 -Step sampling       # equal-budget sampling control (about $6)
#   .\scripts\run_experiments.ps1 -Step evaluate       # derive every rule + aggregate (free)
#   .\scripts\run_experiments.ps1 -Step all            # everything above, in the order that reuses the cache most
#
# Every run is resumable: finished LLM calls are in results\cache\llm_cache.sqlite, so after a crash,
# a rate-limit stop or a cost-cap stop, run the same command again and it continues from the cache.
# Run ONE step at a time (one process on the cache file).

param(
    [ValidateSet("smoke", "verifier", "baseline", "tot_only", "tot_aco", "sampling", "evaluate", "all")]
    [string]$Step = "all",
    [int[]]$Seeds = @(0, 1, 2),
    [int]$TaskConcurrency = 8,
    [int]$MaxRequests = 32,
    # per-run caps on what a run may PAY the provider (cache hits are free). A fresh ToT run is about $9.
    [double]$TotCap = 12.0,
    [double]$SamplingCap = 3.5,
    [double]$BaselineCap = 0.5
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Missing $Py - create the virtual environment first (see PHASE0.md)." }

function Invoke-Py {
    & $Py @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit code $LASTEXITCODE): python $args" }
}

function Run-Verifier { Invoke-Py scripts\run_verifier_solver.py }

function Run-Baseline {
    foreach ($s in $Seeds) {
        Invoke-Py scripts\run_baseline.py --seed $s --task_concurrency $TaskConcurrency --max_concurrent_requests 16 --max_cost_usd $BaselineCap
    }
}

function Run-Tot([string]$Aco) {
    foreach ($s in $Seeds) {
        Invoke-Py scripts\run_experiment.py --aco $Aco --seed $s --task_concurrency $TaskConcurrency --max_concurrent_requests $MaxRequests --max_cost_usd $TotCap
    }
}

function Run-Sampling {
    foreach ($s in $Seeds) {
        Invoke-Py scripts\run_sampling_control.py --seed $s --task_concurrency 4 --max_concurrent_requests $MaxRequests --max_cost_usd $SamplingCap
    }
}

function Run-Evaluate { Invoke-Py scripts\evaluate_runs.py }

function Run-Smoke {
    # 2 dev tasks, every mode, seed 0, results\smoke_test\ (never mixed with the real runs)
    $out = "results\smoke_test\runs"
    $data = "data\dev_100.json"
    Invoke-Py scripts\run_verifier_solver.py --data_path $data --limit 2 --run_dir "$out\verifier_solver"
    Invoke-Py scripts\run_baseline.py --data_path $data --limit 2 --seed 0 --run_dir "$out\baseline\seed_0" --max_cost_usd 0.2
    Invoke-Py scripts\run_experiment.py --data_path $data --limit 2 --aco false --seed 0 --run_dir "$out\tot_only\seed_0" --task_concurrency 2 --max_concurrent_requests 16 --max_cost_usd 1
    Invoke-Py scripts\run_experiment.py --data_path $data --limit 2 --aco true --seed 0 --run_dir "$out\tot_aco\seed_0" --task_concurrency 2 --max_concurrent_requests 16 --max_cost_usd 1
    Invoke-Py scripts\run_sampling_control.py --data_path $data --limit 2 --seed 0 --run_dir "$out\sampling_control\seed_0" --max_cost_usd 0.5
    Invoke-Py scripts\evaluate_runs.py --runs_root $out --out_dir results\smoke_test\aggregate
}

switch ($Step) {
    "smoke"    { Run-Smoke }
    "verifier" { Run-Verifier }
    "baseline" { Run-Baseline }
    "tot_only" { Run-Tot "false" }
    "tot_aco"  { Run-Tot "true" }
    "sampling" { Run-Sampling }
    "evaluate" { Run-Evaluate }
    "all" {
        Run-Verifier
        Run-Baseline
        Run-Tot "false"
        Run-Tot "true"
        Run-Sampling
        Run-Evaluate
    }
}
