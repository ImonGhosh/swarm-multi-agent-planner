# Experiment driver: runs one configuration (or all) on the 400 test tasks, one seed after the other.
#
# Every run reads its settings from configs\<name>.yaml (via --config). Only what you pass explicitly
# (-Seeds, -TaskConcurrency, -MaxRequests, -TotCap, -SamplingCap, -BaselineCap) overrides the file.
# Add -DryRun to print the commands without running anything.
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
    # The values below are NOT defaults of this script: when you leave them out, the value in
    # configs\<name>.yaml is used (speed settings and per-run caps on what a run may PAY the
    # provider; cache hits are free; a fresh ToT run is about $9).
    [int]$TaskConcurrency,
    [int]$MaxRequests,
    [double]$TotCap,
    [double]$SamplingCap,
    [double]$BaselineCap,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Bound = $PSBoundParameters   # the script's own parameters (inside a function $PSBoundParameters is the function's)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not $DryRun -and -not (Test-Path $Py)) { throw "Missing $Py - create the virtual environment first (see PHASE0.md)." }

function Invoke-Py {
    if ($DryRun) { Write-Host "python $args"; return }
    & $Py @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit code $LASTEXITCODE): python $args" }
}

# Extra flags that override the config file, only for parameters the caller actually passed.
function Get-Overrides([hashtable]$Map) {
    $flags = @()
    foreach ($name in $Map.Keys) {
        if ($Bound.ContainsKey($name)) { $flags += @($Map[$name], $Bound[$name]) }
    }
    return $flags
}

function Run-Verifier { Invoke-Py scripts\run_verifier_solver.py --config configs\verifier_solver.yaml }

function Run-Baseline {
    $extra = Get-Overrides @{ TaskConcurrency = "--task_concurrency"; BaselineCap = "--max_cost_usd" }
    foreach ($s in $Seeds) {
        Invoke-Py scripts\run_baseline.py --config configs\baseline.yaml --seed $s @extra
    }
}

function Run-Tot([string]$Config) {
    $extra = Get-Overrides @{ TaskConcurrency = "--task_concurrency"; MaxRequests = "--max_concurrent_requests"; TotCap = "--max_cost_usd" }
    foreach ($s in $Seeds) {
        Invoke-Py scripts\run_experiment.py --config "configs\$Config.yaml" --seed $s @extra
    }
}

function Run-Sampling {
    $extra = Get-Overrides @{ TaskConcurrency = "--task_concurrency"; MaxRequests = "--max_concurrent_requests"; SamplingCap = "--max_cost_usd" }
    foreach ($s in $Seeds) {
        Invoke-Py scripts\run_sampling_control.py --config configs\sampling_control.yaml --seed $s @extra
    }
}

function Run-Evaluate { Invoke-Py scripts\evaluate_runs.py }

function Run-Smoke {
    # 2 dev tasks, every mode, seed 0, results\smoke_test\ (never mixed with the real runs).
    # The same config files as the real runs, with a small dataset, small caps and their own folder.
    $out = "results\smoke_test\runs"
    $data = "data\dev_100.json"
    Invoke-Py scripts\run_verifier_solver.py --config configs\verifier_solver.yaml --data_path $data --limit 2 --run_dir "$out\verifier_solver"
    Invoke-Py scripts\run_baseline.py --config configs\baseline.yaml --data_path $data --limit 2 --seed 0 --run_dir "$out\baseline\seed_0" --max_cost_usd 0.2
    Invoke-Py scripts\run_experiment.py --config configs\tot_only.yaml --data_path $data --limit 2 --seed 0 --run_dir "$out\tot_only\seed_0" --task_concurrency 2 --max_concurrent_requests 16 --max_cost_usd 1
    Invoke-Py scripts\run_experiment.py --config configs\tot_aco.yaml --data_path $data --limit 2 --seed 0 --run_dir "$out\tot_aco\seed_0" --task_concurrency 2 --max_concurrent_requests 16 --max_cost_usd 1
    Invoke-Py scripts\run_sampling_control.py --config configs\sampling_control.yaml --data_path $data --limit 2 --seed 0 --run_dir "$out\sampling_control\seed_0" --max_cost_usd 0.5
    Invoke-Py scripts\evaluate_runs.py --runs_root $out --out_dir results\smoke_test\aggregate
}

switch ($Step) {
    "smoke"    { Run-Smoke }
    "verifier" { Run-Verifier }
    "baseline" { Run-Baseline }
    "tot_only" { Run-Tot "tot_only" }
    "tot_aco"  { Run-Tot "tot_aco" }
    "sampling" { Run-Sampling }
    "evaluate" { Run-Evaluate }
    "all" {
        Run-Verifier
        Run-Baseline
        Run-Tot "tot_only"
        Run-Tot "tot_aco"
        Run-Sampling
        Run-Evaluate
    }
}
