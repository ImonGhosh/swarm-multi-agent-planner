# Phase 6 — v2 repo structure (our point 9)

**Status:** DONE under option A (decided 2026-11-18): the frozen core code stays where it is, and the structure work is limited to changes that cannot alter a run. Added: `configs/` with `--config` support in the run scripts, a bash driver, a checksum manifest for the release archive, and updated run docs. 163 tests pass. **Not done on purpose:** `paper/` (no LaTeX source yet, decision 4 in PLAN.md) and the README (Phase 7). **To do by the user:** commit the changes and create the tag `config-support` (Section 4).

## 1. Decisions

| # | Question | Decision |
|---|---|---|
| 18 | How far to restructure `src/` | **Option A:** do not move or rename modules. The analysis modules stay at the package top level and `legacy_eta.py` stays where it is; the README explains both (Phase 7). Reason: the runs were made with the code at tag `experiments-freeze`, and leaving it byte-identical keeps the claim "these numbers come from the frozen code" simple. |
| 19 | Are `configs/` needed, and do the scripts read them | **Yes, with `--config` support** (user's decision). Each file documents what one reported run used, and the scripts and both drivers read it, so there is one source of truth. |
| 20 | `paper/` | Skipped for now (no LaTeX source yet, decision 4). |

## 2. What was added

| Item | Where |
|---|---|
| `--config FILE.yaml` loader: flat YAML mapping of option names to values; precedence command line > config file > built-in default; unknown keys, values outside an option's choices and values of the wrong type are rejected; relative `data_path` / `run_dir` / `cache_path` resolve against the project root; `cache_path: none` keeps meaning "no cache". New module, no existing code touched. | `src/swarm_planner/run_config.py` |
| `--config` wired into all five run scripts (`parse_args(argv=None)` now takes an optional argument list; existing flags and defaults unchanged) | `scripts/run_experiment.py`, `run_baseline.py`, `run_sampling_control.py`, `run_verifier_solver.py` |
| One config per reported row. They name the settings the reported runs used and leave out `seed` and `run_dir` (given on the command line). | `configs/baseline.yaml`, `tot_only.yaml`, `tot_aco.yaml`, `sampling_control.yaml`, `verifier_solver.yaml` |
| PowerShell driver reads the configs. Only parameters passed explicitly (`-Seeds`, `-TaskConcurrency`, `-MaxRequests`, `-TotCap`, `-SamplingCap`, `-BaselineCap`) override them. New `-DryRun` prints the commands. The cap and concurrency parameters no longer have script defaults: the config's value applies unless one is passed. | `scripts/run_experiments.ps1` |
| Bash twin of the driver (same steps; overrides through environment variables `SEEDS`, `TASK_CONCURRENCY`, `MAX_REQUESTS`, `TOT_CAP`, `SAMPLING_CAP`, `BASELINE_CAP`; `DRY_RUN=1`) | `scripts/run_all.sh` |
| SHA-256 manifest in `sha256sum` format of everything that goes into the release archive, and a `--verify` mode | `scripts/make_release_manifest.py` (writes `results/MANIFEST.sha256`) |
| `pyyaml>=6.0` added to the dependencies (it was already installed and pinned in `requirements.lock`) | `pyproject.toml` |
| 18 new tests (145 → 163) | `tests/test_run_config.py`, `tests/test_release_manifest.py` |
| `results/cache_import/` added to `.gitignore` | `.gitignore` |

Differences from the Section 4 layout of the earlier plan, on purpose: `tot.yaml` is called `tot_only.yaml` (matches the run folder `results/runs/tot_only`); `tot_aco_v2.yaml` does not exist (the v1 rule is no longer run); `run_all.ps1` is called `run_experiments.ps1` and has the bash twin `run_all.sh`; `aco/heuristics.py` does not exist (η was removed, the Phase 3 ablation code is `legacy_eta.py`).

## 3. Using the configs

```powershell
# one run
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\tot_aco.yaml --seed 0
# a flag overrides the file, the file overrides the defaults
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\tot_aco.yaml --seed 0 --max_cost_usd 15
# the drivers
.\scripts\run_experiments.ps1 -Step tot_aco -Seeds 0,1,2 -DryRun     # print the commands only
.\scripts\run_experiments.ps1 -Step tot_aco
SEEDS="0 1 2" TOT_CAP=15 scripts/run_all.sh tot_aco                  # bash
```
The run still writes `config.resolved.json` into its run folder, which stays the record of what actually ran (the config file path itself is not recorded; the resolved values are).

## 4. Provenance after this phase

- `git diff experiments-freeze -- src` lists only the two analysis modules from Phase 5 (`analysis_data.py`, `analysis_metrics.py`) and the new `run_config.py`. **No existing solver, verifier, ACO, ToT, LLM or logging module changed.** Scripts changed only in argument parsing (`--config`) and in the driver (reads the configs); `evaluate.py` and `evaluate_runs.py` are untouched.
- Checked at $0 on 2026-11-18: ToT+ACO from `configs\tot_aco.yaml`, seed 0, 3 tasks, `--max_cost_usd 0` (a cache miss would stop the run): 864 cache hits, 0 paid calls, `config.resolved.json` identical to the stored run (apart from limit, task count, paths and cap), predictions identical. A test also checks that every shipped config matches `results/runs/<name>/seed_0/config.resolved.json` for the shared settings.
- The paper's statement is therefore: *the experiments ran at `experiments-freeze` (baseline) and `786b3b9` (ToT runs, sampling control at `ec5e5d4`); the run code is unchanged since; later commits add the analysis code, `--config` support and release tools.*
- **To do by the user:** commit and tag, for example `git tag config-support` in the repository root (name by content, not by phase). I did not commit or tag anything.

## 5. Script index

| Script | Purpose | Cost |
|---|---|---|
| `run_experiment.py` | ToT-only (`--aco false`) and ToT+ACO (`--aco true`) | LLM |
| `run_baseline.py`, `run_sampling_control.py` | baseline and equal-budget sampling control | LLM |
| `run_verifier_solver.py` | verifier-as-solver reference row | free |
| `run_experiments.ps1`, `run_all.sh` | drivers for all of the above plus evaluation | per step |
| `derive_predictions.py`, `evaluate_runs.py`, `evaluate.py` | derive every selection rule from the logs; aggregate over seeds; official Natural Plan evaluator (Apache-2.0 header kept) | free |
| `make_tables.py`, `make_figures.py`, `investigate_aco_effect.py` | analysis tables, figures, best-of-all investigation (PHASE5.md) | free |
| `build_datasets.py`, `audit_verifier.py` | rebuild `data/`, audit the verifier against all 1000 gold answers | free |
| `enumerate_paths.py`, `simulate_aco.py`, `analyze_path_table.py`, `analyze_lookahead.py`, `validate_simulator.py` | the path-table analysis and simulation (PHASE3.md) | `enumerate_paths.py` uses the LLM, the rest is free |
| `make_release_manifest.py` | SHA-256 manifest for the release archive | free |

## 6. Still open

1. Commit and tag (Section 4).
2. `paper/`: waits for the LaTeX source (decision 4).
3. Release archive: logs, results, merged cache (about 430 MB) and `results/MANIFEST.sha256` as a GitHub Release or Zenodo archive. Build the manifest right before archiving (`scripts/make_release_manifest.py`, it hashes the 430 MB cache), and put the archive link in the README (Phase 7).
4. You may delete `results/cache_import/` (411 MB) and `results/cache/llm_cache.backup.sqlite` (the pre-merge cache) once you are happy with the merge. The manifest skips the backup.
5. README and the cache-replay demo: Phase 7 (not started).
