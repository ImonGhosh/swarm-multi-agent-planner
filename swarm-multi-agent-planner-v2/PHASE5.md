# Phase 5 — Analysis and figures showing ACO's effect

**Status:** DONE for all four runs (baseline, ToT-only, ToT+ACO and, run afterwards on 2026-11-15, the equal-budget sampling control; Section 2.5). Everything is derived from the shipped logs, at no LLM cost. The one open question (Section 3, the unpaired random draws) was decided: the current runs stay.

## 1. How to regenerate

From the v2 root (matplotlib is needed for figures only: `uv pip install -e ".[analysis]"`):

```powershell
.venv\Scripts\python.exe scripts\evaluate_runs.py                # main table (Phase 4)
.venv\Scripts\python.exe scripts\make_tables.py                  # CSV tables (about 35 s)
.venv\Scripts\python.exe scripts\make_figures.py                 # PNG + PDF figures
.venv\Scripts\python.exe scripts\investigate_aco_effect.py       # the best-of-all investigation (Section 3)
```

| Output | Content |
|---|---|
| `results/aggregate/analysis/tables/per_iteration.csv` | per run and iteration: verified-trace share (all tasks, and tasks that already had a success), best and mean score, hard-valid and exact share, cumulative tasks with a verified / exact trace, cumulative distinct paths |
| `.../verified_share_difference.csv` | ACO minus ToT-only verified share per iteration, 95% bootstrap CI over tasks |
| `.../first_success.csv` | per task, run: iteration of first verified / exact trace, distinct paths |
| `.../pheromone.csv` | mean colony probability per seed, iteration, level, branch (all tasks, and tasks after a deposit), normalised entropy |
| `.../pheromone_vs_quality.csv` | pheromone against the true branch quality (Section 2.3) |
| `.../cost.csv` | calls, distinct requests, tokens, USD, latency per run (baseline, ToT-only, ToT+ACO, sampling control) |
| `.../candidate_diversity.csv` | per run: verified share per candidate, share of tasks with a verified candidate among the 48, distinct answers per task (Section 2.5) |
| `results/aggregate/analysis/figures/` | `verified_share_per_iteration`, `scores_per_iteration`, `first_success_and_paths`, `pheromone_heatmap_and_entropy`, `pheromone_vs_branch_quality`, `accuracy_by_rule_and_category` (with the sampling control), `sampling_control_comparison` (.png and .pdf) |
| `results/aggregate/analysis/best_of_all_investigation.txt` | the investigation of Section 3 |

Code: `src/swarm_planner/analysis_data.py` (log loader), `analysis_metrics.py` (measures), `tests/test_analysis_metrics.py` (145 tests pass in total).

## 2. Findings (400 test tasks, 3 seeds)

### 2.1 Accuracy (from Phase 4, `results/aggregate/main/report.txt`)
Baseline 21.75%, ToT-only 57.4%, ToT+ACO 59.3% under the headline rule `verified_last3` (SD over seeds 0.4, 0.3, 1.3 points). ACO minus ToT-only: +1.9 points, 95% CI [+0.3, +3.6]; per seed +3.0, +1.0, +1.75 and only seed 0 is significant (McNemar p = 0.043). Both modes beat the baseline by 36–38 points (all p < 0.0001). Rules that read late iterations favour ACO strongly (last 2 iterations +10 points, last iteration +19, majority vote +25) because ToT-only is handicapped by them; the paper should say that.

### 2.2 ACO changes the search (Req. 4, the main evidence)
- **Share of verified traces per iteration** (`verified_share_per_iteration.png`): ToT-only stays flat at 12.5% → 13.6%; ToT+ACO rises 12.6% → 39.3%. Difference at iteration 6: +25.7 points, CI [+23.4, +28.0]; iteration 1 is +0.04 [−0.7, +0.8] (nothing can differ yet). Among tasks that already had a verified trace: ToT-only 22%, ToT+ACO 62% by iteration 6.
- **Score per iteration** (`scores_per_iteration.png`): ACO's best and mean trace score rise (mean 0.476 → 0.637), ToT-only's stay flat (0.475 → 0.481). The share of exact-match traces rises 11% → 36% for ACO and stays at 11–12% for ToT-only.
- **Pheromone moves away from uniform** (`pheromone_heatmap_and_entropy.png`): normalised entropy after the last update is 0.53 (L1), 0.54 (L2), 0.61 (L3), 0.75 (L4), 0.78 (L5). The heatmap of the *task-averaged* probability stays near 0.33 because the best branch differs between tasks; entropy and Section 2.3 show the concentration.
- **Cost**: ACO revisits paths, so it needs fewer distinct requests (about 55.8k against 63.3k per run) and explores fewer distinct paths per task (38.6 against 43.6 of 243). With the cache shared, a cold run costs about $9.0 (ToT-only) and $8.0 (ToT+ACO); all 115,200 calls without any caching would be about $16.1; the baseline costs $0.046 per seed. Measured total of Phase 4: the six ToT runs together paid for 176,394 distinct requests, worth $25.38 at the logged per-call costs; plus the sampling control $6.61 and the baseline $0.14, about **$32.1** in all (inside the $30–35 estimate). (`cost.csv`; the rest of the numbers are tokens, about 305 M in and 8.1 M out per run.)

### 2.3 Does the pheromone point at the branches that really work? (the key figure)
Reference: per task, level and branch, the verified rate in the **independent** ToT-only runs (uniform sampling, so unbiased and not influenced by ACO). Informative tasks only (a unique best and worst branch with a gap of at least 5 points) and tasks where ACO deposited at least once, mean over seeds (`pheromone_vs_branch_quality.png`):

| Level | Final probability on the best branch (chance 0.333) | Most probable branch is the best / the worst (chance 33% each) |
|---|---|---|
| L1 | 0.588 | 65% / 5% |
| L2 | 0.546 | 61% / 8% |
| L3 | 0.492 | 54% / 15% |
| L4 | 0.392 | 44% / 23% |
| L5 | 0.369 | 38% / 29% |

So ACO learns where the branch choice matters (L1, L2) and learns little where it does not (L4, L5). That is the same ranking the Phase 3 path table found (only L1 and weakly L2 change correctness), obtained here on different tasks and a different method. The dev simulation expected the L1 probability on the best branch to rise from 0.333 to about 0.66 (a different subset of tasks); the observed 0.59 is in line.

### 2.4 First success and exploration (`first_success_and_paths.png`)
Cumulative share of tasks with a verified trace after iteration 6: 62.0% (ToT-only) and 64.3% (ToT+ACO). See Section 3 for why that 2.3-point gap should not be attributed to ACO.

### 2.5 Equal-budget sampling control (what the ToT structure adds beyond "many samples plus the verifier")
The control asks the single-call baseline prompt 48 times per task at temperature 0.7 (a different cache nonce per sample, so 48 real provider calls), scores every answer with the verifier and returns the best (`best_of_samples`). It has the same number of calls and the same verifier as ToT, but no tree. 3 seeds, 400 tasks, 19,200 calls and $2.204 per seed ($6.61 in total, 0 cache hits, about 8.5 minutes per seed).

| Row | Exact match, mean (SD) | Seeds 0 / 1 / 2 | Tasks with a verified answer |
|---|---|---|---|
| Single call (baseline) | 21.75% (0.4) | 21.50 / 21.50 / 22.25 | 24.3% |
| **Sampling control, best of 48 by verifier** | **27.67% (0.9)**, CI [23.3, 31.9] | 26.75 / 28.50 / 27.75 | 30.7% |
| Sampling control, majority vote | 21.75% (0.3) | 21.50 / 21.75 / 22.00 | 24.5% |
| ToT-only, `verified_last3` | 57.42% (0.3) | 57.75 / 57.25 / 57.25 | 62.0% |
| ToT+ACO, `verified_last3` | 59.33% (1.3) | 60.75 / 58.25 / 59.00 | 64.3% |

- **Paired tests (McNemar exact per seed; pooled mean difference with bootstrap CI over tasks).** ToT-only minus control: +29.8 points [+24.8, +34.8] under best-of-all (+31.3 / +28.8 / +29.5 per seed) and +29.8 [+24.8, +34.8] under the headline rule. ToT+ACO minus control: +31.8 [+26.8, +36.8] under best-of-all and +31.7 [+26.6, +36.7] under the headline rule. Every per-seed p-value is below 0.0001.
- **Reading.** Sampling plus the verifier explains only about 6 of the 36-point gain over the single call (21.75% to 27.67%); majority vote adds nothing. The remaining gain over the control (about 30 points) comes from the 5-level reasoning tree and its 243 paths. So the ToT improvement is not an artefact of "48 attempts plus a verifier", and the paper's planned limitation (that the tree cannot be separated from sampling plus verifier) is no longer needed.
- **Why the control is so much weaker: it does not explore** (`candidate_diversity.csv`, `sampling_control_comparison.png`). A direct sample is verified *more* often than a ToT trace (23.7–24.2% of the 48 samples against 12.9% for ToT-only; ToT+ACO 25.9%, because the pheromone raises it), but the 48 samples of a task contain only 1.7 distinct answers on average, and in 51.9% of the tasks all 48 are the same answer. ToT-only has 8.0 distinct answers per task and ToT+ACO 7.5. Temperature 0.7 on one prompt keeps returning the same plan, so best-of-48 can rarely repair a wrong answer; the 5 levels x 3 branches force different intermediate reasoning and therefore different answers. Result: a verified answer exists in 30.7% of the tasks for the control and in 62.0% / 64.3% for ToT-only / ToT+ACO.
- **Caveat.** The control uses temperature 0.7 as decided (decision 15). A higher temperature or a diversity prompt might widen the 1.7 distinct answers and raise the control; this was not tested, so the comparison is "the sampling control as specified", not the best possible sampling baseline. The paper should say that.
- **Per category** (`per_category.csv`): the control is above the single call in 9 of 10 categories and below ToT in all 10 (for example 2p5d: 8.3% against 2.5% for the single call and 23.3% for ToT-only; 7p1d: 42.5% against 35.0% and 71.7%).

## 3. The +2 points under best-of-all: a measurement artefact, not an ACO effect (finding F16)

Phase 3 predicted that ToT-only and ToT+ACO tie under best-of-all selection (nothing is deposited before the first verified trace, so ACO cannot change whether a task ever finds one). Phase 4 measured +2.0 points [+0.4, +3.7]. `scripts/investigate_aco_effect.py` finds:

1. **The two modes never draw the same paths.** ToT-only picks a branch with `rng.integers(3)` and ToT+ACO with `rng.random()` (`tot/graph.py`), from the same per-call seed stream. Iteration-0 paths are identical in 0 of 400 tasks, in every seed. The two modes are therefore compared on the same tasks but on **independent** draws (the paired test pairs tasks, not draws).
2. **Before a task's first verified trace the pheromone is exactly uniform** (3,089 iterations checked; the largest deviation from pure evaporation is 1e-16), and the iteration-0 branch shares and the iteration-0 verified share are equal in both modes (12.5%). So until the first success both modes are the same process.
3. **The gap is already there after iteration 1 and does not grow:** +1.9, +1.1, +2.5, +2.7, +2.3, +2.3 points after iterations 1 to 6. If ACO caused it, it would grow once the pheromone acts.
4. A sign test on the tasks where only one mode found a verified trace gives 36 (ToT-only) against 64 (ACO), p = 0.007, but that test treats every task-seed as independent and was one of several looked-at statistics, so it overstates the evidence. Seed-to-seed differences inside one mode are as large (ToT+ACO seeds 0 vs 1: 21 vs 12).

**Conclusion for the paper:** under best-of-all (and early stopping) ToT-only and ToT+ACO are the same search by construction and any difference is sampling noise; the +2.0 points should be reported as such, with the iteration-1 gap as the evidence. This is consistent with F15.

**Decision (2026-11-15): keep the current runs, no common-random-numbers re-run; the paper reports the best-of-all gap as noise with the iteration-1 evidence.** The option, for the record: common random numbers. Making ToT-only use the same `rng.random()` sampler would couple the two modes exactly until ACO's first deposit, so any later difference is caused by ACO alone and the best-of-all comparison would tie by construction, with no noise. Because the ACO runs already hold those paths in the cache, re-running ToT-only with that sampler would mostly hit the cache; only the post-success iterations would need new calls (rough guess: a few dollars for 3 seeds). It changes frozen code (`tot/graph.py`) and gives a different ToT-only row, so it is the user's decision. Alternative: keep the current rows and report the measurement as above.

## 4. Provenance checks
- Run dates (UTC, from `run_start`): baseline 2026-11-07, ToT-only seed 0 on 2026-11-08 and seeds 1–2 on 2026-11-11, ToT+ACO 2026-11-11, sampling control 2026-11-15.
- Baseline ran at commit `4a88377` (the `experiments-freeze` tag), ToT-only and ToT+ACO at `786b3b9`; `git_dirty` is False in these nine runs. The sampling control ran at `ec5e5d4` with `git_dirty: True`. `git diff experiments-freeze ec5e5d4 -- src scripts tests configs pyproject.toml` is empty, so the run code is the frozen code. The only files that changed afterwards (commit `6669682`) are the Phase 5 analysis files (`analysis_data.py`, `analysis_metrics.py`, `make_tables.py`, `make_figures.py`, `investigate_aco_effect.py`, `test_analysis_metrics.py`) plus result files; those analysis files are the likely cause of the dirty flag, but the log does not record which files were dirty, so say "dirty flag, no change to run code" in the paper rather than "clean".
- All ToT runs share one cache (cache size grows monotonically from 63k to 176k entries); an identical path always got an identical answer in both modes (0 mismatches).
- **Two cache files, two machines (resolved 2026-11-18).** The ToT runs wrote to a cache on the other machine (`D:\...\results\cache\llm_cache.sqlite`, 176,394 entries, started empty, all DeepInfra, logged cost sum $25.38 = the cost we computed from the logs). The baseline and the sampling control wrote to the cache in this folder (74,221 entries: 15.4k from the Phase 3 dev table, plus 3 × 400 baseline and 3 × 19,200 sampling-control calls). The ToT logs only hold 16-character request hashes, so the ToT cache was copied to `results/cache_import/` and **merged into `results/cache/llm_cache.sqlite`** with `INSERT OR IGNORE` (no key overlap, 0 conflicts; 250,615 entries; the pre-merge file is kept as `results/cache/llm_cache.backup.sqlite`). **Replay check:** ToT-only and ToT+ACO, seed 0, re-run with `--max_cost_usd 0` into a scratch directory: 115,200 cache hits and 0 paid calls each, predictions identical to `results/runs/*/seed_0/predictions.json` for all 400 tasks, and for ToT-only all 2,400 iteration trace records are identical. Seeds 1 and 2 were not replayed (same cache, same code). Both cache directories are git-ignored.
- The baseline scores 21.75%, below v1's 24.75%. Different provider, run date and a frozen realisation of temperature 0 are the likely reasons; state it in the paper.
- Seeds 1 and 2 of the baseline differ only through the cache nonce (SD 0.43 points).

## 5. Still open
1. (Decided: keep the current runs, Section 3.)
2. (Done 2026-11-15: the sampling control was run, Section 2.5. The ToT structure, not "sampling plus verifier", carries the gain; state the temperature-0.7 caveat in the paper.)
3. Phase 6 onwards: repo layout, README, paper (Phase 8: use Sections 2.1–2.3 and 3 for the discussion, and report the late-iteration rules next to the headline).
4. (Done 2026-11-18: the ToT cache is merged into `results/cache/`, Section 4.) Still to do for Req. 2: put the merged cache (about 430 MB) into the release archive with a checksum (Phase 6/7).
