# Phase 4 — Main experiments

**Status:** CODE DONE, LIVE RUNS NOT STARTED. All decisions D1–D8 are made (Section 3). The code, the tests (140 pass), the run scripts and the evaluation pipeline are implemented and were checked offline, including a full-budget ToT run on 20 cached dev tasks at $0 (Section 4b). **The commands, the output locations and what to expect are in [RUN_DETAILS.md](RUN_DETAILS.md).** The user runs the live steps; afterwards Phase 5 reads the logs.

---

## 1. Resume checklist for a new session

1. Read, in this order: the root `CLAUDE.md`, then `PLAN.md`, `PHASE0.md`, `PHASE1.md`, `PHASE2.md`, `PHASE3.md`, this file and [RUN_DETAILS.md](RUN_DETAILS.md).
2. Work from `swarm-multi-agent-planner/swarm-multi-agent-planner-v2/`. The interpreter is `.venv/Scripts/python.exe`.
3. Check the state: `pytest` should report **140 passed**. `.env` (git-ignored) holds `OPENROUTER_API_KEY`. The shared LLM cache is `results/cache/llm_cache.sqlite`.
4. The user runs the large live LLM steps manually and reports back. Do not start live runs without an estimate and an explicit go-ahead (budget constraint).
5. If the runs are finished: run `scripts/run_experiments.ps1 -Step evaluate`, read `results/aggregate/main/report.txt`, write the Phase 4 results into this file and PLAN.md, then start Phase 5.

---

## 2. What is already settled

| Item | State |
|---|---|
| Model and provider | `google/gemma-3-4b-it` on OpenRouter, pinned to DeepInfra (no fallback), temperature 0, top_p 1, max_tokens 400. |
| Test set | `data/curated_400.json` (40 tasks per category). Dev set `data/dev_100.json`, of which 20 tasks were analysed. |
| Quality signal | The fixed verifier of Phase 1. Gold is the top-scoring slot on all 1000 tasks. 66 tasks have several equally good answers (F11), so no verifier can reach 100%. Verifier-as-solver: 92.75% on the test set. |
| Infrastructure | LLM gateway with SQLite cache and in-flight dedupe, event log (`events.jsonl.gz`), cost and token tracking, run directories, task concurrency (Phase 2). |
| v2 ACO method | No heuristic η. p_ij = (τ_ij + ε)^α / Σ_k (τ_ik + ε)^α, α = 1, ε = 0.001, τ₀ = 1. Update τ ← (1 − ρ)τ + ΣQ with ρ = 0.1. 8 agents × 6 iterations, **full budget, no early stopping** in the main runs. Deposit: top-only (D1). Final selection: headline `verified_last3` (D2). |
| Removed | The v1-rule row (η, score deposit) is no longer run. Early stopping is an offline analysis only. The 3-level ToT ablation, the reward model and the feasibility checker are dropped. |
| Why no early stopping | With early stopping the pheromone would only be updated after a verified answer, which ends the search, so ACO could never act (finding F15). |

Phase 4 table (from PLAN.md): single-SLM baseline, ToT-only, ToT+ACO (frozen v2), equal-budget sampling control (48 samples at temperature 0.7, best picked by the verifier), and the verifier-as-solver reference row. Seeds 0, 1, 2. Metric: the official Natural Plan exact match, plus statistics (mean ± SD over seeds, 95% bootstrap interval over tasks, paired McNemar tests, per-category tables).

---

## 3. Decisions (all made; the reasoning is kept below)

**Outcome (the user accepted the recommendations on 2026-11-07):**

| # | Decision | Outcome |
|---|---|---|
| D1 | Deposit rule | **Top-only**: a trace deposits only if the verifier scores its answer 1.0. |
| D2 | Final selection rule | **Headline `verified_last3`**: verified traces of the last 3 iterations, **earliest verified slot**; no verified trace there → best trace of the whole run. Window fixed before the test runs. Reported next to it: `verified_last2`, the true frequency vote (`verified_vote_last3/2`, ties to the earliest slot), `best_of_all`, `best_of_last3/2`, `last_iteration` (v1 rule), `majority_vote`, `early_stop`. With the earliest-slot tiebreak the headline equals best-of-last-k restricted to verified traces; the paper says so instead of calling it a vote (D2a/D2b). |
| D3 | Budget | About $30–35 accepted, with a per-run cost cap (`--max_cost_usd`). |
| D4 | Baseline seeds | The seed is the cache nonce (each seed is a real call). |
| D5 | Acceptance criterion | Both ToT modes above the baseline under the headline rule, mean over seeds. |
| D6 | Sampling control | Best of 48 by the verifier, compared with `best_of_all` (and the headline), stated openly. |
| D7 | Valid-answer rate | Reported as a column (`verified_answer_rate`). |
| D8 | Git commit and tag | Yes, before the first test-set run (see RUN_DETAILS.md Section 6). |

The original option descriptions follow for the record.

### D1. Top-only deposit (decided: yes)

A path leaves pheromone (Q = 1 on each of its edges) **only if the verifier scores its answer 1.0**. Everything else deposits nothing.

| Choice | Consequence |
|---|---|
| **Yes (my recommendation)** | The pheromone learns only from perfect answers. Simplest rule, and it fixes F7 (v1 let wrong answers deposit). It starts moving after the first perfect answer of a task. In simulation the probability on the best level-1 branch goes from 0.333 to 0.661. |
| No, **valid-only** (hard-valid answers deposit their score) | An earlier signal, because imperfect valid answers also deposit. Accuracy within noise of top-only under last-iteration selection (45.6 vs 46.6), slightly lower under best-of-all (−1.7 [−5.3, +0.2]). |
| No, the v1 score deposit | The pheromone does not learn (0.292 → 0.295) and best-of-all drops (−2.4). Not recommended. |
| No, best-of-iteration or score-minus-mean | Similar to valid-only, not better. |

### D2. Final prediction selection rule (decided: headline `verified_last3`, earliest-slot tiebreak)

The search produces 48 traces. The rule picks the one answer that is submitted. It is separate from the quality score: the score drives the pheromone during the search, the rule only chooses the final answer. The logs keep every trace, so any rule can be derived offline and several can be reported side by side. The open question is which one is the **headline**, and whether the same rule is applied to both modes (it should be).

**What a vote is.** A *trace* is one agent's full path, which ends on one proposed slot. A vote counts **traces per answer**, not distinct answers: if 12 of the traces ended on "Monday 14:00 – 15:00" and 9 on "Monday 16:00 – 16:30", the first has 12 votes and wins. There are two kinds of vote, which must not be confused:
- **Plain majority vote:** all 48 traces vote, and the verifier is not involved in the choice.
- **Verified vote:** only the traces the verifier scores 1.0 vote, optionally only those of the last 2 or 3 iterations. The verifier filters, the vote chooses. The pheromone values themselves are not read.

Simulated on the 20-task dev sample (full 8 × 6 budget, top-only deposit, 400 searches per task, intervals over tasks):

| Rule | ToT-only | ToT+ACO | ACO minus ToT-only | Consequence |
|---|---|---|---|---|
| **Best of all iterations** | 48.6% | 48.6% | +0.0 | Fair and standard in ACO (best-so-far). Highest accuracy for both. ACO shows **no effect on this number**, because the best answer is secured at the first success, possibly before any pheromone exists. Both modes are well above the baseline. |
| **Best of the last iteration** | 38.2% | 47.7% | **+9.4 [+3.5, +15.8]** | The answer must come from the pheromone-guided iteration, so ACO's effect shows. It is what v1 did. But ToT-only is handicapped (it discards 40 of 48 traversals), and ACO only recovers what best-of-all gives ToT-only for free. Both above the baseline. |
| **Best of the last 2 iterations** | 44.5% | 48.1% | +3.7 [+0.7, +7.2] | A compromise. ACO's advantage shrinks as more iterations count. |
| **Best of the last 3 iterations** | 46.6% | 48.3% | +1.7 [+0.1, +3.7] | Closer to best-of-all, a small ACO effect. |
| **Verified vote**, all iterations | 48.6% | 48.6% | +0.0 | Same accuracy as best-of-all on this sample. |
| **Verified vote**, last 3 iterations | 46.6% | 48.3% | **+1.7 [+0.1, +3.7]** | **The user's current preferred approach.** Both modes far above the baseline, ACO slightly ahead, and the answer comes from the later, pheromone-guided iterations. See D2a–D2c for its caveats. |
| **Verified vote**, last 2 iterations | 44.5% | 48.1% | +3.7 [+0.7, +7.2] | Same idea with a bigger ACO gap and a smaller window. |
| Hard-valid vote (every hard-valid trace votes), last 3 iterations | 40.8% | 47.6% | +6.9 [+0.3, +16.6] | Weaker for ToT-only, because valid-but-imperfect answers compete in the vote. Not recommended: it exaggerates the ACO gap by handicapping ToT-only. |
| **Plain majority vote** over the 48 traces | 13.3% | 36.3% | **+23.0 [+9.0, +38.0]** | Needs no verifier at selection time (the verifier only updates the pheromone), and ACO's effect is largest. But ToT-only falls **below the single-agent baseline**, which breaks the requirement that both modes beat it, and ACO falls below the 0.40 of v1. A secondary row only. |

For orientation, the v1 test-set numbers were: baseline 24.75%, ToT-only 39.5%, ToT+ACO 40.0% (last-iteration selection).

Considerations:
- ACO can only affect the final answer in proportion to how much of the answer comes from after the pheromone has built up. A rule that includes iteration 1 dilutes or removes ACO's effect on that number.
- Whatever the headline, report the others next to it, apply the **same rule to both modes**, and state openly which one is the headline.
- My suggestion: report best-of-all and best-of-last side by side (plus last-3 and majority vote), and name the headline explicitly. My lean for the headline is best-of-all (fair, highest accuracy, requirement met). If you want the answer to come from the deposited branches, best-of-last-2 or last-3 is the compromise, and best-of-last is the most ACO-visible but the most handicapping for ToT-only.
- The dev numbers come from 20 tasks (10 solvable), so the intervals are wide.

**Current preference of the user: the verified vote** (window to be fixed, the last 2 or 3 iterations). It meets the requirement that both modes beat the baseline, it gives ACO a small visible edge, and the answer comes from the pheromone-guided iterations. It is **not yet recorded as final**, because of the points below.

**D2a. It is nearly the same as "best of the last k iterations".** On the dev sample the verified vote and best-of-last-k give identical accuracy (46.6 / 48.3 for k = 3), because almost every verified answer is correct. They differ only on tasks that have **several different verified answers** (F11): best-of-last-k takes the earliest slot, the vote takes the most frequent one.

**D2b. The tie-handling can cost accuracy on the test set.** 70 of the 400 test tasks (17.5%) have several verified answers. The earliest-slot rule matches gold on 41 of them (58.6%), while an arbitrary pick among the verified answers would match on only about 41%. If the vote picks by frequency alone, accuracy could fall by up to about **3 points overall** compared with the earliest-slot rule. The votes carry no information about which valid answer is the gold one. Options: (i) vote by frequency, accepting that risk; (ii) require the vote to be broken by the earliest slot only for exact ties, which does not remove the risk; (iii) use the verified vote to decide *whether* a verified answer exists but pick among verified answers by earliest slot, which is then the same as best-of-last-k. This must be decided before the runs.

**D2c. Smaller choices to fix in advance:**
- **The window:** the last 2 or the last 3 iterations. I would fix it **before** the test runs. The dev numbers were looked at when this idea came up, so a window picked now is slightly tuned to dev, and the test-set effect may be smaller.
- **No verified trace in the window:** fall back to the best-scoring trace of the whole run (the simulator falls back to a plain majority of the window).
- **Same rule for both modes**, and the window is counted in iterations, not traces.

**Also noted, not tested:** choosing the answer with the **maximum pheromone** along its path is a different rule from a vote. It behaves like a recency-weighted vote at the branch level, and could be compared offline on the dev table if wanted.

The simulator reports all these rules (`scripts/simulate_aco.py` Section 6, `results/path_table/dev_20/analysis/simulation_report.txt`).

### D3. Phase 4 budget (decided: accept, with per-run caps)

| Item | Estimated cost |
|---|---|
| One 400-task ToT run (full budget) | about $9 |
| Six ToT-based runs (ToT-only and ToT+ACO, 3 seeds each) sharing the cache | about **$25** |
| Sampling control (48 samples per task, 3 seeds, about 58k single calls) | about $5–7 |
| Single-call baseline | about $0.1–0.3 |
| **Total** | **about $30–35** |

Estimated from the dev table's real per-request costs (`scripts/simulate_aco.py`, Section 7), extrapolated from 20 tasks. Options to cut it: a smaller test subset (for example 200 tasks, about half), fewer seeds for the sampling control, or running ToT-only and ToT+ACO on fewer seeds (the professor asked for at least 3). Decide before any run. Adding a per-run cost cap to the run script is advisable.

### D4. Baseline seed handling (decided: seed as nonce)

With the cache, a baseline repeated over three seeds returns identical answers (SD exactly 0), since temperature 0 and identical requests hit the same entry. Options: use the seed as the cache nonce, so each seed is a real provider call and the SD reflects provider noise (about $0.1–0.3), or accept identical replays and say so. I recommend the nonce.

### D5. Acceptance criterion against the baseline (decided)

"Both modes score higher than the single-agent baseline" is an empirical outcome, not something the design can guarantee. Decide the criterion (for example the headline selection rule, per mode, mean over seeds). I can also run the baseline on the 20 dev tasks for about a cent to see the margin before the test runs. Note that ToT-only paths alone are less accurate (about 13% per path on dev) than the baseline, so the gain comes from selecting among many traces with the verifier.

### D6. Sampling control selection rule (decided)

The control picks the best of 48 samples by the verifier, which corresponds to best-of-all. If D2 chooses another headline, decide how the control is compared (same rule is not possible for independent samples), and state it.

### D7. Valid-answer rate as a secondary column (decided: yes)

Exact match rejects valid answers on the 166 of 1000 tasks that have several equally good answers (F11). Reporting the share of answers with verifier score 1.0 next to exact match shows how much of any miss is due to that ambiguity. I recommend including it.

### D8. Git commit and tag (decided: yes)

v2 is untracked. The plan says to freeze the verifier and the update rule before any test-set run. I suggest committing and tagging a version right before the first test-set run, so the code behind every number is identifiable (the run log already records the git commit).

### Not decisions, but to keep in mind
- **Parallel processes on one cache file** and **rate limits at 400 tasks** are untested. Run one configuration at a time and start with a smoke test.
- **Reading the Natural Plan paper** on how gold answers are generated, and whether the **LaTeX source** of our paper exists (decision 4 in PLAN.md), are needed at the paper stage (Phase 8), not for Phase 4.

---

## 3b. Justifying the selection strategy to the professor

This is for the paper (Phase 8) and for the reassessment meeting. It was written for the verified vote; the headline is now the verified rule with the earliest-slot tiebreak (D2), so say in the paper that it is best-of-last-3 restricted to verified traces, and report the true frequency vote as a comparison row. Most of it holds for any rule that includes later iterations. The professor's feedback asks for evidence that ACO influences the search (Req. 4) and for conclusions that follow the evidence, with an explicit discussion if ACO ≈ ToT-only (Req. 5). The argument should be evidence-based and **fixed in advance**, not result-driven.

**The argument, in order**
1. **Start from a v1 weakness we found (F8).** v1 selected only the best trace of the last iteration, so 40 of the 48 traversals were discarded. That under-uses the search and handicaps ToT-only: selecting over all iterations raises ToT-only by +9.8 points [+3.5, +16.7] in the dev simulation, at no cost.
2. **Explain why best-of-all alone is not enough (F15).** With an exact verifier the best answer is secured at the first success, possibly before any pheromone exists, so ACO cannot change a best-of-all result. We show this structurally and in simulation (identical results). This is the honest discussion of ACO ≈ ToT-only that Req. 5 asks for.
3. **Present the verified vote as principled, not tuned.** The verifier filters the candidates, the swarm votes (a verifier-filtered form of self-consistency voting, Wang et al., ICLR 2023, already planned as a citation), and only the later iterations count, because that is where the pheromone-guided colony has converged. The rule rests on a design reason, not on which rule makes ACO look best.
4. **Be transparent about how it was chosen.**
   - Fix the rule and the window **before** the test runs, and commit and tag the code (D8).
   - Report all the rules side by side: best of the last iteration (comparable with v1), best of all, best of the last 3 and 2, the verified vote, and the plain majority vote.
   - Show the sensitivity to the window (2, 3, all iterations) with intervals.
   - State that the rule was chosen from the 20-task dev analysis and never tuned on the test set.
5. **Claim only what the data supports.** ACO equals ToT-only under best-of-all. ACO concentrates the swarm on verified answers (the verified share per iteration rises from 13% to 32% in the dev simulation), which helps every rule that takes the answer from late iterations. Do not claim that ACO beats ToT-only in general.
6. **Explain the tie-handling** (D2b). Some tasks have several valid answers (F11). Say which rule breaks ties and why (earliest slot matches gold more often than an arbitrary pick: 59% against about 41% on the test set).

**The professor's likely questions**
- *"Why not just best-of-all?"* We report it too. It shows that ToT-only is already as good under that rule, which is a finding, not a failure.
- *"Did you choose the window to favour ACO?"* The window was fixed before the test runs, sensitivity is shown, and the effect is small, which the paper says openly.
- *"Does the verifier decide the answer?"* Yes: the system is verifier-guided generation, and the verifier-as-solver row (92.75% on the test set) is reported as a reference. The plain majority vote row shows what happens without it.

**Will both modes stay above the single-agent baseline?** Very likely, but it is an expectation, not a guarantee (D5).
- **Dev simulation:** the verified vote over the last 3 iterations gives about 47% (ToT-only) and 48% (ToT+ACO), against a baseline of about 25%.
- **A floor from v1:** the verified vote uses a superset of the candidates of v1's rule (best of the last iteration), and v1 measured 39.5% and 40.0% on the test set. In the worst case of the tie-handling risk (up to about 3 points, D2b) both modes still sit around 36%, above 24.75%.
- **Only the plain majority vote breaks it,** because ToT-only falls to about 13%. It is therefore a secondary row only.
- **Caveats:** 20 dev tasks, and the verifier was fixed after v1. Phase 4 measures the baseline and both modes on the same test set and states the result as it comes out.

---

## 4. What was implemented (no LLM cost)

| Item | Where |
|---|---|
| η removed from the live search (no β, no task features in the graph). The old code is kept **only** for the Phase 3 offline ablation. | `aco/selection.py`, `tot/graph.py`, `solver.py`; η in `legacy_eta.py` (used by `simulate.py` only) |
| Top-only deposit wired into the pheromone update | `aco/pheromone.py` (`trace_deposit`), `verifier/scorer.py` (`deposit_quality(..., "top_only")`) |
| Selection rules, shared by the solver (headline) and the offline derivation (all rules) | `selection_rules.py` |
| The solver applies the headline rule after the last iteration, never stops early, logs every trace (plus `parsed` in the verifier record) | `solver.py` |
| Shared run directory, progress line (elapsed, ETA, money spent), run config | `run_io.py` |
| Hard per-run cost cap: no provider call once the run has paid the cap, cache hits are free, a rerun resumes | `llm/gateway.py` (`CostCapExceeded`, `--max_cost_usd`) |
| Single-call baseline, sampling control (48 samples, temperature 0.7, nonce per sample), verifier-as-solver | `baseline.py`, `scripts/run_baseline.py`, `run_sampling_control.py`, `run_verifier_solver.py` |
| All selection rules derived from the logs, with per-task tables, oracle pass@k, per-iteration verified share, early-stop budget | `derive.py`, `scripts/derive_predictions.py` |
| Official exact match in memory (checked equal to `scripts/evaluate.py`), SD over seeds, bootstrap CI over tasks, exact McNemar | `metrics.py`, `stats.py` |
| Aggregation: main table, per-category table, paired tests (ACO vs ToT-only for every rule; both vs baseline; both vs the sampling control) | `scripts/evaluate_runs.py` |
| One-command driver with caps | `scripts/run_experiments.ps1` |
| 41 new or rewritten tests (99 → 140) | `tests/test_aco.py`, `test_solver.py`, `test_selection_rules.py`, `test_metrics_and_stats.py`, `test_experiment_runs.py` |

`train` and the v1-rule row are gone; `scripts/run_experiment.py` no longer has `--beta` (and gained `--selection_policy`, `--max_cost_usd`). The Phase 3 simulator still accepts `beta` because it reproduces the η ablation.

### 4b. Checks done offline

- **Live headline equals derived headline** on every run (fake LLM in tests; 6 real runs below).
- **Real full-budget run at $0:** ToT-only and ToT+ACO, seeds 0–2, on the 20 dev tasks of the Phase 3 path table (every request is in the cache; run with `--max_cost_usd 0`, which guarantees no spend). 5,760 calls per run, all cache hits, 7 s per run. The derived numbers agree with the Phase 3 simulation: last-iteration selection ToT-only 40% (sim 38%), ACO 48% (sim 48%); plain majority vote 12% / 37% (sim 13% / 36%); headline 50% for both (the ceiling). The official `scripts/evaluate.py` gives the same numbers as `evaluate_runs.py`.
- **Verifier-as-solver on the real test set:** 92.75% (371/400), identical to Phase 1. It is already stored in `results/runs/verifier_solver/`.

## 4c. Still to do

1. **Smoke test live** (about $0.02): `scripts/run_experiments.ps1 -Step smoke`.
2. **The live runs** (about $30–35): see [RUN_DETAILS.md](RUN_DETAILS.md).
3. **Evaluate**: `scripts/run_experiments.ps1 -Step evaluate`. Write the results here and in PLAN.md.
4. Hand over to Phase 5 (figures and tables from the logs).

---

## 5. Useful facts and where things are

**Dev-sample findings to keep in mind** (details in PHASE3.md, 20 tasks, wide intervals):
- 10 of 20 tasks have no correct path among the 243. Generation, not search or selection, is the limit.
- ToT-only with the full budget and best-of-all reaches 48.6%, near the ceiling of 50%.
- Only level 1 (and weakly level 2) change correctness. L3–L5 are interchangeable.
- η gave no benefit, so it was dropped.
- With the full budget, ACO raises the verified share of traces from 13% to 32% by iteration 6 and moves the pheromone (best level-1 branch 0.333 → 0.661). It does not change the best-of-all accuracy.

**Code and data:**

| Path | Content |
|---|---|
| `src/swarm_planner/` | `solver.py`, `tot/` (graph, prompts, levels), `aco/`, `verifier/`, `llm/` (gateway, cache), `logging.py`, `run_io.py`, `selection_rules.py`, `baseline.py`, `derive.py`, `metrics.py`, `stats.py`, `path_table.py`, `simulate.py`, `legacy_eta.py` (Phase 3 only). |
| `scripts/run_experiment.py` | ToT-only / ToT+ACO on a dataset (flags: `--aco`, `--seed`, `--selection_policy`, `--max_cost_usd`, `--run_dir`, `--cache_path`, `--limit`, `--task_concurrency`, `--max_concurrent_requests`). |
| `scripts/run_baseline.py`, `run_sampling_control.py`, `run_verifier_solver.py`, `run_experiments.ps1` | The other Phase 4 rows and the driver ([RUN_DETAILS.md](RUN_DETAILS.md)). |
| `scripts/derive_predictions.py`, `evaluate_runs.py` | Every selection rule from the logs; aggregation over seeds. |
| `scripts/enumerate_paths.py` | Builds a path table (used for the dev set). |
| `scripts/simulate_aco.py`, `analyze_path_table.py`, `analyze_lookahead.py`, `validate_simulator.py`, `audit_verifier.py` | Offline analyses (no LLM cost). |
| `results/path_table/dev_20/` | The 20-task path table, its events and the analysis reports. |
| `results/verifier_audit/` | The verifier audit against all gold answers. |
| `results/cache/llm_cache.sqlite` | The shared LLM cache (git-ignored). |
| `data/` | `curated_400.json`, `dev_100.json`, `verifier_known_exceptions.json`. |

**Useful commands** (from the v2 root):

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\simulate_aco.py --reps 400 --out results\path_table\dev_20\analysis\simulation_report.txt
.venv\Scripts\python.exe scripts\run_experiment.py --data_path data\dev_100.json --limit 2 --run_dir results\runs\demo\seed_0 --seed 0 --aco true
```

**Known limitations to state in the paper:** the fixed verifier is a near-perfect signal and effectively a solver (F4); exact match undercounts valid answers on tasks with several valid answers (F11); the cache freezes one LLM answer per request, so seeds vary the branch choices but not the model's own noise; the dev conclusions rest on 20 tasks.
