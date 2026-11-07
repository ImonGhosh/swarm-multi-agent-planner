# Phase 3 — Path-table analysis on the dev set

**Status:** done. The analysis is complete on a **20-task dev sample** (2 tasks per category, chosen because of the budget), and the **v2 method is frozen** (Section 5). The conclusions carry the wide uncertainty of a 20-task sample, and Phase 4 on the test set is the real check. One finding shaped the final design (Section 5.3): with early stopping ACO could not act at all, so the main runs use the full budget and early stopping is only derived offline.

**Goal:** decide, on the dev set only, whether the 5-level ToT has a bottleneck (our point 5), whether the heuristic η is reasonable (point 6) and whether the n × T design is sound (point 8), and freeze the v2 ACO update rule before any test-set run.

---

## 1. Decisions made

| Decision | Outcome |
|---|---|
| Decision 1, temperature | Stays 0, as in v1. |
| Decision 3, ToT redesign | No 3-level ablation. The 5-level tree stays, and Phase 3 only analyses it. |
| Size of the dev run | **20 tasks, not 100.** The full table (about $7 by my first estimate, closer to $9 in fact) was over budget. The 20 are chosen round-robin, 2 per category. |
| Who ran the enumeration | You, manually. |
| Heuristic desirability η | **Dropped altogether.** The branch-selection formula loses η and β (Section 5.1). |
| Selection and stopping | **No early stopping in the main runs** (it would stop the search before ACO can act). Early stopping is an offline analysis. **The final selection rule was reopened after this phase** (best of all, of the last iteration, of the last few, or majority vote): see PHASE4.md D2. |
| v1-rule row | **Removed** from the Phase 4 plan. |
| More dev tests | **None.** The 20-task dev sample stands. |
| Update-rule deposit | Top-only (invalid answers deposit nothing). This was my recommendation and is **still assumed, to be confirmed** before Phase 4. |

---

## 2. The tooling

- **Enumerator** (`scripts/enumerate_paths.py`, logic in `src/swarm_planner/path_table.py`): walks all 243 paths of every task (606 distinct LLM calls) and scores each answer with the Phase 1 verifier. Details of the output are in Section 3.
- **Same requests as the solver.** The prompt text lives in `tot/prompts.py`, and both the graph and the enumerator build their messages there, so the table's cached answers are reused by ToT-only and ToT+ACO runs.
- **Balanced task order.** The dev file lists tasks category by category (the first 50 are all 3–7 people on one day, and the hard 2-person multi-day tasks come last). The enumerator processes tasks round-robin over the 10 categories by default (`--order interleaved`), so `--limit 20` gives exactly 2 per category.
- **Safe to interrupt.** Every answer is cached first, and an interrupted or failed run still writes the table for the tasks it finished.
- **Simulator** (`src/swarm_planner/simulate.py`, report in `scripts/simulate_aco.py`): replays ToT-only and ToT+ACO searches offline from the table, with any budget, update rule and selection policy, at no LLM cost. Its branch probabilities, pheromone update and answer selection mirror the solver's (a test checks the selection order against the solver's own sort key).
- **Analysis report** (`scripts/analyze_path_table.py`) and **solver check** (`scripts/validate_simulator.py`).
- After the final decisions the simulator also reports, for a full-budget run, the verified share per iteration, last-iteration and majority-vote accuracy, and an estimate of the Phase 4 cost (`scripts/simulate_aco.py`, Sections 6 and 7).
- 98 tests pass (87 before the analysis, 11 new for the simulator).

---

## 3. The run (2026-11-05)

| Item | Result |
|---|---|
| Tasks and paths | 20 tasks, 4,860 paths, 634 correct (13.0%) |
| LLM calls | 12,120 (1,212 cache hits) |
| Paid / without cache | **$1.59** / $1.72 |
| Wall time | 24 minutes. The average paid call took 2.1 s. |
| Errors | none. Every paid call (10,908) was answered by DeepInfra. |
| Answers that stopped at `max_tokens=400` | 51 of 12,120 (0.42%) |

The estimate was low: the real cost is about **$0.086 per task**, not $0.067, because the task I measured first was among the cheapest. Longer tasks (more people or days) have longer prompts (about 2.8k tokens per call on average).

**Integrity checks** (all passed): 243 distinct paths and 606 distinct requests per task; 2 tasks in each of the 10 categories; every correct path scores 1.0 and is hard-valid; no `task_error` events; one `llm_call` event per node.

Output files in `results/path_table/dev_20/`:

| File | Content |
|---|---|
| `path_table.csv.gz` | One row per path (4,860 rows): `task_id`, `path`, `b1`…`b5`, `req1`…`req5`, `req_final`, `answer`, `score`, `hard_valid`, `soft_satisfied`, `soft_total`, `slot_rank`, `correct`. |
| `nodes.jsonl.gz` | One row per LLM call (12,120 rows) with the answer text, tokens and finish reason. |
| `events.jsonl.gz` | `run_start` (config, task list, git commit, packages), one `llm_call` per call, one `task_result` per task, `run_end`. |
| `summary.json`, `config.resolved.json` | Totals and settings. |
| `analysis/analysis_report.txt` | The full analysis output (Section 4). |
| `analysis/simulation_report.txt` | The full simulation output (Section 5). |
| `analysis/simulator_validation.txt` | Simulator against the real solver. |

---

## 4. What the table shows

All intervals are 95% bootstrap intervals over the 20 tasks. The task is the unit that counts, so intervals are wide.

### 4.1 Generation is the limit, not search or selection

- **Half of the tasks cannot be solved by any path.** 10 of 20 tasks have no correct path among the 243, and none has every path correct. The path-level accuracy is 13.0% on average but with a median of only 1.4%, so a few tasks carry the average.
- **The unsolved tasks fail in three ways:**
  - 7 of the 10 never produce even a hard-valid answer (6 or 7 people, or 2 people over 4–5 days);
  - 2 (both 2 people over 3 days) produce hard-valid answers that violate a preference or the earliest-slot rule;
  - 1 has a path that the verifier scores 1.0 but that is not the gold answer (several valid answers, F11).
- **By category,** the 2-person 3, 4 and 5-day tasks are all at zero (6 tasks). Single-day tasks with 2–5 people are the most solvable.
- **Searching more cannot help much.** ToT-only with the full 48 traversals reaches **48.6%** (interval 27.5–69.7%), against an oracle ceiling of **50%** (the 10 solvable tasks). The verifier's selection loses nothing relative to the oracle.
- **The simulator is faithful.** On the 10 solvable tasks the real solver (replayed from the cache, 30 seeds) scores 75.3% for ToT-only and 77.7% for ACO v1, against 76.9% and 77.0% predicted (standard error about 2 points, per-task correlation 0.95 and 0.99). The simulated ToT-only and ACO v1 accuracies with last-iteration selection (38.3% and 38.7%) also match v1's reported 39.5% and 40.0% on the test set.

### 4.2 Which levels matter (point 5)

Branch accuracy minus the level mean, in points:

| Level | Effect |
|---|---|
| **L1** (parse) | constraint-first **+6.6 [+0.9, +13.0]**, preference-first **−4.5 [−8.0, −1.6]**, people-first −2.1 [−6.9, +2.0]. Clearly more than chance (permutation p < 0.001). |
| L2 (organise calendars) | all within ±1.5 points; barely above chance (p = 0.02). |
| **L3, L4, L5** | all within ±1 point; **indistinguishable from chance** (p = 0.33, 0.34, 0.58). |

- **The effective choice is at L1.** L3–L5 are interchangeable on this sample, so their 27 × 3 sub-branches add search diversity but no learnable structure. A level-1 plus level-2 model explains about 43% of the variation within a task (descriptive), and the task itself explains 32% of the total variation.
- **The deeper levels add little accuracy.** The slot the model states at L3 equals gold in 9.2% of paths that state one, at L4 in 11.4%, at L5 in 11.6%. The finalize call changes the slot of level 5 in about 23% of paths, so it is not a pure copy.
- **Intermediate answers are often not real computation.** A few L1 and L2 answers just repeat the instruction (6–7%), and 19% of L3 answers already announce a final answer, sometimes outside the work hours. L4 and L5 answers are short (34 and 27 tokens on average).
- The tree is not what limits the result: the 4B model cannot reliably do the slot arithmetic, and no choice of prompt wording at L3–L5 changes that.

### 4.3 The heuristic η (point 6)

- **η gives no benefit.** An η-guided agent (before any pheromone learning) is 12.9% accurate, against 13.1% for a uniform agent (difference −0.2 [−1.1, +0.8] points).
- **At L1, the only level that matters, η points the wrong way.** In the 10 tasks where L1 branches differ, η's favourite branch is the best one in 10% of the tasks and the worst one in 70%. It favours preference-first whenever the task has a preference, and that is the weakest branch. At L2–L5 η's favourite is about as often best as worst.
- η takes only the values 1.0, 1.2 and 1.5, so with β = 2 it shifts branch probabilities by at most a factor 2.25. It biases the search without helping it.

### 4.4 The overall design (point 8)

All figures are simulated, 400 searches per task, 48 traversals unless stated. "Last" is the best answer of the last iteration (the v1 rule), "all" the best over all iterations.

| Setting | last | all | difference vs ToT-only, last | difference, all |
|---|---|---|---|---|
| ToT-only | 38.3 | 48.5 | | |
| **ACO v1 rule** | 38.7 | 46.1 | +0.4 [−2.8, +3.9] | **−2.4 [−6.6, −0.1]** |
| η only (no learning) | 37.7 | 47.6 | −0.6 [−2.5, +1.2] | −0.9 [−3.0, +0.2] |
| pheromone only (β = 0, v1 deposit) | 42.4 | 48.1 | +4.1 [+1.7, +6.7] | −0.4 [−1.1, 0.0] |
| ACO, valid-only deposit | 45.6 | 46.8 | +7.3 [+2.2, +13.1] | −1.7 [−5.3, +0.2] |
| ACO, top-only deposit | 46.6 | 47.6 | +8.3 [+3.1, +14.1] | −0.9 [−3.0, +0.2] |
| ACO, best-of-iteration deposit | 45.1 | 46.9 | +6.8 [+2.2, +12.0] | −1.7 [−5.1, +0.1] |
| ACO, (score − mean) deposit | 45.0 | 47.5 | +6.7 [+2.3, +11.8] | −1.1 [−3.4, +0.2] |
| pheromone only, top-only deposit | 47.8 | 48.5 | +9.6 [+3.6, +15.8] | 0.0 |

1. **Selection policy matters most.** For ToT-only, selecting over all iterations instead of the last one raises accuracy by **+9.8 points [+3.5, +16.7]**, at no LLM cost. This is finding F8 quantified.
2. **The v1 ACO "gain" is an artefact of last-iteration selection.** With the v1 rule ACO is no better than ToT-only (+0.4) and slightly worse when selecting over all iterations (−2.4). With a better deposit rule ACO gains 7–10 points under last-iteration selection, but this only compensates for discarding 40 of the 48 traversals. Under best-of-all selection ToT-only already reaches the same 48.5%.
3. **With an exact verifier and a "deposit only on success" rule, ACO cannot improve best-of-all accuracy.** The pheromone changes only after a verified answer has been found, and by then the search has already succeeded. Graded deposits (valid-only, v1) give a signal earlier but did not help either.
4. **The pheromone does learn when the deposit is informative.** The probability on the best level-1 branch (uniform is 0.333) goes from 0.292 to **0.295** with the v1 rule (no learning), to 0.545 with a top-only deposit, and from 0.333 to 0.666 with pheromone only and a top-only deposit. This is the kind of evidence Req. 4 asks for, and it needs the fixed deposit.
5. **The n × T split does not matter for best-of-all** (48.8% for every split from 48 × 1 to 1 × 48). For last-iteration selection ToT-only collapses from 48.8% (48 × 1) to 13.2% (1 × 48), while ACO with a top-only deposit stays at about 48%. ACO's learning is exactly what rescues last-iteration selection.
6. **Budget.** With best-of-all selection, accuracy is 38.5% with 8 traversals, 45.0% with 16, 46.9% with 24 and 48.9% with 48. Halving the budget to 24 costs about 2 points.
7. **Early stopping is free accuracy-wise and saves cost.** Stopping after the first iteration that contains a verifier score-1.0 answer gives 49.0% against 48.5% (difference +0.4 [0.0, +1.1]) and uses **62% of the budget**: 1.5 of 6 iterations on the solvable tasks, 5.95 on the unsolvable ones, where all budget is spent.

---

## 5. Final decisions and the frozen v2 method

### 5.1 The frozen v2 method

| Part | Specification |
|---|---|
| Tree and budget | The 5-level, 3-branch tree as in v1, n = 8 agents, T = 6 iterations (48 traversals), temperature 0. |
| Branch probability | **No heuristic.** p_ij = (τ_ij + ε)^α / Σ_k (τ_ik + ε)^α with α = 1.0 and ε = 0.001. The pheromone starts at τ₀ = 1.0, so the first iteration is uniform. |
| Quality signal | The fixed verifier from Phase 1. |
| Pheromone update | τ ← (1 − ρ)·τ + Σ Q with ρ = 0.1. A trace deposits Q = 1 on each of its edges if the verifier scores its answer 1.0, otherwise Q = 0. Invalid answers deposit nothing (the F7 fix). |
| Answer | Chosen by a final selection rule that is **open** (PHASE4.md D2). The candidates use the solver's ordering (verifier score, hard validity, soft ratio, earliest slot, path score, lowest agent id) over all iterations, the last iteration, or the last few. Majority vote needs no ordering. |
| Stopping | **None in the main runs.** The full 48 traversals are always run, so the pheromone can act. Early stopping is derived offline from the logs. |

### 5.2 How the Phase 3 recommendations were decided

| Recommendation | Decision |
|---|---|
| Select over all iterations, not the last one (+9.8 points, no cost) | **Adopted at first, then reopened**: best-of-all gives ACO no effect on the headline. The choice of the final selection rule is now open (PHASE4.md D2). |
| Early stopping at the first verified answer (same accuracy, about 38% fewer calls) | **Not used in the main runs** (Section 5.3). Reported as an offline analysis. |
| v2 deposit: nothing for invalid answers | **Adopted**, top-only. This is my recommendation, assumed rather than explicitly confirmed. |
| Drop η, or report it as an ablation | **Dropped altogether.** The formula loses β. The ablation is reported from the dev simulation. |
| Keep the 5-level tree and the 8 × 6 budget | **Kept** (decision 3). |
| Verifier look-ahead heuristic | Not adopted (Section 5b). |
| More dev tests | **None.** |
| The v1-rule row of the published paper | **Removed.** Only the frozen v2 method and ToT-only are run. |

### 5.3 How ACO acts in the final design

**Why early stopping had to go (finding F15).** With a top-only deposit, early stopping and best-of-all selection, ACO cannot act. The pheromone only changes after a verified answer, and a verified answer ends the search. ToT+ACO and ToT-only were identical in simulation (48.6% accuracy, 3.73 of 6 iterations), and there was almost no pheromone movement to show for Req. 4.

**With the full budget ACO changes the runs, but not the headline.** Simulated on the dev sample (400 searches per task, 8 × 6, best trace over all iterations; differences with a 95% interval over the 20 tasks):

| Measure | ToT-only | ACO v2 | Difference |
|---|---|---|---|
| Verified share of traces, iteration 1 | 13.3% | 13.3% | +0.0 |
| Verified share of traces, iteration 6 | 13.0% | **32.4%** | **+19.3 [+9.6, +29.6]** |
| Verified share, mean over the run | 13.1% | 23.6% | +10.5 [+5.0, +16.3] |
| Accuracy, best over all iterations (headline) | 48.6% | 48.6% | +0.0 |
| Accuracy, best of the last iteration | 38.2% | 47.7% | +9.4 [+3.5, +15.8] |
| Accuracy, **majority vote** over the 48 traces (no verifier at selection time) | 13.3% | **36.3%** | **+23.0 [+9.0, +38.0]** |
| Probability of the best level-1 branch, iteration 1 → 6 | 0.333 → 0.333 | 0.333 → **0.661** | |

How to read it:
- **The best-of-all accuracy cannot differ.** With an exact verifier the best answer is secured at the first success, whatever the later iterations do. The paper has to say that ACO adds no best-of-all accuracy at equal budget (Req. 5).
- **ACO concentrates the swarm on verified answers.** From iteration 2 on, ACO draws far more verified traces than ToT-only. Because of that a plain **majority vote** over the traces becomes reliable (36% against 13%), which is a result that does not depend on the verifier at selection time. The last iteration also becomes much better.
- **These are the measures Req. 4 asks for,** together with the pheromone trajectory and the branch probabilities, all of which are logged.
- **Caution:** 20 tasks (10 solvable), so the intervals are wide, and Phase 4 on the test set is the real check. The verifier is still used to *update* the pheromone, which the paper must state.

### 5.4 Code changes that follow (Phase 4 preparation, not made yet)

- Remove the heuristic: `aco/heuristics.py`, its tests, the β parameter and the use of task features for η.
- Switch the solver's answer to the best trace over all iterations, keeping last-iteration selection derivable from the logs.
- Add the top-only deposit and wire it into the pheromone update (`deposit_quality` already exists from Phase 1).
- No early-stopping logic in the solver: it is computed offline from the logged iterations.
- Keep logging every trace, so all selection policies can be derived offline.
- Add the baseline and sampling-control runners, and the configs.

---

## 5b. Check of a verifier look-ahead heuristic (offline, $0)

**Idea.** The verifier is exact, so it can also judge the slot that an intermediate answer already states (by level 3, 52% of paths state exactly one slot). That could serve as a better heuristic: prune paths whose partial answer is already invalid, or give the pheromone an earlier signal. Script: `scripts/analyze_lookahead.py`, report: `analysis/lookahead_report.txt`.

**1. The signal is real.** A verified (top-scoring) slot at level 3 ends in a correct final answer in **74%** of paths, at L4 in 81% and at L5 in 83%. An invalid slot at level 3 ends correct in only **5.7%** (L4: 2.8%). A hard-valid but not top-scoring slot is never correct.

**2. Pruning is harmless but gains little.**
- Pruning a path as soon as a level states an invalid slot is right 94–97% of the time and cuts 1.2–1.7 of 6 calls per traversal. But it also **drops 13–22% of the correct paths** (rule dependent), because the later calls sometimes repair an invalid slot. No task loses all of its correct paths.
- At an **equal number of LLM calls**, accuracy changes by **+0.4 to +0.6 points at 288 calls** (intervals [0.0, +1.2] to [0.0, +1.5]), and by about +1.0 at small budgets (72 calls). No accuracy loss at any budget.
- On top of early stopping it saves only about 2% of the calls (164 → 161 on average; 43 → 38 for solvable tasks), because the budget is dominated by the unsolvable tasks, which use nearly all 288 calls either way.

**3. An early signal does not help the pheromone either.** Adding "the level-3 slot is verified" as an extra deposit changes best-of-all accuracy by +0.0 to +0.1 points at every budget, and the probability on the best level-1 branch only from 0.662 to 0.691.

**Verdict.** It works as a signal and does no harm, but the gain is too small to justify a second parsing path and a 13–22% false-prune rate. **Not recommended as part of the main method.** The bigger levers are best-of-all selection (+9.8 points) and early stopping (38% fewer calls).

**What follows for the heuristic.** I suggested an evidence-based η at level 1 only, or dropping η; the decision was to drop it (Section 5.2). If you want to cut cost further, the lever to look at is the **unsolvable tasks**, which consume almost all of the budget: a rule that gives up when no hard-valid answer has appeared after some number of traversals. That is untested, and risky for rarely solvable tasks.

---

## 6. Limits and cautions

- **20 tasks.** The headline accuracy has an interval of 27.5–69.7%. Level-1 findings rest on 10 informative tasks. Per-category numbers are mostly noise (2 tasks each).
- **One LLM realisation.** The cache freezes one answer per request, so conclusions describe that realisation.
- **F11.** "Correct" (equals gold) and "verifier score 1.0" differ on tasks with several valid answers (1 task here).
- **The simulator** is validated against the real solver on solvable tasks only (unsolvable tasks give 0 in both).
- **Identical prompts share cached answers across tasks.** This only matters if two tasks had identical text, which is not the case.

---

## 7. Budget note for Phase 4

Phase 4 is where most of the money goes. With the full budget and no early stopping, a call costs about $0.00014 on average and one 400-task ToT run costs about **$9** (about $0.022 per task for the distinct requests of 48 traversals). The six ToT-based runs (ToT-only and ToT+ACO, three seeds each) share the cache and cost about **$25** together. This comes from `scripts/simulate_aco.py` (Section 7), which uses the real per-request costs of the dev table, so it is a decent estimate but extrapolated from 20 tasks. Adding the temperature-0.7 sampling control (48 samples per task, three seeds, about 58k single calls, roughly $5–7) and the single-call baseline (about $0.1–0.3) gives **about $30–35 in total**. If that is too much, the options are fewer configurations or a smaller test subset (for example 200 tasks, about half), which we should decide before Phase 4 starts.
