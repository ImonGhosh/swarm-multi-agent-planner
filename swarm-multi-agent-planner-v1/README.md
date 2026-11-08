# Swarm Calendar Planning

This project solves calendar scheduling tasks with a fixed 5-level Tree-of-Thought (ToT) workflow and an Ant Colony Optimization (ACO) branch-selection policy.

At a high level, each task asks the model to find a meeting slot that satisfies:

- participant availability
- meeting duration
- allowed day(s)
- work-hour limits
- optional soft preferences such as "earliest availability" or "avoid Tuesday after 14:00"

The current recommended workflow is **per-instance solve mode**:

- each task is solved independently
- for each task, `n` agents run for `T` iterations
- each agent follows one 5-step ToT path
- if `aco=true`, local pheromones are updated across iterations for that single task
- if `aco=false`, the same search budget is used but branch choices are random
- the final answer is chosen from the **best trace in the last iteration**

## Project Idea

The project combines three ideas:

1. **Tree-of-Thought (ToT)** gives a fixed reasoning structure instead of asking the model to jump directly to the final answer.
2. **ACO** biases which branch is chosen at each ToT level, based on both heuristics and pheromone learned during the current task.
3. **A deterministic verifier** creates quality scores for each prediction, by comparing it against the parsed task constraints, so candidate answers can be ranked.

The design goal is simple: generate multiple structured candidate solutions, score them reliably, and use ACO to gradually favor more promising reasoning paths within each task.

## The 5-Level ToT

Each agent always moves through the same 5 reasoning levels. The difference between agents is which branch they choose at each level.

### Level 1: Parse the task

Purpose: understand what must be scheduled.

Typical branches:

- `constraint_first`: extract duration, day(s), work hours, and preferences first
- `people_first`: identify participants first, then their constraints
- `preference_first`: focus on special preferences first, then the rest

Intuition: before doing any scheduling, the model must understand the problem statement correctly.

### Level 2: Organize calendars

Purpose: convert raw schedule text into a usable internal view.

Typical branches:

- `participant_wise`: build busy slots participant by participant
- `day_wise`: organize busy times by day
- `mixed_normalization`: normalize by day, then regroup by participant

Intuition: the raw input is free text, so this level tries to structure the calendars before reasoning about free time.

### Level 3: Generate feasible availability

Purpose: find possible meeting windows.

Typical branches:

- `free_slot_computation`: compute free intervals and intersect them
- `candidate_slot_enumeration`: list candidate windows and eliminate invalid ones
- `progressive_elimination`: start from the full day and remove conflicts step by step

Intuition: this is where the agent turns calendar text into candidate meeting slots.

### Level 4: Apply preferences / selection policy

Purpose: choose among valid candidates.

Typical branches:

- `earliest_valid`: prefer the earliest valid slot if requested
- `preference_priority`: apply explicit soft preferences first
- `conservative_selection`: choose a clearly safe slot with margin from nearby meetings

Intuition: many tasks have several valid slots, so this level decides which one is best.

### Level 5: Final answer and verification

Purpose: produce the final answer in the required format.

Typical branches:

- `direct_output`: return the chosen slot directly
- `one_step_self_check`: verify once, then return
- `format_check_output`: verify day, time, duration, and final format

Intuition: this final step converts the reasoning path into a single formatted prediction:

```text
Here is the proposed time: Monday, 10:30 - 11:00
```

## ACO Branch Selection

At each ToT level, the agent must choose one branch. If `aco=true`, the branch is sampled from an ACO probability distribution:

<!--```text
P(branch j at level i) =
((tau_ij + epsilon)^alpha * (eta_ij + epsilon)^beta)
/ sum over all branches k at that level
```-->
<img width="305" height="72" alt="image" src="https://github.com/user-attachments/assets/4eb30c2c-1d30-4f82-bbac-82b85eaff5b0" />

Where:

- `tau_ij` = pheromone value for that branch edge
- `eta_ij` = heuristic desirability score for that branch on the current task
- `alpha` = how strongly pheromone matters
- `beta` = how strongly the heuristic matters
- `epsilon` = small smoothing constant

### Heuristic desirability

The heuristic is hand-designed from simple task features such as:

- number of people
- number of allowed days
- duration
- whether preferences exist
- whether "earliest" is requested

Example intuition:

- multi-day tasks may favor `day_wise`
- preference-heavy tasks may favor `preference_first`
- earliest-request tasks may favor `earliest_valid`

If `aco=false`, the branch is chosen uniformly at random instead of using the formula above.

## Pheromone Update Formula

After an iteration finishes, pheromones are updated for the current task only.

For each branch edge:

<!--```text
tau_ij <- (1 - rho) * tau_ij + sum_k Delta_ij^(k)
```-->
<img width="281" height="72" alt="image" src="https://github.com/user-attachments/assets/63db9da3-ca65-46db-914f-768a080f0660" />

with:

<!--```text
Delta_ij^(k) = Q_k if edge (i,j) is in agent k's path, else 0
```-->
<img width="553" height="95" alt="image" src="https://github.com/user-attachments/assets/a21a8419-a99e-4764-a21c-256b2ef4a27b" />

Where:

- `rho` = evaporation rate
- `Q_k` = quality score of agent `k`

Interpretation:

- **evaporation** reduces older influence
- **deposit** rewards branches used by better-scoring traces

In solve mode, these pheromones are **local to one task**. They are initialized fresh for that task, updated across iterations, and then discarded before the next task.

## How a Full Run Works

Assume the full dataset has **1000 tasks**.

### Solve mode (`mode=solve`)

This is the main workflow for current experiments.

Example commands:

For Gemma, solve mode with ACO enabled:

<pre>python infer_aco_tot_calendar.py <span style="color:red;"><strong>--mode</strong></span> solve <strong>--data_path</strong> ../data/calendar_scheduling_input.json <strong>--out_path</strong> ../data/calendar_scheduling_output_test_small.json <strong>--model</strong> google/gemma-3-4b-it <strong>--agents_per_task</strong> 8 <strong>--iterations_per_task</strong> 6 <strong>--alpha</strong> 1.0 <strong>--beta</strong> 2.0 <strong>--epsilon</strong> 0.001 <strong>--rho</strong> 0.1 <strong>--seed</strong> 42 <span style="color:red;"><strong>--aco</strong></span> true</pre>

For Gemma, solve mode with ACO disabled:

<pre>python infer_aco_tot_calendar.py <span style="color:red;"><strong>--mode</strong></span> solve <strong>--data_path</strong> ../data/calendar_scheduling_input.json <strong>--out_path</strong> ../data/calendar_scheduling_output_test_small.json <strong>--model</strong> google/gemma-3-4b-it <strong>--agents_per_task</strong> 8 <strong>--iterations_per_task</strong> 6 <strong>--alpha</strong> 1.0 <strong>--beta</strong> 2.0 <strong>--epsilon</strong> 0.001 <strong>--rho</strong> 0.1 <strong>--seed</strong> 42 <span style="color:red;"><strong>--aco</strong></span> false</pre>

Argument summary:

- `--mode solve`: specifies the run mode (possible modes are **solve**, **train** and **infer**. We only use solve mode).
- `--data_path`: input dataset JSON file containing the scheduling tasks.
- `--out_path`: output JSON file where final predictions and run metadata are written.
- `--model`: SLM used by the ToT agents to generate candidate schedules.
- `--agents_per_task`: number of agents `n` launched per iteration for each task.
- `--iterations_per_task`: number of local iterations `T` run for each task.
- `--alpha`: pheromone importance in ACO branch selection.
- `--beta`: heuristic importance in ACO branch selection.
- `--epsilon`: small smoothing constant to avoid zero-probability branches.
- `--rho`: pheromone evaporation rate.
- `--seed`: random seed for reproducibility.
- `--aco true|false`: enables pheromone-guided branch selection when `true`, or random ToT sampling when `false`.

Expected outputs:

- `data/calendar_scheduling_output.json`
  Final prediction file for the dataset. Each item should contain the model prediction in `pred_5shot_pro` plus additional metadata.
- `data/logs/solve_run_*.jsonl`
  Per-task structured run log written during the solve run.

For each task, the workflow logic is as below :

1. Task prompt (`prompt_5shot`) is loaded.
2. A fresh local pheromone table is created for the task.
3. For `T` iterations, following is repeated:
   - `n` agents are launched
   - each agent parallelly traverses the 5 ToT levels
   - at each level, a branch is chosen using ACO (if `aco=true`) or randomly (if `aco=false`)
   - a prediction is produced
   - a quality score is calculated for the current prediction with the deterministic verifier
   - if `aco=true`, local pheromones are updated using the agent quality scores
4. After `T` iterations, the **highest quality prediction from the last iteration** is chosen.
5. That prediction is saved as the final output for the task.

Important properties:

- tasks are independent of each other
- pheromone learning happens **within a task**, not across tasks
- `aco=true` and `aco=false` use the same overall search budget (`n * T`)
- this makes ACO vs non-ACO comparisons cleaner

## Quality Scoring using a Deterministic Verifier

The deterministic verifier is utilized to verify and score the candidate outputs produced by the SLM agents.

It:

- normalizes task text
- parses participants, duration, work hours, and allowed days
- parses busy intervals, including multi-day lines
- parses hard and soft preferences
- parses the candidate prediction
- checks hard validity
- computes soft-preference satisfaction
- gives the candidate a deterministic score

The score is always in the range **0 to 1**:

- if the candidate is **hard-valid** (i.e passes all hard constraints), its score is in the range **0.5 to 1.0**
- if the candidate is **hard-invalid** (i.e does not pass all hard constraints), its score is in the range **0.0 to 0.49**

This creates a strict separation between valid and invalid answers: even the best invalid answer scores below the worst valid answer.

In simple terms, the scoring logic is:

```text
If hard-valid:
score = 0.5 + 0.5 x soft_ratio

If hard-invalid:
hard_ratio = hard_checks_passed / hard_checks_total
score = 0.49 x hard_ratio
```

Where:

- `soft_ratio` is the fraction of satisfied soft preferences, so it lies between `0` and `1`
- `hard_ratio` is the fraction of hard checks passed, so it also lies between `0` and `1`

This means:

- a valid candidate that satisfies all soft preferences gets a score close to `1.0`
- a valid candidate that misses soft preferences still remains in the valid band, starting at `0.5`
- an invalid candidate can receive partial credit if some hard checks are correct, but it can never outrank a valid one

A useful detail is `slot_rank`:

- the verifier enumerates all hard-valid slots (i.e predictions which fulfill all hard constraints)
- sorts them from earliest to latest
- `slot_rank = 0` means the candidate is the earliest valid slot

This gives a clean tie-break signal for "earliest availability" style tasks.

## Main Repository Structure

### Top-level folders

- `Aco-ToT-multi-agent-framework/`: main implementation
- `data/`: input datasets and prediction outputs
- `evaluation_script/`: evaluation utilities

## Main Code Files

Below is a short description of the files most relevant to the **current solve-mode workflow**.

### Core ACO / ToT logic

- `Aco-ToT-multi-agent-framework/aco_tot/config.py`
  Defines the fixed 5 ToT levels, branch names, prompts, and global defaults.

- `Aco-ToT-multi-agent-framework/aco_tot/aco.py`
  Computes ACO branch probabilities and samples a branch at each level.

- `Aco-ToT-multi-agent-framework/aco_tot/heuristics.py`
  Provides heuristic desirability values (`eta`) from task features.

- `Aco-ToT-multi-agent-framework/aco_tot/pheromone.py`
  Applies pheromone evaporation and deposit updates.

### LLM execution and prompt flow

- `Aco-ToT-multi-agent-framework/aco_tot/llm_nodes.py`
  Builds the LangGraph execution graph. Each level node asks the model for one intermediate result, and the final node asks for the final meeting slot.

- `Aco-ToT-multi-agent-framework/aco_tot/context_accumulator.py`
  Builds the accumulated reasoning prompt passed from one ToT level to the next.

- `Aco-ToT-multi-agent-framework/aco_tot/prompt_io.py`
  Extracts the current task from few-shot prompts, builds simple task features, and canonicalizes final answers.

### Solve-mode orchestration

- `Aco-ToT-multi-agent-framework/aco_tot/engine.py`
  Main orchestration file. It runs agents, handles solve mode, applies deterministic scoring, updates local pheromones, logs runs, and manages solve-mode checkpointing.

- `Aco-ToT-multi-agent-framework/aco_tot/invoke.py`
  Public API wrapper for invoking one task or dataset run from code.

- `Aco-ToT-multi-agent-framework/infer_aco_tot_calendar.py`
  CLI entrypoint for the current solve workflow.

### Scoring, storage, and logging

- `Aco-ToT-multi-agent-framework/aco_tot/task_quality.py`
  Deterministic verifier used in solve mode. Parses task constraints and scores candidate predictions.

- `Aco-ToT-multi-agent-framework/aco_tot/store.py`
  Loads, initializes, and saves pheromone state files.

- `Aco-ToT-multi-agent-framework/aco_tot/run_logging.py`
  Writes structured JSONL logs for train, infer, and solve runs.

- `Aco-ToT-multi-agent-framework/aco_tot/types.py`
  Shared dataclasses for task features, traces, pheromone state, and run results.

### Reliability and analysis helpers

- `Aco-ToT-multi-agent-framework/aco_tot/openrouter_resilience.py`
  Retry and provider-handling helpers for OpenRouter or OpenAI-compatible backends.

## Testing Workflow

For current project reporting, we have tested in following order :

1. **Single-agent SLM baseline**
   One model call, no ToT search, no ACO.

2. **Multi-agent multi-iteration ToT only**
   Solve mode with `aco=false`.

3. **Multi-agent multi-iteration ACO + ToT**
   Solve mode with `aco=true`.

This isolates:

- the gain from structured ToT search only (`1 -> 2`)
- the extra gain from pheromone-guided search on top of ToT (`2 -> 3`)

## Practical Takeaway

The core idea of the repo is not "let one model solve the task directly."

Instead, it is:

- decompose reasoning into 5 fixed stages
- explore several branch combinations
- score the resulting candidates deterministically
- and, when ACO is enabled, use local pheromone learning to bias later iterations toward more promising reasoning paths for the same task
