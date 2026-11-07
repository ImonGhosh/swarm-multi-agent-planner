# Phase 1 — Audit and fix the verifier

**Status:** done. No LLM calls were made in this phase.

**Goal:** make the deterministic verifier a trustworthy quality signal (decision 5, option 1). It fixes F5 and F6, validates against all 1000 gold answers, prepares the deposit mapping (F7), and reports precision, recall and the verifier-as-solver reference row.

---

## 1. Summary

| Task in PLAN.md | Result |
|---|---|
| Fix name splitting (F5) | Done. Names such as Doris, Alexis and Dennis are no longer split. The glued-word fixes now apply only where a schedule verb follows. |
| Strict priority hard → soft → earliest (F6) | Done. The scorer ranks slots by soft preferences met, then by time. "Earliest" is judged against that ranking. |
| Check against all 1000 gold answers | Done (`scripts/audit_verifier.py`). Gold scores 1.0 and is a top-scoring slot on **1000/1000**. |
| Test on gold answers | Done (`tests/test_verifier_gold.py`). |
| Deposit mapping (F7) | Partly done. `deposit_quality()` exists. Choosing and wiring a rule is the Phase 3 update-rule decision. |
| Precision, recall, verifier-as-solver | Done, see Section 3. |

**Main finding (F11):** the plan assumed Natural Plan has one solution per task. It does not. 166 of the 1000 tasks have several equally good answers, and in 66 of them the gold answer is not the earliest. The official evaluator is exact match, so a valid answer that differs from gold counts as wrong. No deterministic verifier can reach 100%. The target in PLAN.md was corrected accordingly.

48 tests pass: the 33 from Phase 0, 13 new verifier unit tests and 2 gold-audit tests.

---

## 2. What was done

### 2.1 Parser fixes (`verifier/parser.py`)

- **F5, name splitting.** v1 rewrote every `Xis` as `X is`, which turned Doris into "Dor is". The participant then matched no schedule line and their busy times were dropped. The `is` rule never fixed any real typo in the data. The `has` rule is still needed (`Ethanhas meetings`), but it also hit names, because `Dennishas` was split into "Dennis has" and then into "Denn is". Both rules now apply only when a schedule word follows: `has` before `meetings|blocked|no meetings`, and `is` before `busy|free`.
  - F5 affected **52 of the 1000 tasks**, 15 of them in the test set, which matches the plan.
- **Preferences are read sentence by sentence.** The dataset sometimes splits one preference over several sentences, for example "X do not want to meet on Monday. Tuesday before 11:00." v1 applied the bound to every day in the clause. Now a bare day means the whole day, and a bound applies only to the days in its own sentence.
  - Evidence: with both readings the gold answer stays valid on all 1000 tasks, but the per-sentence reading removes false "valid" slots and makes the first-ranked slot match gold on 934 tasks instead of 915.
- **Parser warnings.** `ParsedTask.warnings` lists anything the parser could not explain: a schedule statement that did not match, a participant without a schedule, a statement for an unknown participant, a preference clause that was skipped or names no day, an "earliest" mention that was not parsed, and any gap between the number of `H:MM to H:MM` intervals in the text and the number parsed. The last check catches missed busy times, which a gold check cannot, because gold can only reveal a verifier that is too strict. **Warnings are empty on all 1000 tasks.**
- **"Free the entire week"** was not recognised, only "day". It was harmless for scoring but triggered a warning, so it is fixed.
- **Missing days.** A header with no recognisable days now raises `ValueError`. v1 silently defaulted to Monday.

### 2.2 Scorer changes (`verifier/scorer.py`)

- **Total ranking (F6).** `ranked_slots` sorts the hard-valid slots by soft preferences met (more is better) and then by time. `slot_rank` is the position in this list, so rank 0 is the best slot. The "earliest availability" criterion is met when `slot_rank == 0`. v1 used the position among all hard-valid slots.
- **`best_slot`** replaces `earliest_slot` in the result and the logs. It is the first-ranked slot, which is also what verifier-as-solver answers.
- **`top_slots()`** returns every slot that attains the top score. More than one means the task has several equally good answers.
- **Score formula unchanged.** Invalid answers score at most 0.49 × 4/5 = 0.392, and hard-valid answers score 0.5 to 1.0. A test asserts the two ranges never overlap.
- **`deposit_quality(result, mode)`**:
  - `v1_score` deposits the raw score. Wrong answers still deposit up to 0.392 (F7).
  - `valid_only` deposits the score for hard-valid answers and 0 otherwise.
  - It is **not** wired into `apply_pheromone_update` yet. The v1 rule has to stay available for the v1-rule row in Phase 4, and the v2 rule is chosen in Phase 3.
- **Selection is unaffected.** `solver._solve_trace_sort_key` already breaks ties with `slot_rank`, so among equally scored answers the earliest one still wins.

### 2.3 New script, data and tests

| File | Purpose |
|---|---|
| `scripts/audit_verifier.py` | Scores gold and every hard-valid slot, classifies each task, writes `audit_tasks.csv`, `audit_summary.json` and `known_exceptions.json`. |
| `data/verifier_known_exceptions.json` | The 66 tasks where gold is not the first-ranked of several equally good answers. |
| `results/verifier_audit/` | Audit output for the fixed verifier. |
| `results/verifier_audit/v1_verifier_baseline/` | The same audit run with the old verifier, for the before/after comparison. Its summary uses an older, simpler format. |
| `tests/test_verifier.py` | 13 new tests: the F5 names, the glued `has`, F6, ranking, split sentences, score ranges, deposit modes, warnings and the missing-days error. |
| `tests/test_verifier_gold.py` | Gold audit on the 500 in-repo tasks, and on all 1000 when `natural-plan/` is available. |

---

## 3. Results

### 3.1 Before and after (all 1000 tasks)

| Measure | Old verifier | Fixed verifier |
|---|---|---|
| Gold scores 1.0 | 965 | **1000** |
| Gold is a top-scoring slot | 997 | **1000** |
| Gold is the only top slot | 754 | 834 |
| Tasks with several top slots | 243 | 166 |
| Parser warnings | not tracked | 0 |
| Verifier-as-solver accuracy | 86.0% | **93.4%** (934/1000) |

### 3.2 Task classes

| Class | Meaning | All 1000 | Test 400 | Dev 100 |
|---|---|---|---|---|
| `unique` | gold is the only top slot | 834 | 330 | 83 |
| `tie_gold_is_best` | several top slots, gold is the one ranked first | 100 | 41 | 13 |
| `tie_gold_not_best` | several top slots, gold is not ranked first | 66 | 29 | 4 |
| `gold_not_top` | verifier defect | 0 | 0 | 0 |
| `parse_error` | unparsable text | 0 | 0 | 0 |

All 166 tasks with several top slots have no earliest-availability request. When the task asks for the earliest availability, the answer is a single slot.

### 3.3 Verifier-as-solver (reference row)

| Set | Old | Fixed |
|---|---|---|
| Test 400 | 85.75% (343) | **92.75% (371)** |
| Dev 100 | 88% | **96% (96)** |
| All 1000 | 86.0% | **93.4%** |

For comparison, v1 ToT+ACO reached 40.0% on the test set, so the verifier alone beats the swarm by a wide margin. This is the F4 limitation and it goes into the paper.

By category (all 1000 tasks, 100 per category), the fixed verifier's accuracy ranges from 87% (2 people, 1 day) to 100% (6 people, 1 day). The ties and the exceptions concentrate in the categories with the most valid slots: 2 people with 1–4 days, and 3 people with 1 day.

### 3.4 Precision and recall against gold (all 1000 tasks, slot level, gold is the only positive)

| Decision | Recall | Precision |
|---|---|---|
| Hard-valid | 1.000 | 0.238 |
| Accepted (top-scoring) | 1.000 | 0.779 |

The verifier never rejects gold. The imperfect precision comes from tasks with several valid answers, where the verifier accepts all of them and only one is gold.

### 3.5 What the exceptions look like

Example, `calendar_scheduling_example_1`: Raymond, Billy and Donald, half an hour, Monday. Billy would like to avoid meetings after 15:00. Four slots meet every constraint (9:30, 11:00, 13:30 and 14:30). Gold is 14:30, the verifier's first-ranked slot is 9:30. Nothing in the text singles out 14:30.

The full list is in `data/verifier_known_exceptions.json`.

### 3.6 Whose limitation is it? (for the paper's wording)

The ambiguity is a property of the **Natural Plan benchmark**, not of our pipeline:

- Many tasks have several answers that meet every constraint, and the official evaluator (`scripts/evaluate.py`, copied unchanged) is exact match against a single gold answer. Other valid answers are marked wrong.
- The task prompt itself only says "there exists a solution", not a unique one.

Some parts of the framing are ours, and the paper should word them carefully:

- **The counts come from our verifier.** 166 tasks with several top answers and 66 where gold is not the earliest are our measurements. They are backed by 0 parser warnings, a complete busy-interval check and gold passing on all 1000 tasks, plus two hand-read examples, but they are not an official figure.
- **The 93.4% ceiling depends on our tie-break** (earliest top slot). Another tie-break gives another ceiling.
- **Our selection rule prefers the earliest of equally scored answers.** It matches gold in about 60% of the tie cases, which is a pattern in the data, not a guarantee.
- **We have not checked how the Natural Plan authors generate the gold answers**, nor whether they acknowledge this. The earlier assumption in PLAN.md that Natural Plan has a single solution was ours and was wrong. In the paper, write "we observe that…" and cite the audit, not a claim about the authors' intent.

**Reporting plan.** Keep exact match as the main metric, so results stay comparable with Natural Plan. Add the verifier's valid-answer rate (score 1.0) as a secondary column. The gap between the two shows how much of any system's misses come from this ambiguity, and it costs nothing because the verifier already produces it. **Open:** reading the Natural Plan paper on how gold is generated, before the paper is revised (Phase 8).

---

## 4. Decisions and differences from v1

- **"Earliest" is not rewarded unless it is requested.** On tasks without an earliest request, gold is the earliest top slot in 100 of the 166 tie cases (60%). Adding a bonus for earliness would raise accuracy a little, but it would encode a habit of the dataset in the score. The selection tie-break already prefers the earliest slot, so the effect is kept where it is harmless.
- **Gold-answer target changed.** The plan expected gold to be the unique top slot. The audit shows this is impossible for 166 tasks, so the target is "gold is a top slot, and unique whenever the earliest is requested".
- **Score values are unchanged,** so the v1-rule row in Phase 4 stays comparable. Only the inputs to the score changed (parsing and ranking).
- **`earliest_slot` renamed to `best_slot`** in the quality details, because it is no longer the earliest hard-valid slot. Nothing else in the code reads it.

---

## 5. Verification

| Check | Result |
|---|---|
| `pytest` | 48 passed |
| Audit on all 1000 gold answers | 0 parse errors, 0 warnings, gold hard-valid and score 1.0 on all 1000 |
| Gold-answer test | passes on the 500 in-repo tasks and on all 1000; the 66 exceptions match `data/verifier_known_exceptions.json` exactly |
| Busy-time completeness | every `H:MM to H:MM` interval in the text is parsed, on all 1000 tasks |
| F5 reproduction | 15 of the 400 test tasks had a mangled participant before the fix, 52 of 1000 |

---

## 6. Known limitations carried to later phases

| Item | Phase |
|---|---|
| F7: the deposit mapping is not wired into the pheromone update. Whether invalid answers should deposit nothing, and whether to subtract a baseline, is part of the v2 update rule. | 3 |
| F8: the answer is still the best trace of the last iteration only. | 3, 4 |
| F11: exact match rejects valid answers on 166 tasks. The path table in Phase 3 must keep "correct" (equals gold) apart from "verifier score 1.0". | 3, 8 |
| Off-grid answers (for example 9:15 - 9:45) can be hard-valid but never equal gold. Models rarely produce them. Revisit if the path table shows any. | 3 |
| The effect of the F5 and preference fixes on the F7 pheromone bias can only be measured with LLM traces. | 3, 5 |

---

## 7. How to reproduce

From the v2 root:

```bash
.venv/Scripts/python.exe -m pytest                       # 48 tests
.venv/Scripts/python.exe scripts/audit_verifier.py       # needs ../../natural-plan/ ; writes results/verifier_audit/
.venv/Scripts/python.exe scripts/audit_verifier.py --source data/curated_400.json --out-dir /tmp/audit_test
```

On Linux or macOS the interpreter path is `.venv/bin/python`. After any verifier change, rerun the audit and copy `results/verifier_audit/known_exceptions.json` to `data/verifier_known_exceptions.json` only if the change is intentional.
