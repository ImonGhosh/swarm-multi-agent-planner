# Conceptual Clarifications

Short answers to conceptual questions about the project, kept in one place. Each entry is self-contained.

**Contents**
1. [Why was the `verified_last3` selection rule decided?](#1-why-was-the-verified_last3-selection-rule-decided)
2. [What is a nonce, and why do the baseline and the sampling control use one?](#2-what-is-a-nonce-and-why-do-the-baseline-and-the-sampling-control-use-one)
3. [Why did we keep the fixed verifier as the quality signal, and why is its strength a limitation?](#3-why-did-we-keep-the-fixed-verifier-as-the-quality-signal-and-why-is-its-strength-a-limitation)

---

## 1. Why was the `verified_last3` selection rule decided?

It was decided on 2026-11-07, before any test-set run. The reasoning started from a v1 weakness and went through four steps.

### Step 1: v1 wasted most of the search

v1 submitted only the best answer of the last iteration, so 40 of the 48 answers were thrown away. On the dev analysis, selecting over all iterations instead gave ToT-only +9.8 points at no cost.

### Step 2: best-of-all alone would hide ACO

- With an exact verifier, the best answer is secured at the first success, often before any pheromone exists. Under best-of-all, ToT-only and ToT+ACO are the same search by construction, so ACO cannot show an effect.
- The professor asked for evidence that ACO influences the search, so a headline that structurally hides it was a poor fit.

### Step 3: look for a rule that is fair and takes answers from the later iterations

On the 20-task dev simulation, ToT-only / ToT+ACO scored:

| Rule | ToT-only | ToT+ACO | Problem |
|---|---|---|---|
| Best of all | 48.6 | 48.6 | ACO invisible |
| Last iteration (v1) | 38.2 | 47.7 | Handicaps ToT-only too much |
| Plain majority vote | 13.3 | 36.3 | ToT-only falls below the baseline |
| **Verified, last 3** | 46.6 | 48.3 | Both far above baseline, a small ACO edge, answers from the late iterations |

### Step 4: the details were fixed on principle

- **Verifier as a filter, not a vote.** Only answers scored 1.0 count. This is close to self-consistency voting (Wang et al., 2023), filtered by the verifier.
- **Earliest slot, not the most frequent.**
  - About 17.5% of test tasks have several different verified answers.
  - The earliest one matches gold in about 59% of them, against about 41% for an arbitrary pick.
  - A frequency vote risked losing up to roughly 3 points. On the real runs it cost only 0.1 to 0.25.
- **Fallback to the best of the whole run,** so a task solved early is not lost.
- **Window of 3 fixed before the test runs.** The other windows and rules are reported next to it.

### Honest caveats to state if asked

- **ACO's late-window gains partly measure the rule.** Rules that read late iterations favour ACO because they handicap ToT-only, and the rule was chosen with that in mind.
- **The window was picked after seeing dev numbers,** so it is slightly tuned to dev.
- **On the headline itself the gain is small:** +1.9 points, significant in 1 of 3 seeds.
- **It matches `best_of_all` almost exactly.** The two submit different answers on 0 to 2 tasks of 400, so the headline choice barely changes the accuracy numbers. What it does is make the ACO comparison look at the iterations where ACO acts.

---

## 2. What is a nonce, and why do the baseline and the sampling control use one?

Project guide, Section 3.8: "Stochastic runs (the sampling control, the baseline seeds) add a nonce so they make real separate calls."

### Background: the cache

Every LLM request is stored in a cache, keyed by a hash of (model, parameters, messages). If the same request comes again, the cache returns the stored answer instead of calling the API. This makes ToT-only and ToT+ACO directly comparable and lets us replay runs for free.

### The problem for stochastic runs

Some runs need a fresh answer each time:

- **Baseline, seeds 0, 1, 2:** the same prompt is sent for every seed. Without anything extra, seeds 1 and 2 would hit the cache entry from seed 0 and return the identical answer. The three seeds would be one call copied three times, and the SD would be meaningless.
- **Sampling control:** the same prompt is sent 48 times per task at temperature 0.7. With the cache, calls 2 to 48 would return call 1's answer, giving 1 sample instead of 48.

### The fix: a nonce

A nonce ("number used once") is an extra value, such as the seed or the sample number, added to the cache key. The prompt the model sees is unchanged, but the key differs:

| Request | Key includes | Result |
|---|---|---|
| Baseline seed 0 | prompt + nonce 0 | real API call |
| Baseline seed 1 | prompt + nonce 1 | real API call |
| Sampling sample #7 | prompt + nonce 7 | real API call |

A different key means a cache miss, so a real separate call is made and its answer is stored under that key. Rerunning the same seed reuses the same nonce, so it still replays exactly from the cache.

### Why the swarm does not need one

In ToT runs, identical requests should share answers (same path prefix, same prompt, temperature 0). That sharing is intended, so no nonce is added.

### In one line

The nonce tells the cache "treat this look-alike request as a new one", so repeated runs and samples are really independent, while each is still replayable.

---

## 3. Why did we keep the fixed verifier as the quality signal, and why is its strength a limitation?

Project guide, Section 6, "Quality signal" decision. Reason given: "No time for a learned reward model; a feasibility-only checker would likely look like a regression. Its strength is a limitation we state."

### The decision

The quality signal tells the swarm whether an answer is good or bad. We use the fixed deterministic verifier: ordinary code that checks constraints and preferences and gives a score.

### Part 1: no time for a learned reward model

A learned reward model is a second neural network trained to score answers. It would be more "AI-native" and less like a hand-written solver, but training and validating one needs time and data we do not have.

### Part 2: a feasibility-only checker would look like a regression

A weaker alternative only asks "is this slot legal?" (nobody busy, inside work hours) and ignores preferences and "earliest". That is less of a solver, but a noisier signal: it would accept many non-gold answers, so accuracy would likely fall below v1's. A regression (a step backwards) unrelated to ACO was not acceptable, so we kept the full verifier.

### Part 3: its strength is a limitation

Used alone as a solver, the verifier gets 92.75%. That helps accuracy but causes two problems:

- **The verifier does much of the work.** The SLMs propose and the verifier judges almost perfectly, so the system is verifier-guided generation, not pure LLM reasoning.
- **It is a weak test of ACO.** With a near-perfect reward, a good answer is often found before the pheromone has learned anything (project guide Section 10.4), so ACO has little room to show an accuracy gain.

We state this openly, report the verifier-as-solver row, and name a learned reward model or a feasibility-only checker as future work.

### In one line

We kept the strong verifier because the alternatives were too costly or would look like a step backwards, and we admit that this strength makes the results less purely about the LLM swarm and ACO.
