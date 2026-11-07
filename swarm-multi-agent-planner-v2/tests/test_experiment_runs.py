"""The Phase 4 runners (baseline, sampling control, verifier-as-solver) and the offline pipeline
(derive every selection rule from the logs, aggregate over seeds), all with a fake LLM."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import swarm_planner.solver as solver
from fakes import FakeClient
from swarm_planner.baseline import run_baseline, run_sampling_control, run_verifier_solver
from swarm_planner.derive import derive_run
from swarm_planner.logging import read_events
from swarm_planner.metrics import solve_rate
from swarm_planner.selection_rules import SELECTION_RULES

TASK_TEXT = """TASK: You need to schedule a meeting for Alice and Bob for half an hour between the work hours of 9:00 to 17:00 on Monday.

Here are the existing schedules for everyone during the day:
Alice has meetings on Monday during 9:00 to 9:30;
Bob is free the entire day;

You would like to schedule the meeting at their earliest availability. Find a time that works for everyone's schedule and constraints.
SOLUTION: """
GOLD = "Here is the proposed time: Monday, 9:30 - 10:00"
OPTIONS = [  # verified, valid-but-not-earliest, invalid (Alice is busy), unparseable
    "Here is the proposed time: Monday, 9:30 - 10:00",
    "Here is the proposed time: Monday, 10:00 - 10:30",
    "Here is the proposed time: Monday, 9:00 - 9:30",
    "I cannot decide.",
]


def _prompt(i: int) -> str:
    """A distinct prompt per task (identical prompts would legitimately share cached answers)."""
    return f"You are an expert at scheduling meetings. This is case number {i}.\n\n{TASK_TEXT}"


def _dataset(n_tasks: int = 2) -> dict:
    return {
        f"calendar_scheduling_example_{i}": {
            "num_people": "2",
            "num_days": "1",
            "duration": "0.5",
            "prompt_5shot": _prompt(i),
            "prompt_0shot": _prompt(i),
            "golden_plan": GOLD,
        }
        for i in range(n_tasks)
    }


def _write(tmp_path: Path, n_tasks: int = 2) -> str:
    path = tmp_path / "data.json"
    path.write_text(json.dumps(_dataset(n_tasks)), encoding="utf-8")
    return str(path)


def varied_final_answers(messages):  # type: ignore[no-untyped-def]
    """Intermediate calls answer 'intermediate'; every final call picks one of OPTIONS by a hash of
    its request, so different paths give different answers (deterministic)."""
    if "Produce one final answer" in messages[0]["content"]:
        digest = hashlib.sha256(json.dumps(messages).encode()).digest()
        return OPTIONS[digest[0] % len(OPTIONS)]
    return "intermediate"


# --- baseline --------------------------------------------------------------------------------


def test_baseline_makes_one_call_per_task_and_writes_the_run_directory(tmp_path: Path):
    client = FakeClient(lambda messages: GOLD)
    summary = run_baseline(
        _write(tmp_path), str(tmp_path / "run"), "dummy-model", seed=0, client=client,
        cache_path=str(tmp_path / "c.sqlite"),
    )
    assert len(client.calls) == 2
    assert [m["role"] for m in client.calls[0]["messages"]] == ["user"]  # the 5-shot prompt alone
    assert client.calls[0]["temperature"] == 0.0
    for name in ("predictions.json", "events.jsonl.gz", "summary.json", "config.resolved.json"):
        assert (tmp_path / "run" / name).exists()
    preds = json.loads((tmp_path / "run" / "predictions.json").read_text(encoding="utf-8"))
    assert all(item["pred_5shot_pro"] == GOLD for item in preds.values())
    assert summary["verified_share"] == 1.0 and summary["tasks"] == 2


def test_baseline_seed_is_a_cache_nonce_by_default(tmp_path: Path):
    """D4: each seed is a real call. Without the nonce a second seed replays the cached answer."""
    data, cache = _write(tmp_path), str(tmp_path / "c.sqlite")
    c0, c1 = FakeClient(lambda m: GOLD), FakeClient(lambda m: GOLD)
    run_baseline(data, str(tmp_path / "s0"), "dummy-model", seed=0, client=c0, cache_path=cache)
    run_baseline(data, str(tmp_path / "s1"), "dummy-model", seed=1, client=c1, cache_path=cache)
    assert (len(c0.calls), len(c1.calls)) == (2, 2)

    c2, c3 = FakeClient(lambda m: GOLD), FakeClient(lambda m: GOLD)
    cache2 = str(tmp_path / "c2.sqlite")
    run_baseline(data, str(tmp_path / "n0"), "dummy-model", seed=0, seed_as_nonce=False, client=c2, cache_path=cache2)
    run_baseline(data, str(tmp_path / "n1"), "dummy-model", seed=1, seed_as_nonce=False, client=c3, cache_path=cache2)
    assert (len(c2.calls), len(c3.calls)) == (2, 0)


# --- sampling control ------------------------------------------------------------------------


def test_sampling_control_draws_n_samples_at_temperature_07_with_distinct_nonces(tmp_path: Path):
    counter = {"n": 0}

    def responder(messages):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        return OPTIONS[counter["n"] % len(OPTIONS)]

    client = FakeClient(responder)
    summary = run_sampling_control(
        _write(tmp_path, 2), str(tmp_path / "run"), "dummy-model", seed=0, samples=5,
        client=client, cache_path=str(tmp_path / "c.sqlite"),
    )
    assert len(client.calls) == 10  # 2 tasks x 5 samples: identical prompts did not collapse
    assert all(call["temperature"] == 0.7 for call in client.calls)
    events = list(read_events(tmp_path / "run" / "events.jsonl.gz"))
    sample_events = [e for e in events if e["event"] == "samples"]
    assert len(sample_events) == 2 and all(len(e["samples"]) == 5 for e in sample_events)
    preds = json.loads((tmp_path / "run" / "predictions.json").read_text(encoding="utf-8"))
    # 5 samples cycle through the 4 options, so a verified answer is among them: it is submitted
    assert all(item["pred_5shot_pro"] == GOLD for item in preds.values())
    assert summary["selected_verified_share"] == 1.0
    config = json.loads((tmp_path / "run" / "config.resolved.json").read_text(encoding="utf-8"))
    assert config["temperature"] == 0.7 and config["llm_params"]["temperature"] == 0.7


def test_sampling_control_seeds_give_independent_samples(tmp_path: Path):
    data, cache = _write(tmp_path, 1), str(tmp_path / "c.sqlite")
    clients = []
    for seed in (0, 1):
        client = FakeClient(lambda m: OPTIONS[0])
        run_sampling_control(data, str(tmp_path / f"s{seed}"), "dummy-model", seed=seed, samples=3,
                             client=client, cache_path=cache)
        clients.append(len(client.calls))
    assert clients == [3, 3]  # seed 1 does not replay seed 0's samples


# --- verifier as solver ----------------------------------------------------------------------


def test_verifier_solver_answers_with_the_verifiers_best_slot_without_llm_calls(tmp_path: Path):
    summary = run_verifier_solver(_write(tmp_path), str(tmp_path / "run"))
    preds = json.loads((tmp_path / "run" / "predictions.json").read_text(encoding="utf-8"))
    assert all(item["pred_5shot_pro"] == GOLD for item in preds.values())
    assert summary["tasks"] == 2 and summary["tasks_without_valid_slot"] == 0


# --- derive: every rule from the logs ----------------------------------------------------------


def _swarm_run(tmp_path: Path, name: str, *, aco: bool, seed: int, cache: str, n_tasks: int = 3):  # type: ignore[no-untyped-def]
    run_dir = tmp_path / name
    solver.solve_dataset(
        data_path=_write(tmp_path, n_tasks), run_dir=str(run_dir), model="dummy-model",
        agents_per_task=4, iterations_per_task=5, seed=seed, aco_enabled=aco,
        client=FakeClient(varied_final_answers), cache_path=cache,
    )
    return run_dir


def test_derived_headline_equals_the_live_prediction_and_all_rules_are_written(tmp_path: Path):
    run_dir = _swarm_run(tmp_path, "tot_aco_s0", aco=True, seed=0, cache=str(tmp_path / "c.sqlite"))
    report = derive_run(run_dir)

    live = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    derived = json.loads((run_dir / "derived" / "predictions_verified_last3.json").read_text(encoding="utf-8"))
    assert {k: v["pred_5shot_pro"] for k, v in live.items()} == {k: v["pred_5shot_pro"] for k, v in derived.items()}

    for rule in SELECTION_RULES:
        assert (run_dir / "derived" / f"predictions_{rule}.json").exists()
        assert rule in report["rules"]
    # the written file is what the official evaluator would read
    em = solve_rate(
        [v["pred_5shot_pro"] for v in derived.values()], [v["golden_plan"] for v in derived.values()]
    )
    assert em == report["rules"]["verified_last3"]["exact_match"]
    assert report["headline_rule"] == "verified_last3"
    assert report["oracle"]["pass_at_all_traces"] >= report["oracle"]["pass_at_first_iteration"]
    assert len(report["per_iteration"]["verified_trace_share"]) == 5
    assert 0 < report["early_stop"]["share_of_budget"] <= 1.0

    with (run_dir / "derived" / "per_task.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {"task_id", "category", "verified_last3", "verified_last3.verified", "oracle_pass48",
            "early_stop_iterations"} <= set(rows[0])
    assert rows[0]["category"] == "2p1d"


def test_derive_rejects_an_incomplete_log(tmp_path: Path):
    run_dir = _swarm_run(tmp_path, "run", aco=False, seed=0, cache=str(tmp_path / "c.sqlite"), n_tasks=1)
    data = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    data["extra_task"] = dict(next(iter(data.values())))
    (run_dir / "predictions.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="no logged traces"):
        derive_run(run_dir)


# --- the whole pipeline: runs -> derive -> aggregate ---------------------------------------------


def _load_script(name: str):  # type: ignore[no-untyped-def]
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_runs_aggregates_seeds_and_pairs(tmp_path: Path):
    runs = tmp_path / "runs"
    cache = str(tmp_path / "c.sqlite")
    data = _write(tmp_path, 6)
    for seed in (0, 1):
        for aco in (False, True):
            solver.solve_dataset(
                data_path=data, run_dir=str(runs / ("tot_aco" if aco else "tot_only") / f"seed_{seed}"),
                model="dummy-model", agents_per_task=3, iterations_per_task=4, seed=seed, aco_enabled=aco,
                client=FakeClient(varied_final_answers), cache_path=cache,
            )
        run_baseline(data, str(runs / "baseline" / f"seed_{seed}"), "dummy-model", seed=seed,
                     client=FakeClient(lambda m: OPTIONS[2]), cache_path=cache)
        run_sampling_control(data, str(runs / "sampling_control" / f"seed_{seed}"), "dummy-model", seed=seed,
                             samples=4, client=FakeClient(varied_final_answers), cache_path=cache)
    run_verifier_solver(data, str(runs / "verifier_solver"))

    evaluate = _load_script("evaluate_runs")
    from swarm_planner.derive import find_run_dirs

    for run_dir in find_run_dirs(runs):
        derive_run(run_dir)
    loaded = evaluate.load_runs(runs)
    assert set(loaded) == {"baseline", "tot_only", "tot_aco", "sampling_control", "verifier_solver"}
    assert sorted(loaded["tot_aco"]) == [0, 1]

    main = {(r["config"], r["rule"]): r for r in evaluate.main_table(loaded)}
    assert main[("baseline", "single_call")]["exact_match_mean"] == 0.0  # always the invalid slot
    assert main[("baseline", "single_call")]["exact_match_sd"] == 0.0
    assert main[("verifier_solver", "verifier_best_slot")]["exact_match_mean"] == 1.0
    assert main[("tot_aco", "verified_last3")]["seeds"] == 2 and main[("tot_aco", "verified_last3")]["tasks"] == 6
    assert main[("tot_aco", "verified_last3")]["headline"] == 1
    assert ("sampling_control", "majority_vote") in main

    paired = evaluate.paired_table(loaded)
    labels = {r["comparison"] for r in paired}
    assert "ACO vs ToT-only [verified_last3]" in labels
    assert "tot_aco vs baseline [verified_last3]" in labels
    pooled = [r for r in paired if r["seed"] == "pooled"]
    assert pooled and all("diff_ci95_low_pp" in r for r in pooled)
    per_seed = [r for r in paired if r["comparison"] == "ACO vs ToT-only [verified_last3]" and r["seed"] != "pooled"]
    assert [r["seed"] for r in per_seed] == [0, 1] and all(0.0 <= r["mcnemar_p"] <= 1.0 for r in per_seed)

    categories = evaluate.category_table(loaded)
    assert {r["category"] for r in categories} == {"2p1d"}
    report = evaluate.render_report(evaluate.main_table(loaded), categories, paired, loaded)
    assert "PHASE 4 RESULTS" in report and "verified_last3" in report
