from __future__ import annotations

import json
from pathlib import Path

import swarm_planner.solver as solver
from fakes import FakeClient
from swarm_planner.logging import read_events
from swarm_planner.tot.levels import edge_key
from swarm_planner.types import AgentTrace

CANDIDATE = "Here is the proposed time: Monday, 9:30 - 10:00"

TASK_TEXT = """TASK: You need to schedule a meeting for Alice and Bob for half an hour between the work hours of 9:00 to 17:00 on Monday.

Here are the existing schedules for everyone during the day:
Alice has meetings on Monday during 9:00 to 9:30;
Bob is free the entire day;

You would like to schedule the meeting at their earliest availability. Find a time that works for everyone's schedule and constraints.
SOLUTION: """

A9 = "Here is the proposed time: Monday, 9:00 - 9:30"
A10 = "Here is the proposed time: Monday, 10:00 - 10:30"
A11 = "Here is the proposed time: Monday, 11:00 - 11:30"


def _minimal_prompt(slot_text: str) -> str:
    return (
        "You are an expert at scheduling meetings.\n\n"
        "TASK: You need to schedule a meeting for Alice and Bob for half an hour "
        "between the work hours of 9:00 to 17:00 on Monday.\n"
        f"{slot_text}\n"
        "Find a time that works for everyone's schedule and constraints.\n"
        "SOLUTION: "
    )


def _dataset(prompt: str, n_tasks: int = 1) -> dict:
    return {
        f"t{i}": {
            "num_people": "2",
            "num_days": "1",
            "duration": "0.5",
            "prompt_5shot": prompt,
            "prompt_0shot": prompt,
            "golden_plan": "Here is the proposed time: Monday, 9:30 - 10:00",
        }
        for i in range(1, n_tasks + 1)
    }


def _fake_scorer(monkeypatch):  # type: ignore[no-untyped-def]
    """A scorer by proposed start: 9:00 -> 1.0 (rank 0), 10:00 -> 1.0 (rank 1), else 0.75 (valid)."""

    class _Quality:
        def __init__(self, score: float, slot_rank: int):
            self.score = score
            self.slot_rank = slot_rank

        def to_dict(self):  # type: ignore[no-untyped-def]
            return {
                "score": self.score,
                "candidate_parsed": True,
                "hard_valid": True,
                "soft_satisfied": 1 if self.score == 1.0 else 0,
                "soft_total": 1,
                "slot_rank": self.slot_rank,
            }

    class _Scorer:
        def score_prediction(self, prediction: str):  # type: ignore[no-untyped-def]
            if "9:00 - 9:30" in prediction:
                return _Quality(1.0, 0)
            if "10:00 - 10:30" in prediction:
                return _Quality(1.0, 1)
            return _Quality(0.75, 2)

    monkeypatch.setattr(solver, "build_task_quality_scorer", lambda _: _Scorer())


def _run_with_answers(tmp_path, monkeypatch, answers, policy=None):  # type: ignore[no-untyped-def]
    """One task and one agent per iteration; ``answers[i]`` is the answer of iteration i."""
    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps(_dataset(_minimal_prompt(""))), encoding="utf-8")
    calls: list[int] = []

    async def fake_multi_agent(**kwargs):  # type: ignore[no-untyped-def]
        it = kwargs["iteration"]
        calls.append(it)
        return [
            AgentTrace(
                agent_id=0,
                path=[edge_key(1, "constraint_first")],
                thought_results=[],
                final_answer=answers[it],
                path_score=1.0,
                quality_score=0.0,
            )
        ]

    monkeypatch.setattr(solver, "_run_multi_agent_for_task", fake_multi_agent)
    _fake_scorer(monkeypatch)
    kwargs = {} if policy is None else {"selection_policy": policy}
    summary = solver.solve_dataset(
        data_path=str(data_path),
        run_dir=str(tmp_path / "run"),
        model="dummy-model",
        agents_per_task=1,
        iterations_per_task=len(answers),
        aco_enabled=True,
        **kwargs,
    )
    out = json.loads((tmp_path / "run" / "predictions.json").read_text(encoding="utf-8"))
    return summary, out["t1"]["pred_5shot_pro"], calls


def test_headline_rule_is_verified_last3_and_uses_the_whole_budget(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    # 9:00 is found in iteration 1, but a verified answer exists in the last 3 iterations
    # (10:00 in iteration 5), so that one is chosen. The search never stops early.
    summary, prediction, calls = _run_with_answers(tmp_path, monkeypatch, [A9, A11, A11, A11, A10, A11])
    assert summary["selection_policy"] == "verified_last3"
    assert calls == [0, 1, 2, 3, 4, 5]
    assert prediction == A10


def test_headline_rule_falls_back_to_the_best_trace_of_the_whole_run(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    _, prediction, _ = _run_with_answers(tmp_path, monkeypatch, [A9, A11, A11, A11, A11, A11])
    assert prediction == A9  # no verified trace in iterations 4-6


def test_selection_policy_can_be_changed(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    answers = [A9, A11, A11, A11, A10, A11]
    _, last, _ = _run_with_answers(tmp_path, monkeypatch, answers, policy="last_iteration")
    assert last == A11  # the v1 rule: the last iteration only
    _, best, _ = _run_with_answers(tmp_path, monkeypatch, answers, policy="best_of_all")
    assert best == A9


def test_unknown_selection_policy_is_rejected(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    import pytest

    with pytest.raises(ValueError):
        _run_with_answers(tmp_path, monkeypatch, [A9], policy="nope")


def test_solve_random_tot_when_aco_disabled(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    data_path = tmp_path / "input.json"
    out_path = tmp_path / "run" / "predictions.json"
    data_path.write_text(json.dumps(_dataset(_minimal_prompt(""))), encoding="utf-8")

    observed_use_aco: list[bool] = []
    update_calls: list[int] = []

    async def fake_multi_agent(**kwargs):  # type: ignore[no-untyped-def]
        observed_use_aco.append(kwargs["use_aco"])
        return [
            AgentTrace(
                agent_id=0,
                path=[edge_key(1, "constraint_first")],
                thought_results=[],
                final_answer=A9,
                path_score=1.0 / 3.0,
                quality_score=0.0,
            )
        ]

    def fake_update(**kwargs):  # type: ignore[no-untyped-def]
        update_calls.append(1)
        return kwargs["pheromone_state"]

    monkeypatch.setattr(solver, "_run_multi_agent_for_task", fake_multi_agent)
    _fake_scorer(monkeypatch)
    monkeypatch.setattr(solver, "apply_pheromone_update", fake_update)

    summary = solver.solve_dataset(
        data_path=str(data_path),
        run_dir=str(out_path.parent),
        model="dummy-model",
        agents_per_task=1,
        iterations_per_task=3,
        aco_enabled=False,
    )

    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert observed_use_aco == [False, False, False]
    assert update_calls == []
    assert summary["pheromone_updates_applied"] == 0
    assert out["t1"]["pred_5shot_pro"] == A9


def test_run_seed_is_not_shifted_by_task_position(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    """v1 passed seed + task_offset to each task; v2 passes the run seed and the task id."""
    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps(_dataset(_minimal_prompt(""), n_tasks=3)), encoding="utf-8")
    observed: list[tuple[int, str]] = []

    async def fake_multi_agent(**kwargs):  # type: ignore[no-untyped-def]
        observed.append((kwargs["seed"], kwargs["task_id"]))
        return []

    class _FakeScorer:
        def score_prediction(self, _: str):  # type: ignore[no-untyped-def]
            raise AssertionError("no traces to score")

    monkeypatch.setattr(solver, "_run_multi_agent_for_task", fake_multi_agent)
    monkeypatch.setattr(solver, "build_task_quality_scorer", lambda _: _FakeScorer())

    solver.solve_dataset(
        data_path=str(data_path),
        run_dir=str(tmp_path / "run"),
        model="dummy-model",
        agents_per_task=1,
        iterations_per_task=1,
        seed=7,
    )
    assert observed == [(7, "t1"), (7, "t2"), (7, "t3")]


def test_end_to_end_solve_with_fake_llm_runs_on_clean_import(tmp_path: Path):
    """Real graph, real verifier, fake LLM: the solve path must run and score correctly."""
    data_path = tmp_path / "input.json"
    run_dir = tmp_path / "run"
    data_path.write_text(json.dumps(_dataset(TASK_TEXT)), encoding="utf-8")

    summary = solver.solve_dataset(
        data_path=str(data_path),
        run_dir=str(run_dir),
        model="dummy-model",
        agents_per_task=2,
        iterations_per_task=2,
        seed=0,
        aco_enabled=True,
        client=FakeClient(),
    )

    out = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    assert out["t1"]["pred_5shot_pro"] == CANDIDATE
    assert out["t1"]["aco_tot_selected_slot_rank"] == 0
    assert summary["tasks"] == 1
    assert summary["final_mean_quality"] == 1.0
    assert summary["pheromone_updates_applied"] == 2
    assert Path(summary["run_log_path"]).exists()


def test_live_run_deposits_for_verified_answers_and_never_stops_early(tmp_path: Path):
    """Real graph and verifier. Every fake answer is the gold slot (score 1.0), so every trace
    deposits, and all iterations still run although the first one already succeeded."""
    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps(_dataset(TASK_TEXT)), encoding="utf-8")
    solver.solve_dataset(
        data_path=str(data_path),
        run_dir=str(tmp_path / "run"),
        model="dummy-model",
        agents_per_task=2,
        iterations_per_task=3,
        seed=1,
        aco_enabled=True,
        client=FakeClient(),
    )
    events = list(read_events(tmp_path / "run" / "events.jsonl.gz"))
    iterations = [e for e in events if e["event"] == "iteration"]
    assert [e["it"] for e in iterations] == [0, 1, 2]
    first = iterations[0]
    assert first["tau_before"] == [1.0] * 15
    # 2 verified traces deposit 1 on each of their 5 edges; every edge evaporates to 0.9
    assert abs((sum(first["tau_after"]) - 0.9 * 15) - 2 * 5) < 1e-6
    for trace in first["traces"]:
        assert trace["q"] == 1.0 and trace["v"]["parsed"] is True

    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection_policy"] == "verified_last3"
    assert summary["selected_verified_share"] == 1.0


def test_tot_only_run_has_no_pheromone(tmp_path: Path):
    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps(_dataset(TASK_TEXT)), encoding="utf-8")
    solver.solve_dataset(
        data_path=str(data_path),
        run_dir=str(tmp_path / "run"),
        model="dummy-model",
        agents_per_task=2,
        iterations_per_task=2,
        seed=1,
        aco_enabled=False,
        client=FakeClient(),
    )
    iterations = [
        e for e in read_events(tmp_path / "run" / "events.jsonl.gz") if e["event"] == "iteration"
    ]
    assert len(iterations) == 2
    assert all(e["tau_before"] is None and e["tau_after"] is None for e in iterations)


def test_cost_cap_stops_the_run_and_a_rerun_continues_from_the_cache(tmp_path: Path):
    import pytest

    from swarm_planner.llm.gateway import CostCapExceeded

    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps(_dataset(TASK_TEXT, n_tasks=3)), encoding="utf-8")
    kwargs = dict(
        data_path=str(data_path),
        model="dummy-model",
        agents_per_task=1,
        iterations_per_task=1,
        seed=0,
        aco_enabled=False,
        cache_path=str(tmp_path / "cache.sqlite"),
    )
    capped = FakeClient(cost=0.01)
    with pytest.raises(CostCapExceeded):
        solver.solve_dataset(run_dir=str(tmp_path / "r1"), client=capped, max_cost_usd=0.02, **kwargs)
    assert 2 <= len(capped.calls) < 18  # stopped well before 3 tasks x 6 calls

    resumed = FakeClient(cost=0.01)
    summary = solver.solve_dataset(run_dir=str(tmp_path / "r2"), client=resumed, **kwargs)
    assert summary["tasks"] == 3
    assert len(resumed.calls) < 18 - len(capped.calls) + 1  # the first run's answers were reused
