from __future__ import annotations

import json
from pathlib import Path

import aco_tot.engine as engine
from aco_tot.config import edge_key
from aco_tot.types import AgentTrace, ThoughtResult


def _minimal_prompt(slot_text: str) -> str:
    return (
        "You are an expert at scheduling meetings.\n\n"
        "TASK: You need to schedule a meeting for Alice and Bob for half an hour "
        "between the work hours of 9:00 to 17:00 on Monday.\n"
        f"{slot_text}\n"
        "Find a time that works for everyone's schedule and constraints.\n"
        "SOLUTION: "
    )


def test_train_then_infer_smoke(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    data_path = tmp_path / "input.json"
    out_path = tmp_path / "out.json"
    pheromone_path = tmp_path / "pheromones.json"

    data = {
        "t1": {
            "num_people": "2",
            "num_days": "1",
            "duration": "0.5",
            "prompt_5shot": _minimal_prompt(""),
            "prompt_0shot": "",
            "golden_plan": "Here is the proposed time: Monday, 9:00 - 9:30",
            "pred_5shot_pro": "",
        }
    }
    data_path.write_text(json.dumps(data), encoding="utf-8")

    async def fake_multi_agent(**kwargs):  # type: ignore[no-untyped-def]
        traces = []
        for agent_id in range(8):
            traces.append(
                AgentTrace(
                    agent_id=agent_id,
                    path=[edge_key(1, "constraint_first")],
                    thought_results=[
                        ThoughtResult(
                            level=1,
                            branch_id="constraint_first",
                            branch_name="Constraint-first",
                            thought_prompt="x",
                            intermediate_result="y",
                            selection_probability=0.5,
                        )
                    ],
                    final_answer="Here is the proposed time: Monday, 9:00 - 9:30",
                    path_score=0.5,
                    quality_score=0.0,
                )
            )
        return traces

    monkeypatch.setattr(engine, "_run_multi_agent_for_task", fake_multi_agent)

    summary = engine.train_dataset(
        data_path=str(data_path),
        out_pheromone_path=str(pheromone_path),
        model="dummy-model",
        agents_per_task=8,
        epochs=1,
    )
    assert summary["tasks"] == 1
    assert pheromone_path.exists()

    engine.infer_dataset(
        data_path=str(data_path),
        pheromone_path=str(pheromone_path),
        out_path=str(out_path),
        model="dummy-model",
        agents_per_task=8,
    )
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["t1"]["pred_5shot_pro"] == "Here is the proposed time: Monday, 9:00 - 9:30"


def test_infer_uses_random_tot_when_aco_disabled(
    tmp_path: Path, monkeypatch
):  # type: ignore[no-untyped-def]
    data_path = tmp_path / "input.json"
    out_path = tmp_path / "out.json"
    pheromone_path = tmp_path / "pheromones.json"

    data = {
        "t1": {
            "num_people": "2",
            "num_days": "1",
            "duration": "0.5",
            "prompt_5shot": _minimal_prompt(""),
            "prompt_0shot": "",
            "golden_plan": "Here is the proposed time: Monday, 9:00 - 9:30",
            "pred_5shot_pro": "",
        }
    }
    data_path.write_text(json.dumps(data), encoding="utf-8")
    engine.save_pheromone_state(
        str(pheromone_path), engine.create_initial_pheromone_state()
    )

    observed_use_aco: list[bool] = []

    async def fake_multi_agent(**kwargs):  # type: ignore[no-untyped-def]
        observed_use_aco.append(kwargs["use_aco"])
        return [
            AgentTrace(
                agent_id=0,
                path=[edge_key(1, "constraint_first")],
                thought_results=[
                    ThoughtResult(
                        level=1,
                        branch_id="constraint_first",
                        branch_name="Constraint-first",
                        thought_prompt="x",
                        intermediate_result="y",
                        selection_probability=1.0 / 3.0,
                    )
                ],
                final_answer="Here is the proposed time: Monday, 9:00 - 9:30",
                path_score=1.0 / 3.0,
                quality_score=0.0,
            )
        ]

    monkeypatch.setattr(engine, "_run_multi_agent_for_task", fake_multi_agent)

    engine.infer_dataset(
        data_path=str(data_path),
        pheromone_path=str(pheromone_path),
        out_path=str(out_path),
        model="dummy-model",
        agents_per_task=1,
        aco_enabled=False,
    )

    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert observed_use_aco == [False]
    assert out["t1"]["pred_5shot_pro"] == "Here is the proposed time: Monday, 9:00 - 9:30"


def test_solve_uses_final_iteration_best_not_global(
    tmp_path: Path, monkeypatch
):  # type: ignore[no-untyped-def]
    data_path = tmp_path / "input.json"
    out_path = tmp_path / "out.json"
    data = {
        "t1": {
            "num_people": "2",
            "num_days": "1",
            "duration": "0.5",
            "prompt_5shot": _minimal_prompt(""),
            "prompt_0shot": _minimal_prompt(""),
            "golden_plan": "Here is the proposed time: Monday, 9:00 - 9:30",
            "pred_5shot_pro": "",
        }
    }
    data_path.write_text(json.dumps(data), encoding="utf-8")

    multi_agent_calls: list[int] = []

    async def fake_multi_agent(**kwargs):  # type: ignore[no-untyped-def]
        iteration_idx = len(multi_agent_calls)
        multi_agent_calls.append(iteration_idx)
        prediction = (
            "Here is the proposed time: Monday, 9:00 - 9:30"
            if iteration_idx == 0
            else "Here is the proposed time: Monday, 10:00 - 10:30"
        )
        return [
            AgentTrace(
                agent_id=0,
                path=[edge_key(1, "constraint_first")],
                thought_results=[],
                final_answer=prediction,
                path_score=1.0,
                quality_score=0.0,
            )
        ]

    class _FakeQuality:
        def __init__(self, score: float, slot_rank: int):
            self.score = score
            self.slot_rank = slot_rank

        def to_dict(self):  # type: ignore[no-untyped-def]
            return {
                "score": self.score,
                "hard_valid": True,
                "soft_satisfaction_ratio": self.score,
                "slot_rank": self.slot_rank,
            }

    class _FakeScorer:
        def score_prediction(self, prediction: str):  # type: ignore[no-untyped-def]
            if "9:00 - 9:30" in prediction:
                return _FakeQuality(1.0, 0)
            return _FakeQuality(0.5, 1)

    monkeypatch.setattr(engine, "_run_multi_agent_for_task", fake_multi_agent)
    monkeypatch.setattr(engine, "build_task_quality_scorer", lambda _: _FakeScorer())

    summary = engine.solve_dataset(
        data_path=str(data_path),
        out_path=str(out_path),
        model="dummy-model",
        agents_per_task=1,
        iterations_per_task=2,
        aco_enabled=True,
    )

    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["selection_policy"] == "final_iteration_best"
    assert out["t1"]["pred_5shot_pro"] == "Here is the proposed time: Monday, 10:00 - 10:30"
    assert multi_agent_calls == [0, 1]


def test_solve_random_tot_when_aco_disabled(
    tmp_path: Path, monkeypatch
):  # type: ignore[no-untyped-def]
    data_path = tmp_path / "input.json"
    out_path = tmp_path / "out.json"
    data = {
        "t1": {
            "num_people": "2",
            "num_days": "1",
            "duration": "0.5",
            "prompt_5shot": _minimal_prompt(""),
            "prompt_0shot": _minimal_prompt(""),
            "golden_plan": "Here is the proposed time: Monday, 9:00 - 9:30",
            "pred_5shot_pro": "",
        }
    }
    data_path.write_text(json.dumps(data), encoding="utf-8")

    observed_use_aco: list[bool] = []
    update_calls: list[int] = []

    async def fake_multi_agent(**kwargs):  # type: ignore[no-untyped-def]
        observed_use_aco.append(kwargs["use_aco"])
        return [
            AgentTrace(
                agent_id=0,
                path=[edge_key(1, "constraint_first")],
                thought_results=[],
                final_answer="Here is the proposed time: Monday, 9:00 - 9:30",
                path_score=1.0 / 3.0,
                quality_score=0.0,
            )
        ]

    class _FakeQuality:
        def __init__(self, score: float):
            self.score = score

        def to_dict(self):  # type: ignore[no-untyped-def]
            return {
                "score": self.score,
                "hard_valid": True,
                "soft_satisfaction_ratio": self.score,
                "slot_rank": 0,
            }

    class _FakeScorer:
        def score_prediction(self, _: str):  # type: ignore[no-untyped-def]
            return _FakeQuality(1.0)

    def fake_update(**kwargs):  # type: ignore[no-untyped-def]
        update_calls.append(1)
        return kwargs["pheromone_state"]

    monkeypatch.setattr(engine, "_run_multi_agent_for_task", fake_multi_agent)
    monkeypatch.setattr(engine, "build_task_quality_scorer", lambda _: _FakeScorer())
    monkeypatch.setattr(engine, "apply_pheromone_update", fake_update)

    summary = engine.solve_dataset(
        data_path=str(data_path),
        out_path=str(out_path),
        model="dummy-model",
        agents_per_task=1,
        iterations_per_task=3,
        aco_enabled=False,
    )

    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert observed_use_aco == [False, False, False]
    assert update_calls == []
    assert summary["pheromone_updates_applied"] == 0
    assert out["t1"]["pred_5shot_pro"] == "Here is the proposed time: Monday, 9:00 - 9:30"
