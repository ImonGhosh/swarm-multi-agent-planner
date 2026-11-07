"""The run directory, the event log and cache replay (fake LLM, real graph and verifier)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import swarm_planner.solver as solver
from fakes import CANDIDATE, FakeClient
from swarm_planner.llm.cache import LLMCache
from swarm_planner.logging import EventLog, read_events
from test_solver import TASK_TEXT, _dataset

AGENTS, ITERS = 2, 3
EDGES, LEVELS, BRANCHES = 15, 5, 3


def _run(tmp_path: Path, name: str, *, tasks=1, aco=True, client=None, **kwargs):
    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps(_dataset(TASK_TEXT, n_tasks=tasks)), encoding="utf-8")
    client = client or FakeClient()
    run_dir = tmp_path / name
    summary = solver.solve_dataset(
        data_path=str(data_path),
        run_dir=str(run_dir),
        model="dummy-model",
        agents_per_task=AGENTS,
        iterations_per_task=ITERS,
        seed=3,
        aco_enabled=aco,
        client=client,
        cache_path=kwargs.pop("cache_path", str(tmp_path / "cache.sqlite")),
        **kwargs,
    )
    return run_dir, summary, client


def _events(run_dir: Path):
    return list(read_events(run_dir / "events.jsonl.gz"))


def test_run_directory_has_the_documented_files(tmp_path):
    run_dir, summary, _ = _run(tmp_path, "run")
    for name in ("predictions.json", "events.jsonl.gz", "summary.json", "config.resolved.json"):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "summary.json").read_text()) == summary
    config = json.loads((run_dir / "config.resolved.json").read_text())
    assert config["seed"] == 3 and config["agents_per_task"] == AGENTS
    assert config["provider_preferences"]["only"] == ["DeepInfra"]


def test_event_log_structure(tmp_path):
    run_dir, summary, client = _run(tmp_path, "run")
    events = _events(run_dir)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("iteration") == ITERS
    assert kinds.count("task_result") == 1
    assert kinds.count("llm_call") == AGENTS * ITERS * (LEVELS + 1)

    start = events[0]
    assert len(start["edge_order"]) == EDGES
    assert {k: len(v) for k, v in start["branch_order"].items()} == {str(l): BRANCHES for l in range(1, 6)}
    env = start["environment"]
    assert any(p.lower().startswith("numpy==") for p in env["packages"])
    assert "git_commit" in env and env["python"]
    assert len(start["config"]["data_sha256"]) == 64

    end = events[-1]
    assert end["llm"]["llm_calls"] == summary["llm"]["llm_calls"] == kinds.count("llm_call")
    assert end["llm"]["cache_hits"] + len(client.calls) == end["llm"]["llm_calls"]


def test_iteration_events_hold_pheromone_probabilities_and_verifier_results(tmp_path):
    run_dir, _, _ = _run(tmp_path, "run")
    iterations = [e for e in _events(run_dir) if e["event"] == "iteration"]
    first = iterations[0]
    assert len(first["tau_before"]) == len(first["tau_after"]) == EDGES
    assert all(abs(t - 1.0) < 1e-9 for t in first["tau_before"])  # tau_0 = 1
    assert first["tau_after"] != first["tau_before"]
    # the pheromone of one iteration is the start of the next
    assert iterations[1]["tau_before"] == first["tau_after"]

    assert len(first["traces"]) == AGENTS
    for trace in first["traces"]:
        assert len(trace["p"]) == LEVELS
        for row in trace["p"]:
            assert len(row) == BRANCHES and abs(sum(row) - 1.0) < 1e-4
        assert len(trace["path"]) == LEVELS
        assert len(trace["req"]) == LEVELS + 1 and all(len(h) == 16 for h in trace["req"])
        assert trace["ans"] == CANDIDATE
        assert trace["q"] == 1.0
        assert trace["v"] == {"parsed": True, "valid": True, "hard": 5, "soft": 1, "soft_total": 1, "rank": 0}


def test_tot_only_logs_uniform_probabilities_and_no_pheromone(tmp_path):
    run_dir, _, _ = _run(tmp_path, "run", aco=False)
    iteration = next(e for e in _events(run_dir) if e["event"] == "iteration")
    assert iteration["tau_before"] is None and iteration["tau_after"] is None
    for trace in iteration["traces"]:
        assert all(abs(p - 1 / 3) < 1e-5 for row in trace["p"] for p in row)


def test_logged_hashes_resolve_to_prompts_and_responses_in_the_cache(tmp_path):
    run_dir, _, _ = _run(tmp_path, "run")
    cache = LLMCache(tmp_path / "cache.sqlite")
    events = _events(run_dir)
    hashes = {e["req"] for e in events if e["event"] == "llm_call"}
    from_traces = {h for e in events if e["event"] == "iteration" for t in e["traces"] for h in t["req"]}
    assert from_traces == hashes
    for h in hashes:
        found = cache.find_by_prefix(h)
        assert found is not None and found["request"]["model"] == "dummy-model"
        assert found["request"]["messages"][0]["role"] == "system"
    # a finalize call: its stored request carries the finalize system prompt
    trace = next(e for e in events if e["event"] == "iteration")["traces"][0]
    final = cache.find_by_prefix(trace["req"][-1])
    assert "Produce one final answer" in final["request"]["messages"][0]["content"]
    assert final["text"] == CANDIDATE
    cache.close()


def test_prompt_text_is_not_written_to_the_log(tmp_path):
    run_dir, _, _ = _run(tmp_path, "run")
    raw = gzip.open(run_dir / "events.jsonl.gz", "rt", encoding="utf-8").read()
    assert "You need to schedule a meeting" not in raw
    assert "Latest Thought" not in raw


def test_second_run_replays_from_the_cache_without_provider_calls(tmp_path):
    run_dir_1, summary_1, client_1 = _run(tmp_path, "run1")
    run_dir_2, summary_2, client_2 = _run(tmp_path, "run2")
    assert len(client_1.calls) > 0 and len(client_2.calls) == 0
    assert summary_2["llm"]["cache_hits"] == summary_2["llm"]["llm_calls"]
    assert summary_2["llm"]["cost_usd_paid"] == 0.0
    assert summary_2["llm"]["cost_usd_uncached_equivalent"] == summary_1["llm"]["cost_usd_uncached_equivalent"]
    p1 = json.loads((run_dir_1 / "predictions.json").read_text())
    p2 = json.loads((run_dir_2 / "predictions.json").read_text())
    assert p1 == p2


def test_tot_only_and_aco_share_cached_answers(tmp_path):
    _run(tmp_path, "aco", aco=True)
    _, summary, client = _run(tmp_path, "tot", aco=False)
    # Same questions are answered from the cache; only paths the ACO run never took are new.
    assert summary["llm"]["cache_hits"] > 0
    assert len(client.calls) < summary["llm"]["llm_calls"]


def test_task_concurrency_does_not_change_the_results(tmp_path):
    seq_dir, _, _ = _run(tmp_path, "seq", tasks=3, cache_path=None)
    par_dir, _, _ = _run(tmp_path, "par", tasks=3, task_concurrency=3, cache_path=None)
    seq = json.loads((seq_dir / "predictions.json").read_text())
    par = json.loads((par_dir / "predictions.json").read_text())
    assert list(seq) == list(par) == ["t1", "t2", "t3"]
    assert {k: v["pred_5shot_pro"] for k, v in seq.items()} == {k: v["pred_5shot_pro"] for k, v in par.items()}

    def paths(run_dir):
        return sorted(
            (e["task_id"], e["it"], t["agent"], tuple(t["path"]))
            for e in _events(run_dir)
            if e["event"] == "iteration"
            for t in e["traces"]
        )

    assert paths(seq_dir) == paths(par_dir)


def test_limit_solves_only_the_first_tasks(tmp_path):
    run_dir, summary, _ = _run(tmp_path, "run", tasks=3, limit=2)
    assert summary["tasks"] == 2
    assert list(json.loads((run_dir / "predictions.json").read_text())) == ["t1", "t2"]


def test_failed_task_is_logged_and_stops_the_run(tmp_path):
    def broken(messages):
        raise ValueError("provider exploded")

    with pytest.raises(ValueError):
        _run(tmp_path, "run", client=FakeClient(broken))
    events = _events(tmp_path / "run")
    assert events[0]["event"] == "run_start"
    errors = [e for e in events if e["event"] == "task_error"]
    assert len(errors) == 1 and "provider exploded" in errors[0]["error"]
    assert not (tmp_path / "run" / "predictions.json").exists()


# --- the event log itself -----------------------------------------------------------


def test_event_log_is_readable_after_a_crash(tmp_path):
    path = tmp_path / "events.jsonl.gz"
    log = EventLog(path)
    for i in range(5):
        log.log("task_result", {"i": i}, flush=True)
    # never closed: simulates a killed process
    assert [e["i"] for e in read_events(path)] == [0, 1, 2, 3, 4]
    log.close()


def test_read_events_tolerates_a_truncated_file(tmp_path):
    path = tmp_path / "events.jsonl.gz"
    log = EventLog(path)
    for i in range(50):
        log.log("task_result", {"i": i, "pad": "x" * 200}, flush=True)
    log.close()
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    events = list(read_events(path))
    assert 0 < len(events) < 50
    assert [e["i"] for e in events] == list(range(len(events)))
