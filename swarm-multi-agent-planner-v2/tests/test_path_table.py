from __future__ import annotations

import asyncio
import csv
import gzip
import json
from pathlib import Path

from fakes import CANDIDATE, FakeClient
from swarm_planner.aco.pheromone import create_initial_pheromone_state
from swarm_planner.llm.cache import LLMCache
from swarm_planner.llm.gateway import LLMGateway
from swarm_planner.logging import read_events
from swarm_planner.path_table import (
    PATH_COLUMNS,
    calls_per_task,
    enumerate_dataset,
    enumerate_task,
    paths_per_task,
)
from swarm_planner.tot.context import coerce_original_prompt
from swarm_planner.tot.graph import build_agent_graph
from swarm_planner.verifier.parser import answers_match
from test_solver import TASK_TEXT, _dataset

WRONG = "Here is the proposed time: Monday, 9:00 - 9:30"  # Alice is busy then


def _item():
    return _dataset(TASK_TEXT)["t1"]


def _only_people_first_is_right(messages):
    if "Produce one final answer" in messages[0]["content"]:
        return CANDIDATE if "Extract participants first" in messages[1]["content"] else WRONG
    return "intermediate"


def _enumerate(client, cache=None, task_id="t1"):
    gateway = LLMGateway(model="dummy-model", cache=cache, client=client)
    return asyncio.run(enumerate_task(task_id=task_id, item=_item(), gateway=gateway))


# --- shape ---------------------------------------------------------------------------


def test_the_tree_has_243_paths_and_606_calls():
    assert paths_per_task() == 243
    assert calls_per_task() == 363 + 243 == 606


def test_every_path_is_enumerated_with_606_distinct_requests():
    client = FakeClient()
    rows, nodes = _enumerate(client)
    assert len(rows) == 243 and len(nodes) == 606
    assert len(client.calls) == 606
    assert [r["path"] for r in rows][:3] == ["0-0-0-0-0", "0-0-0-0-1", "0-0-0-0-2"]
    assert len({r["path"] for r in rows}) == 243
    assert set(rows[0]) == set(PATH_COLUMNS)
    # branch ids per level and request hashes are filled in
    assert {r["b1"] for r in rows} == {"constraint_first", "people_first", "preference_first"}
    assert all(len(r[f"req{i}"]) == 16 for r in rows for i in range(1, 6))
    # a prefix shared by several paths is one request
    assert len({r["req1"] for r in rows}) == 3
    assert len({r["req2"] for r in rows}) == 9
    assert len({r["req_final"] for r in rows}) == 243
    steps = [n["step"] for n in nodes]
    assert {s: steps.count(s) for s in ("L1", "L2", "L3", "L4", "L5", "final")} == {
        "L1": 3, "L2": 9, "L3": 27, "L4": 81, "L5": 243, "final": 243,
    }


def test_rows_record_the_verifier_result_and_correctness():
    rows, _ = _enumerate(FakeClient(_only_people_first_is_right))
    right = [r for r in rows if r["correct"]]
    assert len(right) == 81 and {r["b1"] for r in right} == {"people_first"}
    assert all(r["score"] == 1.0 and r["hard_valid"] and r["slot_rank"] == 0 for r in right)
    wrong = [r for r in rows if not r["correct"]]
    assert all(not r["hard_valid"] and r["score"] < 0.5 for r in wrong)
    assert all(r["answer"] == WRONG for r in wrong)


def test_answers_match_compares_day_and_times_exactly():
    assert answers_match("Here is the proposed time: Monday, 9:30 - 10:00", CANDIDATE + " ")
    assert answers_match("monday, 09:30 - 10:00", CANDIDATE)
    assert not answers_match("Here is the proposed time: Monday, 9:00 - 9:30", CANDIDATE)
    assert not answers_match("Here is the proposed time: Tuesday, 9:30 - 10:00", CANDIDATE)
    assert not answers_match("no answer", CANDIDATE)
    assert not answers_match(CANDIDATE, "")


# --- the requests are the solver's requests ------------------------------------------


def test_enumerator_requests_are_identical_to_the_solvers(tmp_path):
    """A graph run (what ToT / ToT+ACO do) caches 6 requests; the enumerator must hit them."""
    cache = LLMCache(tmp_path / "c.sqlite")
    item = _item()
    graph_client = FakeClient()
    gateway = LLMGateway(model="dummy-model", cache=cache, client=graph_client)
    state = {
        "original_prompt": coerce_original_prompt(item["prompt_5shot"]),
        "pheromones": dict(create_initial_pheromone_state().pheromones),
        "alpha": 1.0, "epsilon": 0.001, "use_aco": True,
        "seed": 4, "task_id": "t1", "iteration": 0, "agent_id": 2,
        "selected_path": [], "thought_results": [], "path_score": 1.0,
    }
    output = asyncio.run(build_agent_graph(gateway).ainvoke(state))
    graph_hashes = [t["request_hash"] for t in output["thought_results"]] + [output["final_request_hash"]]
    assert len(graph_client.calls) == 6

    enum_client = FakeClient()
    rows, _ = _enumerate(enum_client, cache)
    assert len(enum_client.calls) == 606 - 6  # exactly the graph's 6 requests were cache hits
    path = [i for r in rows for i in [r] if [r[f"req{k}"] for k in range(1, 6)] + [r["req_final"]] == graph_hashes]
    assert len(path) == 1
    cache.close()


# --- the run ---------------------------------------------------------------------------


def _write_data(tmp_path: Path, n_tasks: int) -> Path:
    """Tasks with distinct prompts (identical prompts would share cached answers)."""
    data = _dataset(TASK_TEXT, n_tasks=n_tasks)
    for index, item in enumerate(data.values()):
        item["prompt_5shot"] = item["prompt_0shot"] = f"Variant {index}.\n\n{TASK_TEXT}"
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _run(tmp_path, name, n_tasks=1, client=None, **kwargs):
    client = client or FakeClient()
    summary = enumerate_dataset(
        data_path=str(_write_data(tmp_path, n_tasks)),
        out_dir=str(tmp_path / name),
        model="dummy-model",
        cache_path=kwargs.pop("cache_path", str(tmp_path / "cache.sqlite")),
        client=client,
        **kwargs,
    )
    return tmp_path / name, summary, client


def _table(out: Path):
    with gzip.open(out / "path_table.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_output_files_and_counts(tmp_path):
    out, summary, client = _run(tmp_path, "out", n_tasks=2)
    for name in ("path_table.csv.gz", "nodes.jsonl.gz", "events.jsonl.gz", "summary.json", "config.resolved.json"):
        assert (out / name).exists(), name
    rows = _table(out)
    assert len(rows) == 2 * 243 and list(rows[0]) == PATH_COLUMNS
    assert [r["task_id"] for r in rows][:243] == ["t1"] * 243
    assert rows[0]["correct"] == "True" and rows[0]["score"] == "1.0"

    with gzip.open(out / "nodes.jsonl.gz", "rt", encoding="utf-8") as handle:
        nodes = [json.loads(line) for line in handle]
    assert len(nodes) == 2 * 606
    assert {"task_id", "node", "step", "req", "text", "in", "out", "fin"} == set(nodes[0])

    assert summary["complete"] is True and summary["paths"] == summary["expected_paths"] == 486
    assert json.loads((out / "summary.json").read_text()) == summary


def test_event_log_has_one_llm_call_per_request_and_a_summary_per_task(tmp_path):
    out, _, client = _run(tmp_path, "out", n_tasks=2)
    events = list(read_events(out / "events.jsonl.gz"))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("task_result") == 2
    calls = [e for e in events if e["event"] == "llm_call"]
    assert len(calls) == 2 * 606
    assert {"task_id", "step", "node", "req", "hit", "in", "out", "lat", "cost", "prov", "fin"} <= set(calls[0])
    start = events[0]
    assert start["path_columns"] == PATH_COLUMNS and len(start["edge_order"]) == 15
    assert start["config"]["llm_calls_per_task"] == 606
    result = next(e for e in events if e["event"] == "task_result")
    assert (result["paths"], result["nodes"], result["correct"], result["score_1"]) == (243, 606, 243, 243)
    # nothing to hide: every call has a node id that exists in the nodes file
    with gzip.open(out / "nodes.jsonl.gz", "rt", encoding="utf-8") as handle:
        node_reqs = {json.loads(line)["req"] for line in handle}
    assert {c["req"] for c in calls} == node_reqs


def test_rerun_resumes_from_the_cache_and_gives_the_same_table(tmp_path):
    out1, _, client1 = _run(tmp_path, "one", n_tasks=2)
    out2, summary2, client2 = _run(tmp_path, "two", n_tasks=2)
    assert len(client1.calls) == 2 * 606
    assert len(client2.calls) == 0
    assert summary2["llm"]["cache_hits"] == summary2["llm"]["llm_calls"]
    assert summary2["llm"]["cost_usd_paid"] == 0.0
    assert _table(out1) == _table(out2)


def test_task_concurrency_does_not_change_the_table(tmp_path):
    out1, _, _ = _run(tmp_path, "seq", n_tasks=3, cache_path=None, task_concurrency=1)
    out3, _, _ = _run(tmp_path, "par", n_tasks=3, cache_path=None, task_concurrency=3)
    assert _table(out1) == _table(out3)


def test_cost_cap_stops_new_tasks_and_keeps_what_finished(tmp_path):
    # each task costs 606 x 0.0001 = $0.0606 with the fake client
    out, summary, _ = _run(tmp_path, "capped", n_tasks=3, task_concurrency=1, max_cost_usd=0.01)
    assert summary["complete"] is False
    assert summary["tasks_completed"] == 1 and summary["tasks_skipped_by_cost_cap"] == ["t2", "t3"]
    assert len(_table(out)) == 243
    # a rerun without the cap finishes the work, paying only for what is missing
    out2, summary2, client2 = _run(tmp_path, "full", n_tasks=3, task_concurrency=1)
    assert summary2["complete"] is True and summary2["paths"] == 3 * 243
    assert len(client2.calls) == 2 * 606


def test_a_failing_task_is_logged_stops_the_run_and_keeps_finished_tasks(tmp_path):
    def breaks_on_the_second_task(messages):
        if "Variant 1." in messages[1]["content"]:
            raise ValueError("provider exploded")
        return CANDIDATE if "Produce one final answer" in messages[0]["content"] else "intermediate"

    try:
        _run(tmp_path, "bad", n_tasks=3, task_concurrency=1, client=FakeClient(breaks_on_the_second_task))
    except ValueError:
        pass
    else:
        raise AssertionError("expected the run to fail")

    out = tmp_path / "bad"
    events = list(read_events(out / "events.jsonl.gz"))
    assert [e["event"] for e in events].count("task_error") == 1
    end = events[-1]
    assert end["event"] == "run_end" and end["interrupted"] is True and end["tasks_completed"] == 1
    # the finished task is still written, and the summary says it is partial
    assert len(_table(out)) == 243
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is False and summary["interrupted"] is True
    assert "provider exploded" in summary["error"] and summary["completed_task_ids"] == ["t1"]


# --- task order ----------------------------------------------------------------------


def _categorised_data(tmp_path: Path) -> Path:
    """9 tasks in file order, category by category: t0-t2 (2p,1d), t3-t5 (2p,2d), t6-t8 (3p,1d)."""
    data = {}
    index = 0
    for people, days in ((2, 1), (2, 2), (3, 1)):
        for _ in range(3):
            item = dict(_item())
            item["num_people"], item["num_days"] = str(people), str(days)
            item["prompt_5shot"] = item["prompt_0shot"] = f"Variant {index}.\n\n{TASK_TEXT}"
            data[f"t{index}"] = item
            index += 1
    path = tmp_path / "categorised.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_interleave_by_category_is_round_robin(tmp_path):
    from swarm_planner.path_table import interleave_by_category

    data = json.loads(_categorised_data(tmp_path).read_text())
    assert interleave_by_category(data) == ["t0", "t3", "t6", "t1", "t4", "t7", "t2", "t5", "t8"]


def test_limit_takes_a_balanced_subset_by_default(tmp_path):
    kwargs = dict(
        data_path=str(_categorised_data(tmp_path)),
        model="dummy-model",
        cache_path=None,
        client=FakeClient(),
        limit=6,
    )
    summary = enumerate_dataset(out_dir=str(tmp_path / "balanced"), **kwargs)
    assert summary["completed_task_ids"] == ["t0", "t3", "t6", "t1", "t4", "t7"]  # 2 per category

    summary = enumerate_dataset(out_dir=str(tmp_path / "file"), order="file", **kwargs)
    assert summary["completed_task_ids"] == ["t0", "t1", "t2", "t3", "t4", "t5"]  # 3+3, one category missing
    config = json.loads((tmp_path / "balanced" / "config.resolved.json").read_text())
    assert config["order"] == "interleaved" and config["task_ids"] == summary["completed_task_ids"][:0] + [
        "t0", "t3", "t6", "t1", "t4", "t7",
    ]
