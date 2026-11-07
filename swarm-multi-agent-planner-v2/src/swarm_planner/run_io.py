"""What every run writes into its run directory, shared by the swarm, baseline and sampling runners.

    predictions.json      the dataset with the prediction in ``pred_5shot_pro`` (official evaluator format)
    events.jsonl.gz       run_start, iteration/samples, llm_call, task_result, run_end events
    summary.json          totals, LLM calls, cache hits, cost
    config.resolved.json  the full resolved configuration
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from .llm.cache import LLMCache
from .llm.gateway import LLMGateway
from .logging import EVENTS_FILENAME, EventLog
from .runinfo import collect_environment, file_sha256

PREDICTIONS_FILENAME = "predictions.json"
SUMMARY_FILENAME = "summary.json"
CONFIG_FILENAME = "config.resolved.json"


def load_dataset(data_path: str, limit: int | None) -> Dict[str, Dict[str, Any]]:
    with Path(data_path).open("r", encoding="utf-8") as handle:
        data: Dict[str, Dict[str, Any]] = json.load(handle)
    if limit is not None:
        data = dict(list(data.items())[: max(0, int(limit))])
    return data


class RunContext:
    """Run directory, event log, cache and gateway of one run."""

    def __init__(
        self,
        *,
        run_dir: str,
        model: str,
        cache_path: str | None,
        max_concurrent_requests: int,
        max_cost_usd: float | None,
        temperature: float | None = None,
        client: Any = None,
    ) -> None:
        self.started = time.perf_counter()
        self.run_path = Path(run_dir)
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_path / EVENTS_FILENAME
        self.predictions_path = self.run_path / PREDICTIONS_FILENAME
        self.event_log = EventLog(self.events_path)
        self.cache = LLMCache(cache_path) if cache_path else None
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        self.gateway = LLMGateway(
            model=model,
            cache=self.cache,
            client=client,
            max_concurrent_requests=max_concurrent_requests,
            event_sink=self.event_log.log_row,
            max_cost_usd=max_cost_usd,
            **kwargs,
        )

    def start(self, config: Dict[str, Any], data_path: str, extra: Dict[str, Any] | None = None) -> None:
        config = {
            **config,
            "data_path": data_path,
            "data_sha256": file_sha256(data_path),
            "llm_params": self.gateway.params,
            "provider_preferences": self.gateway.provider_preferences(),
            "max_cost_usd": self.gateway.max_cost_usd,
        }
        (self.run_path / CONFIG_FILENAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.event_log.log(
            "run_start",
            {"config": config, **(extra or {}), "environment": collect_environment()},
            flush=True,
        )

    def progress(self, tag: str, done: int, total: int, detail: str = "") -> None:
        """One progress line: tasks done, elapsed time, estimated time left and the money spent."""
        elapsed = time.perf_counter() - self.started
        eta = elapsed / done * (total - done) if done else 0.0
        stats = self.gateway.stats
        cap = f"/${self.gateway.max_cost_usd:.2f}" if self.gateway.max_cost_usd is not None else ""
        print(
            f"[{tag}] {done}/{total} elapsed {elapsed / 60:.1f}m eta {eta / 60:.1f}m "
            f"paid ${stats.cost_usd_paid:.3f}{cap} calls {stats.calls} (hits {stats.cache_hits}) {detail}",
            flush=True,
        )

    def finish(self, data: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Write run_end, predictions.json and summary.json, then close everything."""
        wall = round(time.perf_counter() - self.started, 3)
        run_end = {
            "tasks_processed": summary.get("tasks"),
            "llm": self.gateway.stats.to_dict(),
            "cache_entries": len(self.cache) if self.cache is not None else None,
            "wall_s": wall,
        }
        self.event_log.log("run_end", run_end, flush=True)
        self.close()
        with self.predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        summary = {
            **summary,
            "llm": run_end["llm"],
            "cache_entries": run_end["cache_entries"],
            "wall_s": wall,
            "out_path": str(self.predictions_path),
            "run_log_path": str(self.events_path),
            "run_dir": str(self.run_path),
        }
        (self.run_path / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        llm = run_end["llm"]
        print(
            f"LLM calls={llm['llm_calls']} cache_hits={llm['cache_hits']} "
            f"paid=${llm['cost_usd_paid']:.4f} uncached_equivalent=${llm['cost_usd_uncached_equivalent']:.4f}"
        )
        print(f"Run directory: {self.run_path}")
        return summary

    def close(self) -> None:
        self.event_log.close()
        if self.cache is not None:
            self.cache.close()
            self.cache = None
