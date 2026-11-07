"""Typed models used across the ACO-ToT calendar framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskFeatures:
    task_id: str
    num_people: int
    num_days: int
    duration_hours: float
    has_preference: bool
    has_earliest_preference: bool


@dataclass
class ThoughtResult:
    level: int
    branch_id: str
    branch_name: str
    thought_prompt: str
    intermediate_result: str
    selection_probability: float
    # Probability of every branch at this level when the choice was made
    # (uniform 1/3 each for ToT-only runs), in branch order.
    probabilities: Dict[str, float] = field(default_factory=dict)
    # Short hash of the LLM request; the prompt and response are in the LLM cache.
    request_hash: str = ""


@dataclass
class AgentTrace:
    agent_id: int
    path: List[str]
    thought_results: List[ThoughtResult]
    final_answer: str
    path_score: float
    quality_score: float
    quality_details: Dict[str, Any] | None = None
    final_request_hash: str = ""

    def request_hashes(self) -> List[str]:
        """Request hashes of the 5 level calls and the finalize call, in order."""
        return [item.request_hash for item in self.thought_results] + [self.final_request_hash]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "path": self.path,
            "thought_results": [asdict(item) for item in self.thought_results],
            "final_answer": self.final_answer,
            "path_score": self.path_score,
            "quality_score": self.quality_score,
            "quality_details": self.quality_details or {},
            "final_request_hash": self.final_request_hash,
        }


@dataclass
class PheromoneState:
    metadata: Dict[str, Any]
    pheromones: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "pheromones": self.pheromones,
        }


@dataclass
class RunResult:
    prediction: str
    traces: List[AgentTrace]
    pheromone_state: PheromoneState | None = None
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prediction": self.prediction,
            "traces": [trace.to_dict() for trace in self.traces],
        }
        if self.pheromone_state is not None:
            payload["pheromone_state"] = self.pheromone_state.to_dict()
        if self.metadata is not None:
            payload["metadata"] = self.metadata
        return payload
