"""Configuration for the fixed ACO-ToT calendar scheduling framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 2.0
DEFAULT_EPSILON = 0.001
DEFAULT_RHO = 0.1
DEFAULT_AGENTS_PER_TASK = 8
DEFAULT_INITIAL_PHEROMONE = 1.0
DEFAULT_SEED = 42
PHEROMONE_VERSION = "1.0"

FINAL_INSTRUCTION = (
    "Find a time that works for everyone's schedule and constraints."
)


@dataclass(frozen=True)
class BranchConfig:
    branch_id: str
    name: str
    thought_prompt: str


@dataclass(frozen=True)
class LevelConfig:
    level: int
    name: str
    goal: str
    branches: List[BranchConfig]


LEVEL_CONFIGS: List[LevelConfig] = [
    LevelConfig(
        level=1,
        name="Parse the task",
        goal="understand what must be scheduled",
        branches=[
            BranchConfig(
                branch_id="constraint_first",
                name="Constraint-first",
                thought_prompt=(
                    "Extract duration, days, work hours, and preferences first."
                ),
            ),
            BranchConfig(
                branch_id="people_first",
                name="People-first",
                thought_prompt=(
                    "Extract participants first, then list out the constraints for each participant."
                ),
            ),
            BranchConfig(
                branch_id="preference_first",
                name="Preference-first",
                thought_prompt=(
                    "Identify special preferences first, then parse the remaining constraints."
                ),
            ),
        ],
    ),
    LevelConfig(
        level=2,
        name="Organize calendars",
        goal="convert raw text schedules into usable structure",
        branches=[
            BranchConfig(
                branch_id="participant_wise",
                name="Participant-wise",
                thought_prompt=(
                    "Build busy slots participant by participant."
                ),
            ),
            BranchConfig(
                branch_id="day_wise",
                name="Day-wise",
                thought_prompt=(
                    "List out all unique busy slots for each day, across all participants. Strictly output day-wise, without mentioning participant names."
                ),
            ),
            BranchConfig(
                branch_id="mixed_normalization",
                name="Mixed normalization",
                thought_prompt=(
                    "Normalize first by day, then group by participant."
                ),
            ),
        ],
    ),
    LevelConfig(
        level=3,
        name="Generate feasible availability",
        goal="derive possible free meeting windows",
        branches=[
            BranchConfig(
                branch_id="free_slot_computation",
                name="Free-slot computation",
                thought_prompt=(
                    "Compute each participant's free intervals, then find the common slots where all participants are available."
                ),
            ),
            BranchConfig(
                branch_id="candidate_slot_enumeration",
                name="Candidate-slot enumeration",
                thought_prompt=(
                    "List all possible meeting windows within work hours, then remove those that violate duration, availability, or preference constraints. Add very short explanation."
                ),
            ),
            BranchConfig(
                branch_id="progressive_elimination",
                name="Progressive elimination",
                thought_prompt=(
                    "Start with the full workday as the candidate window, then remove each participant's busy intervals step by step. Add very short explanation."
                ),
            ),
        ],
    ),
    LevelConfig(
        level=4,
        name="Apply preferences / selection policy",
        goal="choose among valid candidates",
        branches=[
            BranchConfig(
                branch_id="earliest_valid",
                name="Earliest-valid",
                thought_prompt=(
                    "If the task explicitly asks for the earliest time, choose the earliest valid candidate; otherwise select any candidate that satisfies all hard constraints."
                ),
            ),
            BranchConfig(
                branch_id="preference_priority",
                name="Preference-priority",
                thought_prompt=(
                    "Apply all explicit preference constraints first, then choose a valid candidate that best satisfies them while still meeting all hard constraints."
                ),
            ),
            BranchConfig(
                branch_id="conservative_selection",
                name="Conservative selection",
                thought_prompt=(
                    "Prefer a clearly safe valid candidate with buffer from adjacent busy intervals, and avoid edge-touching slots unless necessary."
                ),
            ),
        ],
    ),
    LevelConfig(
        level=5,
        name="Final answer and verification",
        goal="produce properly formatted output",
        branches=[
            BranchConfig(
                branch_id="direct_output",
                name="Direct output",
                thought_prompt=(
                    "Return the best slot immediately."
                ),
            ),
            BranchConfig(
                branch_id="one_step_self_check",
                name="One-step self-check",
                thought_prompt=(
                    "Verify the chosen slot against all calendars once, then return."
                ),
            ),
            BranchConfig(
                branch_id="format_check_output",
                name="Format-check output",
                thought_prompt=(
                    "Verify day, time, duration, and output format before return."
                ),
            ),
        ],
    ),
]


LEAF_BEHAVIOR_PROMPTS: Dict[str, str] = {
    "direct_output": "Return the best slot immediately.",
    "one_step_self_check": (
        "Verify against all participant calendars once and then return the final answer."
    ),
    "format_check_output": (
        "Verify day, start time, end time, duration, and final output format before returning the final answer."
    ),
}


def get_level_config(level: int) -> LevelConfig:
    for config in LEVEL_CONFIGS:
        if config.level == level:
            return config
    raise ValueError(f"Unknown level: {level}")


def edge_key(level: int, branch_id: str) -> str:
    return f"L{level}:{branch_id}"


def all_edge_keys() -> List[str]:
    keys: List[str] = []
    for level_config in LEVEL_CONFIGS:
        for branch in level_config.branches:
            keys.append(edge_key(level_config.level, branch.branch_id))
    return keys
