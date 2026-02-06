from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class StepRecord:
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float = 0.0
    costs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    steps: List[StepRecord]
    final_reward: float
    final_costs: Dict[str, float]


class Verifier(Protocol):
    def score(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Return verifiable reward and optional diagnostics."""


class Policy(Protocol):
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured action."""


class Environment(Protocol):
    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Reset environment state for a task."""

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action and return next state and status."""
