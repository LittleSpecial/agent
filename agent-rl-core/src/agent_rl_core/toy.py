from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

from agent_rl_core.interfaces import Trajectory


DEFAULT_ACTION_SPACE = ["inspect", "search", "tool_call", "reason", "answer"]


def build_toy_tasks(
    num_tasks: int,
    *,
    seed: int,
    action_space: List[str] | None = None,
    min_plan_steps: int = 3,
    max_plan_steps: int = 6,
    tool_calls_budget: float = 8.0,
    output_tokens_budget: float = 1200.0,
    latency_ms_budget: float = 15000.0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    actions = action_space or DEFAULT_ACTION_SPACE
    if "answer" not in actions:
        actions = list(actions) + ["answer"]

    tasks: List[Dict[str, Any]] = []
    core_actions = [a for a in actions if a != "answer"]

    for idx in range(num_tasks):
        n_steps = rng.randint(min_plan_steps, max_plan_steps)
        plan = [rng.choice(core_actions) for _ in range(max(1, n_steps - 1))]
        plan.append("answer")
        tasks.append(
            {
                "task_id": f"task_{idx:05d}",
                "target_plan": plan,
                "budgets": {
                    "tool_calls": float(tool_calls_budget),
                    "output_tokens": float(output_tokens_budget),
                    "latency_ms": float(latency_ms_budget),
                },
            }
        )
    return tasks


class ToyEnvironment:
    def __init__(self, *, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self._task: Dict[str, Any] = {}
        self._step_idx = 0
        self._matches = 0
        self._done = False

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self._task = task
        self._step_idx = 0
        self._matches = 0
        self._done = False
        return self._state()

    def _state(self) -> Dict[str, Any]:
        plan = list(self._task.get("target_plan", []))
        target_action = plan[self._step_idx] if self._step_idx < len(plan) else None
        return {
            "task_id": self._task.get("task_id", "unknown"),
            "step_idx": self._step_idx,
            "remaining_steps": max(0, len(plan) - self._step_idx),
            "target_action": target_action,
            "budgets": dict(self._task.get("budgets", {})),
            "task_metadata": dict(self._task.get("metadata", {})),
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._done:
            return {
                "state": self._state(),
                "reward": 0.0,
                "costs": {"tool_calls": 0.0, "output_tokens": 0.0, "latency_ms": 0.0},
                "done": True,
                "metadata": {"already_done": True},
            }

        plan = list(self._task.get("target_plan", []))
        target = plan[self._step_idx] if self._step_idx < len(plan) else "answer"
        action_name = str(action.get("name", "reason"))
        matched = action_name == target

        base_reward = 1.0 if matched else -0.2
        self._matches += 1 if matched else 0

        tool_calls = 1.0 if action_name != "answer" else 0.0
        output_tokens = float(max(16, len(str(action.get("content", action_name))) * 8))
        latency_ms = float(self.rng.randint(350, 2200))

        self._step_idx += 1
        self._done = self._step_idx >= len(plan)

        if self._done and self._matches == len(plan):
            base_reward += 1.5

        return {
            "state": self._state(),
            "reward": float(base_reward),
            "costs": {
                "tool_calls": tool_calls,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
            },
            "done": self._done,
            "metadata": {
                "matched_target": matched,
                "target_action": target,
                "task_complete": self._done and self._matches == len(plan),
            },
        }


@dataclass
class ToyPolicy:
    action_space: List[str]
    skill: float = 0.3
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.skill = max(0.01, min(0.99, float(self.skill)))

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        target_action = str(state.get("target_action", "answer"))
        pick_correct = self.rng.random() < self.skill
        if pick_correct:
            action_name = target_action
        else:
            candidates = [a for a in self.action_space if a != target_action]
            action_name = self.rng.choice(candidates) if candidates else target_action
        return {
            "name": action_name,
            "content": f"{action_name} step",
            "confidence": self.skill if pick_correct else (1.0 - self.skill),
        }

    def update_from_rollouts(
        self,
        *,
        rollouts: List[Dict[str, Any]],
        learning_rate: float,
        cost_penalty: float,
    ) -> None:
        if not rollouts:
            return
        avg_score = sum(float(r.get("verifier", {}).get("score", 0.0)) for r in rollouts) / float(len(rollouts))
        avg_tool_calls = sum(float(r.get("final_costs", {}).get("tool_calls", 0.0)) for r in rollouts) / float(
            len(rollouts)
        )
        target = avg_score - cost_penalty * (avg_tool_calls / 10.0)
        step = float(learning_rate) * (target - 0.4)
        self.skill = max(0.01, min(0.99, self.skill + step))

    def state_dict(self) -> Dict[str, Any]:
        return {"skill": self.skill, "action_space": list(self.action_space)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.skill = max(0.01, min(0.99, float(state.get("skill", self.skill))))
        maybe_actions = state.get("action_space")
        if isinstance(maybe_actions, list) and maybe_actions:
            self.action_space = [str(x) for x in maybe_actions]


class ToyVerifier:
    def score(self, trajectory: Trajectory) -> Dict[str, Any]:
        if not trajectory.steps:
            return {"score": 0.0, "success": False, "status": "empty"}

        matched = [
            1.0 if bool(step.metadata.get("matched_target", False)) else 0.0
            for step in trajectory.steps
        ]
        match_ratio = sum(matched) / float(len(matched))

        budgets = dict(trajectory.steps[0].state.get("budgets", {}))
        violations: Dict[str, float] = {}
        for key, budget in budgets.items():
            used = float(trajectory.final_costs.get(key, 0.0))
            violations[key] = max(0.0, used - float(budget))

        violation_penalty = 0.0
        if budgets:
            violation_penalty = sum(1.0 for v in violations.values() if v > 0.0) / float(len(budgets))

        score = max(0.0, min(1.0, match_ratio * (1.0 - 0.5 * violation_penalty)))
        success = score >= 0.95
        return {
            "score": score,
            "success": success,
            "status": "success" if success else "fail",
            "match_ratio": match_ratio,
            "budget_violations": violations,
        }
