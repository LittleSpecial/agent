from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent_rl_core.interfaces import Environment, Policy, StepRecord, Trajectory, Verifier


@dataclass
class RolloutConfig:
    max_steps: int = 20
    stop_on_done: bool = True


class RolloutRunner:
    def __init__(
        self,
        config: RolloutConfig,
        environment: Environment,
        policy: Policy,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self.config = config
        self.environment = environment
        self.policy = policy
        self.verifier = verifier

    def run_episode(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one rollout and return trajectory and diagnostics."""
        task_id = str(task.get("task_id", "unknown"))
        state = self.environment.reset(task)

        steps = []
        total_reward = 0.0
        final_costs: Dict[str, float] = {}
        terminated = False

        for step_idx in range(self.config.max_steps):
            action = self.policy.act(state)
            transition = self.environment.step(action)

            reward = float(transition.get("reward", 0.0))
            costs = dict(transition.get("costs", {}))
            done = bool(transition.get("done", False))
            next_state = dict(transition.get("state", {}))
            metadata = dict(transition.get("metadata", {}))
            metadata["done"] = done
            metadata["step_idx"] = step_idx

            step_record = StepRecord(
                state=dict(state),
                action=dict(action),
                reward=reward,
                costs=costs,
                metadata=metadata,
            )
            steps.append(step_record)

            total_reward += reward
            for key, value in costs.items():
                final_costs[key] = float(final_costs.get(key, 0.0) + float(value))

            state = next_state
            if done and self.config.stop_on_done:
                terminated = True
                break

        trajectory = Trajectory(
            task_id=task_id,
            steps=steps,
            final_reward=total_reward,
            final_costs=final_costs,
        )

        verifier_output: Dict[str, Any] = {}
        if self.verifier is not None:
            verifier_output = dict(self.verifier.score(trajectory))

        return {
            "task_id": task_id,
            "status": verifier_output.get("status", "ok"),
            "terminated": terminated,
            "trajectory": trajectory,
            "steps": steps,
            "verifier": verifier_output,
            "final_reward": trajectory.final_reward,
            "final_costs": trajectory.final_costs,
        }
