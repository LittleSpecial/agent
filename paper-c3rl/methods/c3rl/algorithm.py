from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from agent_rl_core.interfaces import Trajectory


def _signed_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    scale = max(abs(v) for v in values)
    if scale <= 1e-12:
        return [0.0 for _ in values]
    return [float(v / scale) for v in values]


@dataclass
class C3Config:
    dual_lr: float = 1e-3
    max_interventions_per_traj: int = 3
    intervention_policy: str = "selective"
    dual_warmup_steps: int = 20
    lambda_max: float = 10.0
    learning_rate: float = 2e-2
    budget_targets: Dict[str, float] = field(
        default_factory=lambda: {"tool_calls": 8.0, "output_tokens": 1200.0, "latency_ms": 15000.0}
    )


class C3Trainer:
    """
    A -> C3 upgrade path:
    - Baseline A: step success credit only.
    - C3: add cost credits + dual variables over budget constraints.
    """

    def __init__(self, config: C3Config, *, seed: int = 42) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.global_step = 0
        self.dual_vars: Dict[str, float] = {key: 0.0 for key in config.budget_targets}

    def _candidate_steps(self, trajectory: Trajectory) -> List[int]:
        n_steps = len(trajectory.steps)
        if n_steps == 0:
            return []
        if self.config.max_interventions_per_traj >= n_steps:
            return list(range(n_steps))

        if self.config.intervention_policy == "random":
            candidates = list(range(n_steps))
            self.rng.shuffle(candidates)
            return candidates[: self.config.max_interventions_per_traj]

        ranked: List[Tuple[float, int]] = []
        for i, step in enumerate(trajectory.steps):
            confidence = float(step.action.get("confidence", 0.5))
            mismatch = 0.0 if bool(step.metadata.get("matched_target", False)) else 1.0
            uncertainty = (1.0 - confidence) + mismatch
            ranked.append((uncertainty, i))
        ranked.sort(reverse=True)
        return [idx for _, idx in ranked[: self.config.max_interventions_per_traj]]

    def _estimate_success_credit(self, trajectory: Trajectory, step_idx: int) -> float:
        step = trajectory.steps[step_idx]
        matched = bool(step.metadata.get("matched_target", False))
        temporal = 1.0 - (float(step_idx) / max(1.0, float(len(trajectory.steps) - 1)))
        sign = 1.0 if matched else -0.8
        return sign * (0.5 + 0.5 * temporal)

    def _estimate_cost_credit(self, trajectory: Trajectory, step_idx: int) -> Dict[str, float]:
        step = trajectory.steps[step_idx]
        return {key: float(value) for key, value in step.costs.items()}

    def _trajectory_advantages(self, trajectory: Trajectory) -> Dict[str, Any]:
        chosen_steps = self._candidate_steps(trajectory)
        if not chosen_steps:
            return {"a_baseline": [], "a_c3": [], "selected_steps": 0}

        success_raw = [self._estimate_success_credit(trajectory, i) for i in chosen_steps]
        success_norm = _signed_normalize(success_raw)

        cost_by_key: Dict[str, List[float]] = {key: [] for key in self.dual_vars}
        for step_idx in chosen_steps:
            cc = self._estimate_cost_credit(trajectory, step_idx)
            for key in cost_by_key:
                cost_by_key[key].append(float(cc.get(key, 0.0)))
        cost_norm = {key: _signed_normalize(values) for key, values in cost_by_key.items()}

        baseline = list(success_norm)
        c3_values: List[float] = []
        for i in range(len(chosen_steps)):
            cost_term = 0.0
            for key, lam in self.dual_vars.items():
                values = cost_norm.get(key, [])
                value = values[i] if i < len(values) else 0.0
                cost_term += float(lam) * value
            c3_values.append(float(success_norm[i] - cost_term))

        return {
            "a_baseline": baseline,
            "a_c3": c3_values,
            "selected_steps": len(chosen_steps),
        }

    def _update_duals(self, batch_costs: Dict[str, float]) -> None:
        if self.global_step <= self.config.dual_warmup_steps:
            return
        for key, budget in self.config.budget_targets.items():
            violation = float(batch_costs.get(key, 0.0)) - float(budget)
            new_value = self.dual_vars[key] + self.config.dual_lr * violation
            self.dual_vars[key] = max(0.0, min(self.config.lambda_max, float(new_value)))

    def _policy_update(self, policy: Any, signal: float) -> None:
        if policy is None:
            return
        if hasattr(policy, "skill"):
            skill = float(getattr(policy, "skill"))
            skill = max(0.01, min(0.99, skill + self.config.learning_rate * signal * 0.05))
            setattr(policy, "skill", skill)

    def train_step(
        self,
        rollouts: List[Dict[str, Any]],
        *,
        policy: Any = None,
        use_c3: bool = True,
        update_duals: bool = True,
    ) -> Dict[str, Any]:
        self.global_step += 1
        if not rollouts:
            return {"step": self.global_step}

        all_baseline: List[float] = []
        all_c3: List[float] = []
        selected_steps = 0.0
        batch_costs: Dict[str, float] = {key: 0.0 for key in self.dual_vars}
        success = 0.0
        verifier_score = 0.0

        n = float(len(rollouts))
        for rollout in rollouts:
            trajectory = rollout.get("trajectory")
            if not isinstance(trajectory, Trajectory):
                continue
            parts = self._trajectory_advantages(trajectory)
            all_baseline.extend(parts["a_baseline"])
            all_c3.extend(parts["a_c3"])
            selected_steps += float(parts["selected_steps"])

            verifier = dict(rollout.get("verifier", {}))
            verifier_score += float(verifier.get("score", 0.0))
            success += 1.0 if bool(verifier.get("success", False)) else 0.0

            final_costs = dict(rollout.get("final_costs", {}))
            for key in batch_costs:
                batch_costs[key] += float(final_costs.get(key, 0.0))

        for key in list(batch_costs):
            batch_costs[key] /= n

        baseline_objective = sum(all_baseline) / float(max(1, len(all_baseline)))
        c3_objective = sum(all_c3) / float(max(1, len(all_c3)))
        chosen_objective = c3_objective if use_c3 else baseline_objective

        if update_duals:
            self._update_duals(batch_costs)
        self._policy_update(policy, chosen_objective)

        metrics: Dict[str, Any] = {
            "step": self.global_step,
            "baseline_objective": baseline_objective,
            "c3_objective": c3_objective,
            "selected_steps_per_traj": selected_steps / n,
            "avg_verifier_score": verifier_score / n,
            "success_rate": success / n,
        }
        for key, value in batch_costs.items():
            metrics[f"avg_cost_{key}"] = value
        for key, value in self.dual_vars.items():
            metrics[f"lambda_{key}"] = value
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "dual_vars": dict(self.dual_vars),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", self.global_step))
        duals = state.get("dual_vars", {})
        if isinstance(duals, dict):
            for key in self.dual_vars:
                self.dual_vars[key] = float(duals.get(key, self.dual_vars[key]))
