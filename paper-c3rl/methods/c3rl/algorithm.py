from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from shared.envs.base import Trajectory


def _normalize(values: List[float], mode: str) -> List[float]:
    if not values:
        return []

    if mode == "none":
        return [float(v) for v in values]

    if mode == "signed":
        scale = max(abs(v) for v in values)
        if scale <= 1e-12:
            return [0.0 for _ in values]
        return [float(v / scale) for v in values]

    if mode == "minmax":
        lo = min(values)
        hi = max(values)
        if (hi - lo) <= 1e-12:
            return [0.0 for _ in values]
        return [float((v - lo) / (hi - lo)) for v in values]

    if mode == "zscore":
        n = float(len(values))
        mean = sum(float(v) for v in values) / n
        var = sum((float(v) - mean) ** 2 for v in values) / n
        std = var**0.5
        if std <= 1e-12:
            return [0.0 for _ in values]
        return [float((v - mean) / std) for v in values]

    raise ValueError(f"Unknown normalization mode: {mode}")


@dataclass
class C3Config:
    dual_lr: float = 1e-3
    dual_warmup_steps: int = 200
    lambda_max: float = 10.0
    budget_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "tool_calls": 8.0,
            "output_tokens": 1200.0,
            "latency_ms": 15000.0,
        }
    )
    credit_normalization: str = "signed"
    cost_normalization: str = "signed"
    fallback_to_adv_when_zero_credit: bool = True
    zero_credit_threshold: float = 1e-8


class C3Trainer:
    """
    Counterfactual Cost-Credit Constrained controller.

    Responsibilities:
    - maintain/step dual variables for budget constraints
    - combine success credit and cost credit into step-level C3 signal
    - map step-level C3 signal to rollout weight multiplier
    """

    def __init__(self, config: C3Config) -> None:
        self.config = config
        self.global_step = 0
        self.dual_vars: Dict[str, float] = {
            key: 0.0 for key in config.budget_targets
        }

    @staticmethod
    def trajectory_costs(trajectory: Trajectory) -> Dict[str, float]:
        return {
            "tool_calls": float(getattr(trajectory, "total_tool_calls", 0.0)),
            "output_tokens": float(getattr(trajectory, "total_tokens", 0.0)),
            "latency_ms": float(getattr(trajectory, "wall_time_seconds", 0.0)) * 1000.0,
        }

    @staticmethod
    def batch_average_costs(trajectories: List[Trajectory]) -> Dict[str, float]:
        if not trajectories:
            return {"tool_calls": 0.0, "output_tokens": 0.0, "latency_ms": 0.0}
        agg = {"tool_calls": 0.0, "output_tokens": 0.0, "latency_ms": 0.0}
        for traj in trajectories:
            costs = C3Trainer.trajectory_costs(traj)
            for k in agg:
                agg[k] += float(costs.get(k, 0.0))
        n = float(len(trajectories))
        for k in agg:
            agg[k] /= n
        return agg

    def combine_step_signals(
        self,
        *,
        success_step_credits: List[float],
        cost_step_credits: Mapping[str, List[float]],
    ) -> List[float]:
        success_norm = _normalize(success_step_credits, self.config.credit_normalization)
        cost_norm: Dict[str, List[float]] = {
            key: _normalize(list(cost_step_credits.get(key, [])), self.config.cost_normalization)
            for key in self.dual_vars
        }

        out: List[float] = []
        for i in range(len(success_norm)):
            cost_term = 0.0
            for key, lam in self.dual_vars.items():
                vals = cost_norm.get(key, [])
                v = vals[i] if i < len(vals) else 0.0
                cost_term += float(lam) * float(v)
            out.append(float(success_norm[i] - cost_term))
        return out

    def map_to_rollout_weight(
        self,
        *,
        trajectory_advantage: float,
        step_signal: List[float],
        logprob_step_indices: List[int],
    ) -> Tuple[float, float]:
        if not step_signal:
            return float(trajectory_advantage), 1.0

        if logprob_step_indices:
            selected = [
                float(step_signal[i])
                for i in logprob_step_indices
                if 0 <= i < len(step_signal)
            ]
        else:
            selected = [float(v) for v in step_signal]

        if not selected:
            return float(trajectory_advantage), 1.0

        if (
            self.config.fallback_to_adv_when_zero_credit
            and max(abs(v) for v in selected) <= float(self.config.zero_credit_threshold)
        ):
            credit_weight = 1.0
        else:
            credit_weight = sum(selected) / float(len(selected))

        return float(trajectory_advantage) * float(credit_weight), float(credit_weight)

    def update_duals(self, batch_costs: Mapping[str, float]) -> None:
        self.global_step += 1
        if self.global_step <= int(self.config.dual_warmup_steps):
            return

        for key, budget in self.config.budget_targets.items():
            used = float(batch_costs.get(key, 0.0))
            violation = used - float(budget)
            new_value = float(self.dual_vars.get(key, 0.0)) + float(self.config.dual_lr) * violation
            self.dual_vars[key] = max(0.0, min(float(self.config.lambda_max), float(new_value)))

    def violation_rate(self, costs: Mapping[str, float]) -> float:
        if not self.config.budget_targets:
            return 0.0
        violations = 0.0
        total = 0.0
        for key, budget in self.config.budget_targets.items():
            total += 1.0
            if float(costs.get(key, 0.0)) > float(budget):
                violations += 1.0
        return float(violations / max(1.0, total))

    def state_dict(self) -> Dict[str, object]:
        return {
            "global_step": int(self.global_step),
            "dual_vars": dict(self.dual_vars),
            "budget_targets": dict(self.config.budget_targets),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.global_step = int(state.get("global_step", self.global_step))
        duals = state.get("dual_vars", {})
        if isinstance(duals, Mapping):
            for key in self.dual_vars:
                self.dual_vars[key] = float(duals.get(key, self.dual_vars[key]))
