"""
Cost credit estimation for C3-RL.

Estimate per-step marginal contribution for each cost channel from counterfactual replay.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from shared.envs.base import Trajectory

from .counterfactual import CounterfactualResult, InterventionType


@dataclass
class CostCreditMap:
    trajectory_id: str
    channel: str
    step_credits: List[float]
    normalized_credits: List[float]
    metadata: Dict[str, float] = field(default_factory=dict)


class CostCreditEstimator:
    def __init__(self, normalization: str = "signed") -> None:
        self.normalization = normalization

    def estimate(
        self,
        trajectory: Trajectory,
        cf_results: List[CounterfactualResult],
        *,
        channel: str,
    ) -> CostCreditMap:
        n_steps = len(trajectory.steps)
        step_credits = [0.0 for _ in range(n_steps)]
        step_counts = [0 for _ in range(n_steps)]

        base_cost = self._cost_value(trajectory, channel)

        for cf in cf_results:
            if not cf.is_valid or cf.cf_trajectory is None:
                continue

            cf_cost = self._cost_value(cf.cf_trajectory, channel)
            # Positive means removing/replacing a step decreases cost -> that step likely incurs cost.
            delta = float(base_cost - cf_cost)

            affected = self._affected_indices(cf, n_steps)
            if not affected:
                continue

            share = delta / float(len(affected))
            for idx in affected:
                step_credits[idx] += share
                step_counts[idx] += 1

        for i in range(n_steps):
            if step_counts[i] > 0:
                step_credits[i] /= float(step_counts[i])

        normalized = self._normalize(step_credits)
        return CostCreditMap(
            trajectory_id=trajectory.trajectory_id,
            channel=channel,
            step_credits=step_credits,
            normalized_credits=normalized,
            metadata={
                "base_cost": float(base_cost),
                "n_cf_results": float(len(cf_results)),
                "n_valid": float(sum(1 for cf in cf_results if cf.is_valid and cf.cf_trajectory is not None)),
            },
        )

    def estimate_multi(
        self,
        trajectory: Trajectory,
        cf_results: List[CounterfactualResult],
        *,
        channels: List[str],
    ) -> Dict[str, CostCreditMap]:
        return {
            channel: self.estimate(trajectory, cf_results, channel=channel)
            for channel in channels
        }

    def _affected_indices(self, cf: CounterfactualResult, n_steps: int) -> List[int]:
        intv = cf.intervention
        if intv.intervention_type == InterventionType.DELETE_STEP:
            if 0 <= intv.target_step < n_steps:
                return [intv.target_step]
            return []

        if intv.intervention_type == InterventionType.SWAP_STEP:
            if 0 <= intv.target_step < n_steps:
                return [intv.target_step]
            return []

        if intv.intervention_type == InterventionType.DELETE_BLOCK:
            start = int(intv.target_step)
            end = int(intv.end_step if intv.end_step is not None else (start + 1))
            return [i for i in range(start, min(end, n_steps)) if 0 <= i < n_steps]

        if intv.intervention_type == InterventionType.TRUNCATE:
            start = max(0, int(intv.target_step))
            return [i for i in range(start, n_steps)]

        return []

    @staticmethod
    def _cost_value(trajectory: Trajectory, channel: str) -> float:
        channel = str(channel)
        if channel == "tool_calls":
            return float(getattr(trajectory, "total_tool_calls", 0.0))
        if channel == "output_tokens":
            return float(getattr(trajectory, "total_tokens", 0.0))
        if channel == "latency_ms":
            return float(getattr(trajectory, "wall_time_seconds", 0.0)) * 1000.0

        # Generic extension point for extra metrics from metadata.
        extra = (trajectory.metadata or {}).get("extra_costs")
        if isinstance(extra, dict) and channel in extra:
            try:
                return float(extra[channel])
            except Exception:
                return 0.0
        return 0.0

    def _normalize(self, values: List[float]) -> List[float]:
        if not values:
            return []

        mode = str(self.normalization)
        if mode == "none":
            return [float(v) for v in values]

        if mode == "signed":
            max_abs = max(abs(v) for v in values)
            if max_abs <= 1e-12:
                return [0.0 for _ in values]
            return [float(v / max_abs) for v in values]

        if mode == "minmax":
            lo = min(values)
            hi = max(values)
            if (hi - lo) <= 1e-12:
                return [0.0 for _ in values]
            return [float((v - lo) / (hi - lo)) for v in values]

        if mode == "zscore":
            if len(values) <= 1:
                return [0.0 for _ in values]
            mean = statistics.fmean(values)
            std = statistics.pstdev(values)
            if std <= 1e-12:
                return [0.0 for _ in values]
            return [float((v - mean) / std) for v in values]

        if mode == "softmax":
            m = max(values)
            exp_vals = [math.exp(v - m) for v in values]
            denom = sum(exp_vals)
            if denom <= 1e-12:
                return [0.0 for _ in values]
            return [float(v / denom) for v in exp_vals]

        raise ValueError(f"Unknown normalization: {mode}")
