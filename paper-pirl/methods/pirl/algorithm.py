from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass
class PIRLConfig:
    invariance_weight: float = 0.1
    randomization_strength: str = "medium"
    curriculum: bool = True
    learning_rate: float = 2e-2
    retention_weight: float = 0.1
    num_skills: int = 3
    ood_hard_penalty: float = 0.6
    ood_easy_penalty: float = 0.3


class PIRLTrainer:
    def __init__(self, config: PIRLConfig, *, seed: int = 42) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.global_step = 0
        self.skill_ema: Dict[str, float] = {}

    def _strength_base(self) -> float:
        mapping = {"low": 0.1, "medium": 0.2, "high": 0.35}
        return float(mapping.get(self.config.randomization_strength, 0.2))

    def _strength_now(self) -> float:
        base = self._strength_base()
        if not self.config.curriculum:
            return base
        ramp = min(1.0, self.global_step / 120.0)
        return base * (0.25 + 0.75 * ramp)

    def _skill_id(self, rollout: Dict[str, Any]) -> str:
        trajectory = rollout.get("trajectory")
        if trajectory is None or not trajectory.steps:
            return "skill_0"
        state = trajectory.steps[0].state
        task_meta = dict(state.get("task_metadata", {}))
        if "skill_id" in task_meta:
            return str(task_meta["skill_id"])
        task_id = str(state.get("task_id", "task_0"))
        suffix = int(task_id.split("_")[-1]) if task_id.split("_")[-1].isdigit() else 0
        return f"skill_{suffix % max(1, self.config.num_skills)}"

    def _rollout_invariance_loss(self, rollout: Dict[str, Any], strength: float) -> float:
        trajectory = rollout.get("trajectory")
        if trajectory is None or not trajectory.steps:
            return 0.0
        losses = []
        for step in trajectory.steps:
            conf = float(step.action.get("confidence", 0.5))
            perturb = self.rng.uniform(-strength, strength)
            view2 = _clamp01(conf + perturb)
            losses.append(abs(conf - view2))
        if not losses:
            return 0.0
        return float(sum(losses) / len(losses))

    def _retention_bonus(self, skill_id: str, score: float) -> float:
        old = float(self.skill_ema.get(skill_id, score))
        self.skill_ema[skill_id] = 0.9 * old + 0.1 * score
        return float(score - old)

    def _policy_update(self, policy: Any, objective: float) -> None:
        if policy is None:
            return
        if hasattr(policy, "skill"):
            skill = float(getattr(policy, "skill"))
            skill = max(0.01, min(0.99, skill + self.config.learning_rate * objective * 0.05))
            setattr(policy, "skill", skill)

    def train_step(self, rollouts: List[Dict[str, Any]], *, policy: Any = None) -> Dict[str, Any]:
        self.global_step += 1
        if not rollouts:
            return {"step": self.global_step}

        strength = self._strength_now()
        n = float(len(rollouts))
        score_sum = 0.0
        inv_sum = 0.0
        retention_sum = 0.0
        success_sum = 0.0

        for rollout in rollouts:
            verifier = dict(rollout.get("verifier", {}))
            score = float(verifier.get("score", 0.0))
            success = bool(verifier.get("success", False))
            skill_id = self._skill_id(rollout)
            inv = self._rollout_invariance_loss(rollout, strength)
            retention = self._retention_bonus(skill_id, score)

            score_sum += score
            inv_sum += inv
            retention_sum += retention
            success_sum += 1.0 if success else 0.0

        avg_score = score_sum / n
        avg_invariance = inv_sum / n
        avg_retention = retention_sum / n
        objective = avg_score - self.config.invariance_weight * avg_invariance + self.config.retention_weight * avg_retention

        self._policy_update(policy, objective)

        ood_easy = max(0.0, avg_score * (1.0 - self.config.ood_easy_penalty * strength))
        ood_hard = max(0.0, avg_score * (1.0 - self.config.ood_hard_penalty * strength))
        robust_gap = avg_score - ood_hard

        return {
            "step": self.global_step,
            "strength": strength,
            "objective": objective,
            "avg_verifier_score": avg_score,
            "success_rate": success_sum / n,
            "invariance_loss": avg_invariance,
            "retention_delta": avg_retention,
            "ood_easy_score": ood_easy,
            "ood_hard_score": ood_hard,
            "robust_gap": robust_gap,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "skill_ema": dict(self.skill_ema),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", self.global_step))
        skill_ema = state.get("skill_ema", {})
        if isinstance(skill_ema, dict):
            self.skill_ema = {str(k): float(v) for k, v in skill_ema.items()}
