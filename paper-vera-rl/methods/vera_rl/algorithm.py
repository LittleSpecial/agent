from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List


@dataclass
class VERAConfig:
    recheck_budget_ratio: float = 0.1
    uncertainty_threshold: float = 0.7
    calibration_window: int = 2000
    learning_rate: float = 2e-2


class VERATrainer:
    """
    Verifier-uncertainty aware add-on:
    - score uncertainty
    - budgeted selective recheck
    - estimate fp/fn and produce corrected reward
    """

    def __init__(self, config: VERAConfig) -> None:
        self.config = config
        self.global_step = 0
        self._window: Deque[Dict[str, int]] = deque(maxlen=max(10, int(config.calibration_window)))
        self.p_fp = 0.05
        self.p_fn = 0.05

    def _uncertainty(self, rollout: Dict[str, Any]) -> float:
        verifier = dict(rollout.get("verifier", {}))
        score = float(verifier.get("score", 0.0))
        confidence = 0.5
        trajectory = rollout.get("trajectory")
        if trajectory is not None and trajectory.steps:
            confs = [float(step.action.get("confidence", 0.5)) for step in trajectory.steps]
            confidence = sum(confs) / float(len(confs))
        margin_term = 1.0 - abs(score - 0.5) * 2.0
        model_term = 1.0 - confidence
        return float(0.5 * margin_term + 0.5 * model_term)

    def _proxy_true_label(self, rollout: Dict[str, Any]) -> bool:
        verifier = dict(rollout.get("verifier", {}))
        match_ratio = float(verifier.get("match_ratio", verifier.get("score", 0.0)))
        return bool(match_ratio >= 0.9)

    def _update_noise_estimates(self, rechecked: List[Dict[str, Any]]) -> None:
        for rollout in rechecked:
            obs = bool(dict(rollout.get("verifier", {})).get("success", False))
            true = self._proxy_true_label(rollout)
            self._window.append({"obs": int(obs), "true": int(true)})
        if not self._window:
            return

        fp = 0.0
        fn = 0.0
        n_true0 = 0.0
        n_true1 = 0.0
        for item in self._window:
            obs = bool(item["obs"])
            true = bool(item["true"])
            if true:
                n_true1 += 1.0
                if not obs:
                    fn += 1.0
            else:
                n_true0 += 1.0
                if obs:
                    fp += 1.0
        self.p_fp = fp / max(1.0, n_true0)
        self.p_fn = fn / max(1.0, n_true1)

    def _correct_reward(self, rollout: Dict[str, Any]) -> float:
        verifier = dict(rollout.get("verifier", {}))
        obs_success = bool(verifier.get("success", False))
        raw_score = float(verifier.get("score", 0.0))
        if obs_success:
            corrected = (1.0 - self.p_fp) * 1.0 + self.p_fp * 0.0
        else:
            corrected = self.p_fn * 1.0 + (1.0 - self.p_fn) * 0.0
        return float(0.5 * raw_score + 0.5 * corrected)

    def _policy_update(self, policy: Any, corrected_score: float) -> None:
        if policy is None:
            return
        if hasattr(policy, "skill"):
            skill = float(getattr(policy, "skill"))
            step = self.config.learning_rate * (corrected_score - 0.35) * 0.05
            skill = max(0.01, min(0.99, skill + step))
            setattr(policy, "skill", skill)

    def train_step(self, rollouts: List[Dict[str, Any]], *, policy: Any = None) -> Dict[str, Any]:
        self.global_step += 1
        if not rollouts:
            return {"step": self.global_step}

        n = float(len(rollouts))
        scored = [(self._uncertainty(r), r) for r in rollouts]
        scored.sort(key=lambda x: x[0], reverse=True)
        budget = max(1, int(self.config.recheck_budget_ratio * len(rollouts)))
        rechecked = [r for u, r in scored[:budget] if u >= self.config.uncertainty_threshold]
        self._update_noise_estimates(rechecked)

        raw = sum(float(dict(r.get("verifier", {})).get("score", 0.0)) for r in rollouts) / n
        corrected_values = [self._correct_reward(r) for r in rollouts]
        corrected = sum(corrected_values) / n
        success = sum(1.0 for r in rollouts if bool(dict(r.get("verifier", {})).get("success", False))) / n

        self._policy_update(policy, corrected)

        return {
            "step": self.global_step,
            "raw_score": raw,
            "corrected_score": corrected,
            "success_rate": success,
            "recheck_count": float(len(rechecked)),
            "recheck_ratio": float(len(rechecked)) / n,
            "p_fp": self.p_fp,
            "p_fn": self.p_fn,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "p_fp": self.p_fp,
            "p_fn": self.p_fn,
            "window": list(self._window),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", self.global_step))
        self.p_fp = float(state.get("p_fp", self.p_fp))
        self.p_fn = float(state.get("p_fn", self.p_fn))
        window = state.get("window", [])
        if isinstance(window, list):
            self._window.clear()
            for item in window:
                if isinstance(item, dict) and "obs" in item and "true" in item:
                    self._window.append({"obs": int(item["obs"]), "true": int(item["true"])})
