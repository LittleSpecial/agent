from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agent_rl_core.interfaces import StepRecord, Trajectory
from agent_rl_core.toy import DEFAULT_ACTION_SPACE, ToyEnvironment, ToyVerifier


class InterventionType(str, Enum):
    DELETE_STEP = "delete"
    DELETE_BLOCK = "delete_block"
    TRUNCATE = "truncate"
    SWAP_STEP = "swap"


@dataclass
class InterventionSpec:
    intervention_type: InterventionType
    target_step: int
    end_step: Optional[int] = None
    replacement_action: Optional[Dict[str, Any]] = None


def _clamp01(x: float) -> float:
    v = float(x)
    if not (v == v):
        return 0.0
    return max(0.0, min(1.0, v))


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
    max_interventions_per_traj: int = 3
    intervention_policy: str = "selective"
    dual_warmup_steps: int = 20
    lambda_max: float = 10.0
    learning_rate: float = 2e-2
    budget_targets: Dict[str, float] = field(
        default_factory=lambda: {"tool_calls": 8.0, "output_tokens": 1200.0, "latency_ms": 15000.0}
    )

    # Real counterfactual credit settings.
    counterfactual_k: int = 6
    intervention_types: List[str] = field(default_factory=lambda: ["delete", "truncate", "swap"])
    delete_block_size: int = 2
    credit_normalization: str = "signed"
    cost_normalization: str = "signed"
    reward_mode: str = "mixed"  # binary | score | mixed
    reward_blend_alpha: float = 0.7
    failure_reward_floor: float = -0.01
    action_space: List[str] = field(default_factory=lambda: list(DEFAULT_ACTION_SPACE))
    cf_cache_size: int = 20000


class C3Trainer:
    """
    A -> C3 upgrade path with *real counterfactual replay*:
    - Baseline A: counterfactual success credit only.
    - C3: add counterfactual cost credits + dual variables.
    """

    def __init__(self, config: C3Config, *, seed: int = 42) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.seed = int(seed)
        self.global_step = 0
        self.dual_vars: Dict[str, float] = {key: 0.0 for key in config.budget_targets}
        self._cf_cache: Dict[str, Dict[str, Any]] = {}

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

        # Selective policy: focus on uncertain / mismatched actions.
        ranked: List[Tuple[float, int]] = []
        for i, step in enumerate(trajectory.steps):
            confidence = float((step.action or {}).get("confidence", 0.5))
            mismatch = 0.0 if bool(step.metadata.get("matched_target", False)) else 1.0
            uncertainty = (1.0 - confidence) + mismatch
            ranked.append((uncertainty, i))
        ranked.sort(reverse=True)
        return [idx for _, idx in ranked[: self.config.max_interventions_per_traj]]

    def _reward_from_summary(self, summary: Dict[str, Any]) -> float:
        success = 1.0 if bool(summary.get("success", False)) else 0.0
        score = _clamp01(float(summary.get("score", success)))
        mode = str(self.config.reward_mode)
        if mode == "binary":
            reward = success
        elif mode == "score":
            reward = score
        else:
            alpha = max(0.0, min(1.0, float(self.config.reward_blend_alpha)))
            reward = (1.0 - alpha) * success + alpha * score
        if success < 0.5 and reward <= 0.0 and float(self.config.failure_reward_floor) != 0.0:
            reward = float(self.config.failure_reward_floor)
        return float(reward)

    def _stable_digest(self, payload: Any) -> str:
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _make_replay_seed(self, task: Dict[str, Any], trajectory: Trajectory) -> int:
        token = f"{self.seed}|{task.get('task_id', trajectory.task_id)}|{trajectory.task_id}|{self.global_step}"
        return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:12], 16) % (2**31 - 1)

    def _replay_summary(
        self,
        *,
        task: Dict[str, Any],
        actions: List[Dict[str, Any]],
        replay_seed: int,
    ) -> Dict[str, Any]:
        cache_key = self._stable_digest({"task": task, "actions": actions, "seed": replay_seed})
        if cache_key in self._cf_cache:
            return dict(self._cf_cache[cache_key])

        env = ToyEnvironment(seed=int(replay_seed))
        state = env.reset(copy.deepcopy(task))
        steps: List[StepRecord] = []
        final_costs: Dict[str, float] = {}
        total_reward = 0.0

        for step_idx, action in enumerate(actions):
            transition = env.step(copy.deepcopy(action))
            reward = float(transition.get("reward", 0.0))
            costs = dict(transition.get("costs", {}))
            done = bool(transition.get("done", False))
            next_state = dict(transition.get("state", {}))
            metadata = dict(transition.get("metadata", {}))
            metadata["done"] = done
            metadata["step_idx"] = step_idx

            steps.append(
                StepRecord(
                    state=dict(state),
                    action=dict(action),
                    reward=reward,
                    costs=costs,
                    metadata=metadata,
                )
            )
            total_reward += reward
            for key, value in costs.items():
                final_costs[key] = float(final_costs.get(key, 0.0) + float(value))

            state = next_state
            if done:
                break

        trajectory = Trajectory(
            task_id=str(task.get("task_id", "unknown")),
            steps=steps,
            final_reward=total_reward,
            final_costs=final_costs,
        )
        verifier = ToyVerifier().score(trajectory)
        success = bool(verifier.get("success", False))
        score = _clamp01(float(verifier.get("score", 1.0 if success else 0.0)))
        summary = {
            "success": success,
            "score": score,
            "final_costs": {k: float(v) for k, v in final_costs.items()},
        }

        if self.config.cf_cache_size > 0:
            if len(self._cf_cache) >= int(self.config.cf_cache_size):
                # keep cache bounded; removing random entry avoids O(n) LRU maintenance.
                drop_key = next(iter(self._cf_cache))
                self._cf_cache.pop(drop_key, None)
            self._cf_cache[cache_key] = dict(summary)
        return summary

    def _available_actions(self, task: Dict[str, Any]) -> List[str]:
        actions = list(self.config.action_space or DEFAULT_ACTION_SPACE)
        plan = task.get("target_plan")
        if isinstance(plan, list):
            for a in plan:
                if str(a) not in actions:
                    actions.append(str(a))
        if "answer" not in actions:
            actions.append("answer")
        if len(actions) < 2:
            actions = list(DEFAULT_ACTION_SPACE)
            if "answer" not in actions:
                actions.append("answer")
        return actions

    def _make_swap_action(self, *, task: Dict[str, Any], original_action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = str(original_action.get("name", "reason"))
        candidates = [a for a in self._available_actions(task) if a != name]
        if not candidates:
            return None
        repl_name = self.rng.choice(candidates)
        replacement = dict(original_action)
        replacement["name"] = repl_name
        replacement["content"] = f"{repl_name} step"
        replacement["confidence"] = float(original_action.get("confidence", 0.5)) * 0.5
        return replacement

    def _parse_intervention_types(self) -> List[InterventionType]:
        parsed: List[InterventionType] = []
        for raw in self.config.intervention_types:
            txt = str(raw).strip().lower()
            if txt in {"delete", "delete_step"}:
                parsed.append(InterventionType.DELETE_STEP)
            elif txt in {"delete_block", "block"}:
                parsed.append(InterventionType.DELETE_BLOCK)
            elif txt == "truncate":
                parsed.append(InterventionType.TRUNCATE)
            elif txt in {"swap", "swap_step"}:
                parsed.append(InterventionType.SWAP_STEP)
        if not parsed:
            parsed = [InterventionType.DELETE_STEP, InterventionType.TRUNCATE, InterventionType.SWAP_STEP]
        return parsed

    def _generate_interventions(
        self,
        *,
        task: Dict[str, Any],
        trajectory: Trajectory,
        base_actions: List[Dict[str, Any]],
    ) -> List[InterventionSpec]:
        n_steps = len(base_actions)
        if n_steps == 0:
            return []

        candidates = self._candidate_steps(trajectory)
        int_types = self._parse_intervention_types()
        specs: List[InterventionSpec] = []
        for step_idx in candidates:
            for int_type in int_types:
                if int_type == InterventionType.DELETE_STEP:
                    specs.append(
                        InterventionSpec(intervention_type=InterventionType.DELETE_STEP, target_step=int(step_idx))
                    )
                elif int_type == InterventionType.DELETE_BLOCK:
                    end = min(n_steps, int(step_idx) + max(1, int(self.config.delete_block_size)))
                    specs.append(
                        InterventionSpec(
                            intervention_type=InterventionType.DELETE_BLOCK,
                            target_step=int(step_idx),
                            end_step=int(end),
                        )
                    )
                elif int_type == InterventionType.TRUNCATE:
                    specs.append(InterventionSpec(intervention_type=InterventionType.TRUNCATE, target_step=int(step_idx)))
                elif int_type == InterventionType.SWAP_STEP:
                    replacement = self._make_swap_action(task=task, original_action=base_actions[step_idx])
                    if replacement is not None:
                        specs.append(
                            InterventionSpec(
                                intervention_type=InterventionType.SWAP_STEP,
                                target_step=int(step_idx),
                                replacement_action=replacement,
                            )
                        )

        self.rng.shuffle(specs)
        k = max(1, int(self.config.counterfactual_k))
        return specs[:k]

    def _apply_intervention(
        self,
        *,
        base_actions: List[Dict[str, Any]],
        spec: InterventionSpec,
    ) -> Optional[List[Dict[str, Any]]]:
        new_actions = [dict(a) for a in base_actions]
        n_steps = len(new_actions)

        if spec.intervention_type == InterventionType.DELETE_STEP:
            if not (0 <= spec.target_step < n_steps):
                return None
            del new_actions[spec.target_step]
            return new_actions

        if spec.intervention_type == InterventionType.DELETE_BLOCK:
            end = int(spec.end_step if spec.end_step is not None else (spec.target_step + 1))
            if not (0 <= spec.target_step < end <= n_steps):
                return None
            return new_actions[: spec.target_step] + new_actions[end:]

        if spec.intervention_type == InterventionType.TRUNCATE:
            if not (0 <= spec.target_step <= n_steps):
                return None
            return new_actions[: spec.target_step]

        if spec.intervention_type == InterventionType.SWAP_STEP:
            if not (0 <= spec.target_step < n_steps):
                return None
            if spec.replacement_action is None:
                return None
            new_actions[spec.target_step] = dict(spec.replacement_action)
            return new_actions

        return None

    def _affected_indices(self, spec: InterventionSpec, n_steps: int) -> List[int]:
        if spec.intervention_type == InterventionType.DELETE_BLOCK:
            end = int(spec.end_step if spec.end_step is not None else (spec.target_step + 1))
            return [i for i in range(spec.target_step, min(n_steps, end)) if 0 <= i < n_steps]

        if spec.intervention_type == InterventionType.TRUNCATE:
            # Truncating at t removes suffix [t, ...]; distribute credit across removed suffix.
            start = max(0, int(spec.target_step))
            return [i for i in range(start, n_steps)]

        if 0 <= spec.target_step < n_steps:
            return [int(spec.target_step)]
        return []

    def _counterfactual_credits(self, rollout: Dict[str, Any]) -> Dict[str, Any]:
        trajectory = rollout.get("trajectory")
        if not isinstance(trajectory, Trajectory):
            return {
                "success_credit": [],
                "cost_credit": {k: [] for k in self.dual_vars},
                "selected_steps": 0,
                "generated_counterfactuals": 0,
                "valid_counterfactuals": 0,
            }
        task = rollout.get("task")
        if not isinstance(task, dict):
            raise ValueError(
                "C3 real counterfactual replay requires rollout['task']. "
                "Attach the original task dict to each rollout before calling train_step()."
            )

        n_steps = len(trajectory.steps)
        if n_steps == 0:
            return {
                "success_credit": [],
                "cost_credit": {k: [] for k in self.dual_vars},
                "selected_steps": 0,
                "generated_counterfactuals": 0,
                "valid_counterfactuals": 0,
            }

        base_actions = [dict(s.action) for s in trajectory.steps]
        specs = self._generate_interventions(task=task, trajectory=trajectory, base_actions=base_actions)
        if not specs:
            return {
                "success_credit": [0.0 for _ in range(n_steps)],
                "cost_credit": {k: [0.0 for _ in range(n_steps)] for k in self.dual_vars},
                "selected_steps": 0,
                "generated_counterfactuals": 0,
                "valid_counterfactuals": 0,
            }

        replay_seed = self._make_replay_seed(task, trajectory)
        base_summary = self._replay_summary(task=task, actions=base_actions, replay_seed=replay_seed)
        base_reward = self._reward_from_summary(base_summary)
        base_costs = {k: float(base_summary.get("final_costs", {}).get(k, 0.0)) for k in self.dual_vars}

        success_credit = [0.0 for _ in range(n_steps)]
        success_counts = [0 for _ in range(n_steps)]
        cost_credit: Dict[str, List[float]] = {k: [0.0 for _ in range(n_steps)] for k in self.dual_vars}
        cost_counts: Dict[str, List[int]] = {k: [0 for _ in range(n_steps)] for k in self.dual_vars}
        valid_counterfactuals = 0

        for spec in specs:
            cf_actions = self._apply_intervention(base_actions=base_actions, spec=spec)
            if cf_actions is None:
                continue

            cf_summary = self._replay_summary(task=task, actions=cf_actions, replay_seed=replay_seed)
            cf_reward = self._reward_from_summary(cf_summary)
            delta_reward = float(base_reward - cf_reward)

            affected = self._affected_indices(spec, n_steps)
            if not affected:
                continue
            valid_counterfactuals += 1

            for idx in affected:
                success_credit[idx] += delta_reward
                success_counts[idx] += 1

            cf_costs = {k: float(cf_summary.get("final_costs", {}).get(k, 0.0)) for k in self.dual_vars}
            for key in self.dual_vars:
                delta_cost = float(base_costs.get(key, 0.0) - cf_costs.get(key, 0.0))
                for idx in affected:
                    cost_credit[key][idx] += delta_cost
                    cost_counts[key][idx] += 1

        for i in range(n_steps):
            if success_counts[i] > 0:
                success_credit[i] /= float(success_counts[i])
        for key in self.dual_vars:
            for i in range(n_steps):
                if cost_counts[key][i] > 0:
                    cost_credit[key][i] /= float(cost_counts[key][i])

        selected_steps = sum(1 for c in success_counts if c > 0)
        return {
            "success_credit": success_credit,
            "cost_credit": cost_credit,
            "selected_steps": int(selected_steps),
            "generated_counterfactuals": int(len(specs)),
            "valid_counterfactuals": int(valid_counterfactuals),
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
        generated_counterfactuals = 0
        valid_counterfactuals = 0

        n = float(len(rollouts))
        for rollout in rollouts:
            trajectory = rollout.get("trajectory")
            if not isinstance(trajectory, Trajectory):
                continue

            cf_pack = self._counterfactual_credits(rollout)
            success_raw = [float(x) for x in cf_pack["success_credit"]]
            cost_raw: Dict[str, List[float]] = {
                key: [float(v) for v in values]
                for key, values in dict(cf_pack["cost_credit"]).items()
            }
            success_norm = _normalize(success_raw, self.config.credit_normalization)
            cost_norm = {
                key: _normalize(cost_raw.get(key, []), self.config.cost_normalization)
                for key in self.dual_vars
            }

            baseline = list(success_norm)
            c3_values: List[float] = []
            for i in range(len(success_norm)):
                cost_term = 0.0
                for key, lam in self.dual_vars.items():
                    vals = cost_norm.get(key, [])
                    value = vals[i] if i < len(vals) else 0.0
                    cost_term += float(lam) * float(value)
                c3_values.append(float(success_norm[i] - cost_term))

            all_baseline.extend(baseline)
            all_c3.extend(c3_values)
            selected_steps += float(cf_pack["selected_steps"])
            generated_counterfactuals += int(cf_pack["generated_counterfactuals"])
            valid_counterfactuals += int(cf_pack["valid_counterfactuals"])

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
            "cf_generated_per_traj": float(generated_counterfactuals) / n,
            "cf_valid_per_traj": float(valid_counterfactuals) / n,
            "cf_valid_ratio": (
                float(valid_counterfactuals) / float(max(1, generated_counterfactuals))
            ),
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
