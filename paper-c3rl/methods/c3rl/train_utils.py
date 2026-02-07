"""
Shared helpers for Paper A training backends.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from shared.envs.base import Trajectory


@dataclass
class RolloutState:
    task: Dict
    observation: str
    done: bool = False
    trajectory: Optional[Trajectory] = None
    action_count: int = 0
    latest_feedback: str = ""


def clone_task_with_group_id(task: Dict, *, group_id: str) -> Dict:
    """
    Make a shallow-cloned task dict with a unique task_id for GRPO grouping.
    Keep the original dataset task id in metadata.base_task_id.
    """
    cloned = dict(task)
    base_task_id = str(cloned.get("task_id", "unknown"))
    cloned["task_id"] = group_id
    meta = dict(cloned.get("metadata") or {})
    meta.setdefault("base_task_id", base_task_id)
    meta.setdefault("group_id", group_id)
    cloned["metadata"] = meta
    return cloned


def clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not (v == v):  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def trajectory_reward(
    trajectory: Trajectory,
    *,
    mode: str,
    blend_alpha: float,
    failure_reward_floor: float = 0.0,
) -> float:
    binary = clamp01(float(trajectory.r_final))
    score = clamp01(float(getattr(trajectory.verifier_info, "score", binary)))
    if mode == "binary":
        reward = binary
    elif mode == "score":
        reward = score
    else:
        alpha = max(0.0, min(1.0, float(blend_alpha)))
        reward = (1.0 - alpha) * binary + alpha * score
    if binary < 0.5 and reward <= 0.0 and failure_reward_floor != 0.0:
        reward = float(failure_reward_floor)
    return float(reward)


def cf_result_reward(
    cf_result,
    *,
    mode: str,
    blend_alpha: float,
    failure_reward_floor: float = 0.0,
) -> float:
    binary = clamp01(float(getattr(cf_result, "r_final_cf", 0.0)))
    score = binary
    cf_traj = getattr(cf_result, "cf_trajectory", None)
    if cf_traj is not None and getattr(cf_traj, "verifier_info", None) is not None:
        score = clamp01(float(getattr(cf_traj.verifier_info, "score", binary)))
    if mode == "binary":
        reward = binary
    elif mode == "score":
        reward = score
    else:
        alpha = max(0.0, min(1.0, float(blend_alpha)))
        reward = (1.0 - alpha) * binary + alpha * score
    if binary < 0.5 and reward <= 0.0 and failure_reward_floor != 0.0:
        reward = float(failure_reward_floor)
    return float(reward)


def base_task_id_from_trajectory(trajectory: Trajectory) -> Optional[str]:
    task_meta = ((trajectory.metadata or {}).get("task") or {}).get("metadata") or {}
    base_id = task_meta.get("base_task_id")
    if base_id is None:
        return None
    return str(base_id)


def compute_grpo_advantages(
    trajectories: List[Trajectory],
    rewards: List[float],
    *,
    flat_group_fallback: str = "batch_centered",
) -> List[float]:
    if len(trajectories) != len(rewards):
        raise ValueError("trajectories/rewards length mismatch")
    if not trajectories:
        return []

    rewards = [float(r) for r in rewards]
    batch_mean = statistics.fmean(rewards)
    task_groups: Dict[str, List[int]] = {}
    for i, traj in enumerate(trajectories):
        task_groups.setdefault(traj.task_id, []).append(i)

    advantages = [0.0 for _ in trajectories]
    for indices in task_groups.values():
        group_rewards = [rewards[i] for i in indices]
        if len(group_rewards) > 1:
            mean = statistics.fmean(group_rewards)
            std = statistics.pstdev(group_rewards)
            if std > 1e-8:
                for i in indices:
                    advantages[i] = (rewards[i] - mean) / std
            else:
                for i in indices:
                    if flat_group_fallback == "zero":
                        advantages[i] = 0.0
                    elif flat_group_fallback == "batch_centered":
                        advantages[i] = rewards[i] - batch_mean
                    elif flat_group_fallback == "raw":
                        advantages[i] = rewards[i]
                    else:
                        raise ValueError(f"Unknown flat_group_fallback: {flat_group_fallback}")
        else:
            i = indices[0]
            if flat_group_fallback == "zero":
                advantages[i] = 0.0
            elif flat_group_fallback == "batch_centered":
                advantages[i] = rewards[i] - batch_mean
            elif flat_group_fallback == "raw":
                advantages[i] = rewards[i]
            else:
                raise ValueError(f"Unknown flat_group_fallback: {flat_group_fallback}")

    return advantages


def to_float_list(x) -> List[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    if hasattr(x, "tolist"):
        return [float(v) for v in x.tolist()]
    return [float(v) for v in x]


def pass_at_k_from_groups(trajectories: List[Trajectory], k: int) -> float:
    groups: Dict[str, List[Trajectory]] = {}
    for traj in trajectories:
        groups.setdefault(traj.task_id, []).append(traj)

    if not groups:
        return 0.0

    hits = 0
    for ts in groups.values():
        ts_k = ts[:k]
        if any(t.r_final > 0.5 for t in ts_k):
            hits += 1
    return hits / len(groups)


def build_code_followup_observation(
    task: Dict,
    *,
    current_code: str,
    last_feedback: str,
    show_tests: bool,
) -> str:
    """
    Build a richer multi-turn observation so the model can iteratively refine code.
    """
    language = str(task.get("language", "python"))
    parts: List[str] = [f"Task: {task.get('prompt', '')}"]
    if current_code:
        parts.append(f"Current Code:\n```{language}\n{current_code}\n```")
    if last_feedback:
        parts.append(f"Last Test Feedback:\n{last_feedback}")
    test_code = task.get("test_code")
    if show_tests and isinstance(test_code, str) and test_code:
        parts.append(f"Test Cases:\n```{language}\n{test_code}\n```")
    return "\n\n".join(parts).strip()


def guess_lora_target_modules(model) -> List[str]:
    """
    Best-effort LoRA target module guessing for common decoder-only LMs.
    """
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    picked = [c for c in candidates if c in names]
    return picked if picked else ["q_proj", "k_proj", "v_proj", "o_proj"]
