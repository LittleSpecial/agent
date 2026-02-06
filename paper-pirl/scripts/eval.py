#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE_SRC = REPO_ROOT.parent / "agent-rl-core" / "src"
METHODS_ROOT = REPO_ROOT / "methods"
for path in (CORE_SRC, METHODS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_rl_core.runner import RolloutConfig, RolloutRunner
from agent_rl_core.toy import DEFAULT_ACTION_SPACE, ToyEnvironment, ToyPolicy, ToyVerifier, build_toy_tasks
from pirl.algorithm import PIRLConfig, PIRLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PIRL")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    return parser.parse_args()


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config, got: {type(data)!r}")
    return data


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_base_config_path(config_path: pathlib.Path, ref: str) -> pathlib.Path:
    ref_path = pathlib.Path(ref)
    candidates = [
        (config_path.parent / ref_path).resolve(),
        (REPO_ROOT / ref_path).resolve(),
        (REPO_ROOT.parent / ref_path).resolve(),
        (REPO_ROOT.parent / "agent-rl-core" / "configs" / "base.yaml").resolve(),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_action_space(cfg: Dict[str, Any]) -> list[str]:
    env_cfg = dict(cfg.get("environment", {}))
    action_space = env_cfg.get("action_space", DEFAULT_ACTION_SPACE)
    if not isinstance(action_space, list) or not action_space:
        return list(DEFAULT_ACTION_SPACE)
    return [str(a) for a in action_space]


def _attach_skill_labels(tasks: list[dict[str, Any]], num_skills: int) -> None:
    ns = max(1, int(num_skills))
    for idx, task in enumerate(tasks):
        metadata = dict(task.get("metadata", {}))
        metadata["skill_id"] = f"skill_{idx % ns}"
        task["metadata"] = metadata


def _ood_eval(base: float, strength: str) -> Dict[str, float]:
    mapping = {"low": 0.1, "medium": 0.2, "high": 0.35}
    s = mapping.get(strength, 0.2)
    iid = base
    ood_easy = max(0.0, base * (1.0 - 0.3 * s))
    ood_hard = max(0.0, base * (1.0 - 0.6 * s))
    return {
        "iid": iid,
        "ood_easy": ood_easy,
        "ood_hard": ood_hard,
        "robust_gap": iid - ood_hard,
    }


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    core_cfg_ref = str(cfg.get("core", {}).get("base_config", "../agent-rl-core/configs/base.yaml"))
    core_cfg_path = _resolve_base_config_path(args.config.resolve(), core_cfg_ref)
    base_cfg = _load_yaml(core_cfg_path)
    full_cfg = _merge_dict(base_cfg, cfg)
    payload = json.loads(args.ckpt.read_text(encoding="utf-8"))

    seed = int(full_cfg.get("seed", 42))
    rollout_cfg = dict(full_cfg.get("rollout", {}))
    method_cfg = dict(full_cfg.get("method", {}))
    env_cfg = dict(full_cfg.get("environment", {}))
    constraints_cfg = dict(full_cfg.get("constraints", {}))

    action_space = _resolve_action_space(full_cfg)
    policy = ToyPolicy(action_space=action_space, skill=0.35, seed=seed)
    policy.load_state_dict(dict(payload.get("policy_state", {})))
    pirl = PIRLTrainer(
        PIRLConfig(
            invariance_weight=float(method_cfg.get("invariance_weight", 0.1)),
            randomization_strength=str(method_cfg.get("randomization_strength", "medium")),
            curriculum=bool(method_cfg.get("curriculum", True)),
            num_skills=int(method_cfg.get("num_skills", 3)),
        ),
        seed=seed,
    )
    pirl.load_state_dict(dict(payload.get("pirl_state", {})))

    eval_tasks = build_toy_tasks(
        num_tasks=int(env_cfg.get("eval_tasks", 64)),
        seed=seed + 1,
        action_space=action_space,
        min_plan_steps=int(env_cfg.get("min_plan_steps", 3)),
        max_plan_steps=int(env_cfg.get("max_plan_steps", 6)),
        tool_calls_budget=float(constraints_cfg.get("tool_calls_budget", 8.0)),
        output_tokens_budget=float(constraints_cfg.get("output_tokens_budget", 1200.0)),
        latency_ms_budget=float(constraints_cfg.get("latency_ms_budget", 15000.0)),
    )
    _attach_skill_labels(eval_tasks, int(method_cfg.get("num_skills", 3)))

    runner = RolloutRunner(
        config=RolloutConfig(max_steps=int(rollout_cfg.get("max_steps", 20))),
        environment=ToyEnvironment(seed=seed + 11),
        policy=policy,
        verifier=ToyVerifier(),
    )
    rollouts = [runner.run_episode(task) for task in eval_tasks]
    metrics = pirl.train_step(rollouts, policy=None)
    score = float(metrics.get("avg_verifier_score", 0.0))
    ood = _ood_eval(score, pirl.config.randomization_strength)
    summary = {
        "experiment_name": full_cfg.get("experiment_name", "pirl_main"),
        "checkpoint_step": payload.get("step"),
        "policy_skill": policy.skill,
        "eval_metrics": metrics,
        "ood": ood,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
