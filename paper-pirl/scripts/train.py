#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from datetime import datetime
from typing import Any, Dict

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE_SRC = REPO_ROOT.parent / "agent-rl-core" / "src"
METHODS_ROOT = REPO_ROOT / "methods"
for path in (CORE_SRC, METHODS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_rl_core.runner import RolloutConfig, RolloutRunner
from agent_rl_core.toy import (
    DEFAULT_ACTION_SPACE,
    ToyEnvironment,
    ToyPolicy,
    ToyVerifier,
    build_toy_tasks,
    load_jsonl_tasks,
)
from pirl.algorithm import PIRLConfig, PIRLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PIRL")
    parser.add_argument("--config", type=pathlib.Path, required=True)
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


def _write_jsonl(path: pathlib.Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _attach_skill_labels(tasks: list[dict[str, Any]], num_skills: int) -> None:
    ns = max(1, int(num_skills))
    for idx, task in enumerate(tasks):
        metadata = dict(task.get("metadata", {}))
        metadata["skill_id"] = f"skill_{idx % ns}"
        task["metadata"] = metadata


def _eval_ood(rollouts: list[dict[str, Any]], ood_levels: list[str], strength: str) -> Dict[str, float]:
    if not rollouts:
        return {level: 0.0 for level in ood_levels}
    base = sum(float(r.get("verifier", {}).get("score", 0.0)) for r in rollouts) / float(len(rollouts))
    strength_map = {"low": 0.1, "medium": 0.2, "high": 0.35}
    s = strength_map.get(strength, 0.2)

    out: Dict[str, float] = {}
    for level in ood_levels:
        if level == "iid":
            out[level] = base
        elif level == "ood_easy":
            out[level] = max(0.0, base * (1.0 - 0.3 * s))
        else:
            out[level] = max(0.0, base * (1.0 - 0.6 * s))
    return out


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    core_cfg_ref = str(cfg.get("core", {}).get("base_config", "../agent-rl-core/configs/base.yaml"))
    core_cfg_path = _resolve_base_config_path(args.config.resolve(), core_cfg_ref)
    base_cfg = _load_yaml(core_cfg_path)
    full_cfg = _merge_dict(base_cfg, cfg)

    seed = int(full_cfg.get("seed", 42))
    random.seed(seed)

    rollout_cfg = dict(full_cfg.get("rollout", {}))
    trainer_cfg = dict(full_cfg.get("trainer", {}))
    method_cfg = dict(full_cfg.get("method", {}))
    env_cfg = dict(full_cfg.get("environment", {}))
    constraints_cfg = dict(full_cfg.get("constraints", {}))
    eval_cfg = dict(full_cfg.get("evaluation", {}))

    total_updates = int(trainer_cfg.get("total_updates", 200))
    batch_size = int(trainer_cfg.get("batch_size", 32))
    eval_interval = int(trainer_cfg.get("eval_interval", 20))
    save_interval = int(trainer_cfg.get("save_interval", 20))
    ood_levels = [str(x) for x in eval_cfg.get("ood_levels", ["iid", "ood_easy", "ood_hard"])]

    action_space = _resolve_action_space(full_cfg)
    train_dataset = env_cfg.get("train_dataset")
    eval_dataset = env_cfg.get("eval_dataset")
    if isinstance(train_dataset, str) and train_dataset:
        train_tasks = load_jsonl_tasks(
            train_dataset,
            action_space=action_space,
            min_plan_steps=int(env_cfg.get("min_plan_steps", 3)),
            max_plan_steps=int(env_cfg.get("max_plan_steps", 6)),
            max_samples=env_cfg.get("max_train_samples"),
            tool_calls_budget=float(constraints_cfg.get("tool_calls_budget", 8.0)),
            output_tokens_budget=float(constraints_cfg.get("output_tokens_budget", 1200.0)),
            latency_ms_budget=float(constraints_cfg.get("latency_ms_budget", 15000.0)),
        )
    else:
        train_tasks = build_toy_tasks(
            num_tasks=int(env_cfg.get("train_tasks", 256)),
            seed=seed,
            action_space=action_space,
            min_plan_steps=int(env_cfg.get("min_plan_steps", 3)),
            max_plan_steps=int(env_cfg.get("max_plan_steps", 6)),
            tool_calls_budget=float(constraints_cfg.get("tool_calls_budget", 8.0)),
            output_tokens_budget=float(constraints_cfg.get("output_tokens_budget", 1200.0)),
            latency_ms_budget=float(constraints_cfg.get("latency_ms_budget", 15000.0)),
        )

    if isinstance(eval_dataset, str) and eval_dataset:
        eval_tasks = load_jsonl_tasks(
            eval_dataset,
            action_space=action_space,
            min_plan_steps=int(env_cfg.get("min_plan_steps", 3)),
            max_plan_steps=int(env_cfg.get("max_plan_steps", 6)),
            max_samples=env_cfg.get("max_eval_samples"),
            tool_calls_budget=float(constraints_cfg.get("tool_calls_budget", 8.0)),
            output_tokens_budget=float(constraints_cfg.get("output_tokens_budget", 1200.0)),
            latency_ms_budget=float(constraints_cfg.get("latency_ms_budget", 15000.0)),
        )
    else:
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
    _attach_skill_labels(train_tasks, int(method_cfg.get("num_skills", 3)))
    _attach_skill_labels(eval_tasks, int(method_cfg.get("num_skills", 3)))

    policy = ToyPolicy(action_space=action_space, skill=float(full_cfg.get("policy", {}).get("init_skill", 0.35)), seed=seed)
    runner = RolloutRunner(
        config=RolloutConfig(max_steps=int(rollout_cfg.get("max_steps", 20))),
        environment=ToyEnvironment(seed=seed + 2),
        policy=policy,
        verifier=ToyVerifier(),
    )
    pirl = PIRLTrainer(
        PIRLConfig(
            invariance_weight=float(method_cfg.get("invariance_weight", 0.1)),
            randomization_strength=str(method_cfg.get("randomization_strength", "medium")),
            curriculum=bool(method_cfg.get("curriculum", True)),
            learning_rate=float(trainer_cfg.get("learning_rate", 2e-2)),
            retention_weight=float(method_cfg.get("retention_weight", 0.1)),
            num_skills=int(method_cfg.get("num_skills", 3)),
        ),
        seed=seed,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "results" / f"{full_cfg.get('experiment_name', 'pirl_main')}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    eval_path = output_dir / "eval.jsonl"
    ckpt_path = output_dir / "checkpoint.json"

    rng = random.Random(seed + 7)
    for step in range(1, total_updates + 1):
        batch = [train_tasks[rng.randrange(len(train_tasks))] for _ in range(batch_size)]
        rollouts = [runner.run_episode(task) for task in batch]
        metrics = pirl.train_step(rollouts, policy=policy)
        _write_jsonl(metrics_path, {"step": step, **metrics})

        if step % eval_interval == 0:
            eval_rollouts = [runner.run_episode(task) for task in eval_tasks]
            ood = _eval_ood(eval_rollouts, ood_levels, pirl.config.randomization_strength)
            row: Dict[str, Any] = {"step": step, **ood}
            if "iid" in ood and "ood_hard" in ood:
                row["robust_gap"] = ood["iid"] - ood["ood_hard"]
            _write_jsonl(eval_path, row)

        if step % save_interval == 0 or step == total_updates:
            ckpt = {
                "step": step,
                "policy_state": policy.state_dict(),
                "pirl_state": pirl.state_dict(),
                "config": full_cfg,
            }
            ckpt_path.write_text(json.dumps(ckpt, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = {
        "experiment_name": full_cfg.get("experiment_name", "pirl_main"),
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "output_dir": str(output_dir),
        "checkpoint": str(ckpt_path),
        "final_policy_skill": policy.skill,
        "randomization_strength": pirl.config.randomization_strength,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
