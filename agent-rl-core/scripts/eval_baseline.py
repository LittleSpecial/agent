#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

import yaml

from agent_rl_core.runner import RolloutConfig, RolloutRunner
from agent_rl_core.toy import (
    DEFAULT_ACTION_SPACE,
    ToyEnvironment,
    ToyPolicy,
    ToyVerifier,
    build_toy_tasks,
    load_jsonl_tasks,
)
from agent_rl_core.trainer import BaselineTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline evaluation entrypoint")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt", type=pathlib.Path, required=False)
    return parser.parse_args()


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config, got: {type(data)!r}")
    return data


def _resolve_action_space(config: Dict[str, Any]) -> list[str]:
    env_cfg = dict(config.get("environment", {}))
    action_space = env_cfg.get("action_space", DEFAULT_ACTION_SPACE)
    if not isinstance(action_space, list) or not action_space:
        return list(DEFAULT_ACTION_SPACE)
    return [str(a) for a in action_space]


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    seed = int(cfg.get("seed", 42))

    rollout_cfg = dict(cfg.get("rollout", {}))
    trainer_cfg = dict(cfg.get("trainer", {}))
    logging_cfg = dict(cfg.get("logging", {}))
    env_cfg = dict(cfg.get("environment", {}))
    policy_cfg = dict(cfg.get("policy", {}))
    constraints_cfg = dict(cfg.get("constraints", {}))

    action_space = _resolve_action_space(cfg)
    policy = ToyPolicy(
        action_space=action_space,
        skill=float(policy_cfg.get("init_skill", 0.35)),
        seed=seed,
    )
    if args.ckpt is not None and args.ckpt.exists():
        payload = json.loads(args.ckpt.read_text(encoding="utf-8"))
        policy.load_state_dict(dict(payload.get("policy_state", {})))

    env = ToyEnvironment(seed=seed + 7)
    verifier = ToyVerifier()
    runner = RolloutRunner(
        config=RolloutConfig(max_steps=int(rollout_cfg.get("max_steps", 20))),
        environment=env,
        policy=policy,
        verifier=verifier,
    )
    eval_dataset = env_cfg.get("eval_dataset")
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
    trainer = BaselineTrainer(
        config=TrainConfig(
            total_updates=0,
            batch_size=int(trainer_cfg.get("batch_size", 32)),
            seed=seed,
            output_dir=str(logging_cfg.get("save_dir", "outputs/baseline")),
        ),
        runner=runner,
        policy=policy,
        train_tasks=eval_tasks,
        eval_tasks=eval_tasks,
    )
    metrics = trainer.evaluate(tasks=eval_tasks, step=0)
    summary = {
        "experiment_name": cfg.get("experiment_name", "baseline_core"),
        "seed": seed,
        "eval_tasks": len(eval_tasks),
        "eval_dataset": eval_dataset,
        "checkpoint": str(args.ckpt) if args.ckpt else None,
        "eval_metrics": metrics,
        "policy_skill": policy.skill,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
