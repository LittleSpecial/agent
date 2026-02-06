#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any, Dict

import yaml

from agent_rl_core.runner import RolloutConfig, RolloutRunner
from agent_rl_core.toy import DEFAULT_ACTION_SPACE, ToyEnvironment, ToyPolicy, ToyVerifier, build_toy_tasks
from agent_rl_core.trainer import BaselineTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline training entrypoint")
    parser.add_argument("--config", type=pathlib.Path, required=True)
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
    random.seed(seed)

    rollout_cfg = dict(cfg.get("rollout", {}))
    trainer_cfg = dict(cfg.get("trainer", {}))
    logging_cfg = dict(cfg.get("logging", {}))
    env_cfg = dict(cfg.get("environment", {}))
    policy_cfg = dict(cfg.get("policy", {}))
    constraints_cfg = dict(cfg.get("constraints", {}))

    action_space = _resolve_action_space(cfg)

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

    policy = ToyPolicy(
        action_space=action_space,
        skill=float(policy_cfg.get("init_skill", 0.35)),
        seed=seed,
    )
    env = ToyEnvironment(seed=seed)
    verifier = ToyVerifier()
    runner = RolloutRunner(
        config=RolloutConfig(max_steps=int(rollout_cfg.get("max_steps", 20))),
        environment=env,
        policy=policy,
        verifier=verifier,
    )
    trainer = BaselineTrainer(
        config=TrainConfig(
            total_updates=int(trainer_cfg.get("total_updates", 1000)),
            batch_size=int(trainer_cfg.get("batch_size", 32)),
            learning_rate=float(trainer_cfg.get("learning_rate", 2e-2)),
            cost_penalty=float(trainer_cfg.get("cost_penalty", 0.02)),
            seed=seed,
            log_interval=int(trainer_cfg.get("log_interval", 20)),
            eval_interval=int(trainer_cfg.get("eval_interval", 100)),
            save_interval=int(trainer_cfg.get("save_interval", 100)),
            output_dir=str(logging_cfg.get("save_dir", "outputs/baseline")),
        ),
        runner=runner,
        policy=policy,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
    )
    result = trainer.train()
    summary = {
        "experiment_name": cfg.get("experiment_name", "baseline_core"),
        "seed": seed,
        "checkpoint": result["checkpoint"],
        "output_dir": result["output_dir"],
        "final_policy_skill": policy.skill,
        "latest_eval": result.get("latest_eval", {}),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
