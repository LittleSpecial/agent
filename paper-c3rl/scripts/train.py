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
from c3rl.algorithm import C3Config, C3Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C3-RL (A->C3 upgrade line)")
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


def _avg_rollout_metrics(rollouts: list[dict[str, Any]]) -> Dict[str, float]:
    if not rollouts:
        return {}
    n = float(len(rollouts))
    score = sum(float(r.get("verifier", {}).get("score", 0.0)) for r in rollouts) / n
    success = sum(1.0 for r in rollouts if bool(r.get("verifier", {}).get("success", False))) / n
    out: Dict[str, float] = {"avg_verifier_score": score, "success_rate": success}
    costs: Dict[str, float] = {}
    for rollout in rollouts:
        for key, value in dict(rollout.get("final_costs", {})).items():
            costs[key] = costs.get(key, 0.0) + float(value)
    for key, value in costs.items():
        out[f"avg_cost_{key}"] = value / n
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

    total_updates = int(trainer_cfg.get("total_updates", 200))
    batch_size = int(trainer_cfg.get("batch_size", 32))
    eval_interval = int(trainer_cfg.get("eval_interval", 20))
    save_interval = int(trainer_cfg.get("save_interval", 20))
    baseline_warmup_updates = int(method_cfg.get("baseline_warmup_updates", max(10, total_updates // 5)))

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

    policy = ToyPolicy(action_space=action_space, skill=float(full_cfg.get("policy", {}).get("init_skill", 0.35)), seed=seed)
    env = ToyEnvironment(seed=seed)
    verifier = ToyVerifier()
    runner = RolloutRunner(
        config=RolloutConfig(max_steps=int(rollout_cfg.get("max_steps", 20))),
        environment=env,
        policy=policy,
        verifier=verifier,
    )
    c3 = C3Trainer(
        C3Config(
            dual_lr=float(method_cfg.get("dual_lr", 1e-3)),
            max_interventions_per_traj=int(method_cfg.get("max_interventions_per_traj", 3)),
            intervention_policy=str(method_cfg.get("intervention_policy", "selective")),
            dual_warmup_steps=baseline_warmup_updates,
            learning_rate=float(trainer_cfg.get("learning_rate", 2e-2)),
            budget_targets={
                "tool_calls": float(constraints_cfg.get("tool_calls_budget", 8.0)),
                "output_tokens": float(constraints_cfg.get("output_tokens_budget", 1200.0)),
                "latency_ms": float(constraints_cfg.get("latency_ms_budget", 15000.0)),
            },
            counterfactual_k=int(method_cfg.get("counterfactual_k", 6)),
            intervention_types=[
                str(x) for x in method_cfg.get("intervention_types", ["delete", "truncate", "swap"])
            ],
            delete_block_size=int(method_cfg.get("delete_block_size", 2)),
            credit_normalization=str(method_cfg.get("credit_normalization", "signed")),
            cost_normalization=str(method_cfg.get("cost_normalization", "signed")),
            reward_mode=str(method_cfg.get("reward_mode", "mixed")),
            reward_blend_alpha=float(method_cfg.get("reward_blend_alpha", 0.7)),
            failure_reward_floor=float(method_cfg.get("failure_reward_floor", -0.01)),
            action_space=list(action_space),
            cf_cache_size=int(method_cfg.get("cf_cache_size", 20000)),
        ),
        seed=seed,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "results" / f"{full_cfg.get('experiment_name', 'c3rl_main')}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    eval_path = output_dir / "eval.jsonl"
    ckpt_path = output_dir / "checkpoint.json"

    rng = random.Random(seed + 7)
    for step in range(1, total_updates + 1):
        batch = [train_tasks[rng.randrange(len(train_tasks))] for _ in range(batch_size)]
        rollouts = []
        for task in batch:
            rollout = runner.run_episode(task)
            rollout["task"] = task
            rollouts.append(rollout)
        if step <= baseline_warmup_updates:
            policy.update_from_rollouts(
                rollouts=rollouts,
                learning_rate=float(trainer_cfg.get("learning_rate", 2e-2)),
                cost_penalty=float(trainer_cfg.get("cost_penalty", 0.02)),
            )
            c3_metrics = c3.train_step(rollouts, policy=policy, use_c3=False)
            phase = "A_baseline"
        else:
            c3_metrics = c3.train_step(rollouts, policy=policy, use_c3=True)
            phase = "C3_upgrade"

        row = {"step": step, "phase": phase, **c3_metrics}
        _write_jsonl(metrics_path, row)

        if step % eval_interval == 0:
            eval_rollouts = []
            for task in eval_tasks:
                rollout = runner.run_episode(task)
                rollout["task"] = task
                eval_rollouts.append(rollout)
            eval_metrics = _avg_rollout_metrics(eval_rollouts)
            _write_jsonl(eval_path, {"step": step, **eval_metrics})

        if step % save_interval == 0 or step == total_updates:
            ckpt = {
                "step": step,
                "policy_state": policy.state_dict(),
                "c3_state": c3.state_dict(),
                "config": full_cfg,
            }
            ckpt_path.write_text(json.dumps(ckpt, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = {
        "experiment_name": full_cfg.get("experiment_name", "c3rl_main"),
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "output_dir": str(output_dir),
        "checkpoint": str(ckpt_path),
        "baseline_warmup_updates": baseline_warmup_updates,
        "final_policy_skill": policy.skill,
        "final_dual_vars": c3.dual_vars,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
