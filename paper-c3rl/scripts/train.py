#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS_ROOT = REPO_ROOT / "paper-c3rl" / "methods"
for p in (REPO_ROOT, METHODS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.experiment_tracking import ExperimentConfig, ExperimentTracker, RunMetrics
from shared.hf import (
    DistInfo,
    all_reduce_mean,
    build_code_prompt,
    build_sql_prompt,
    broadcast_object,
    extract_python_code,
    init_distributed,
    load_jsonl,
    weighted_rl_loss,
)
from shared.hf.distributed import barrier
from shared.hf.prompts import DEFAULT_SQL_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

from c3rl import (
    AdvantageMapper,
    C3Config,
    C3Trainer,
    CostCreditEstimator,
    CounterfactualExecutor,
    CounterfactualGenerator,
    CreditEstimator,
)
from c3rl.train_utils import (
    cf_result_reward,
    clone_task_with_group_id,
    compute_grpo_advantages,
    guess_lora_target_modules,
    pass_at_k_from_groups,
    to_float_list,
    trajectory_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C3-RL (HF backend)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "toy"])
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to load config files. Install with `pip install pyyaml`.")
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


def _resolve_base_config_path(config_path: Path, ref: str) -> Path:
    ref_path = Path(ref)
    candidates = [
        (config_path.parent / ref_path).resolve(),
        (REPO_ROOT / ref_path).resolve(),
        (REPO_ROOT / "agent-rl-core" / "configs" / "base.yaml").resolve(),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _build_prompt(
    *,
    env_type: str,
    observation: str,
    tokenizer: Any,
    use_chat_template: bool,
    system_prompt: str,
) -> str:
    if env_type == "code":
        return build_code_prompt(
            observation,
            tokenizer,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
    return build_sql_prompt(
        observation,
        tokenizer,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )


def _extract_completion_ids(
    seq_ids: List[int],
    *,
    prompt_len: int,
    eos_token_id: Optional[int],
) -> List[int]:
    end = len(seq_ids)
    if eos_token_id is not None:
        try:
            end = seq_ids.index(int(eos_token_id), prompt_len) + 1
        except ValueError:
            pass
    comp = seq_ids[prompt_len:end]
    if not comp:
        if eos_token_id is not None:
            return [int(eos_token_id)]
        return [seq_ids[-1]]
    return comp


def _evaluate_pass_budget(
    *,
    model: Any,
    device: Any,
    tokenizer: Any,
    env: Any,
    eval_records: List[Dict[str, Any]],
    env_type: str,
    use_chat_template: bool,
    system_prompt: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    eval_tasks: int,
    c3: C3Trainer,
) -> Dict[str, float]:
    import torch

    if not eval_records:
        return {
            "pass_at_1": 0.0,
            "success_rate": 0.0,
            "success_under_budget": 0.0,
            "violation_rate": 0.0,
            "avg_cost_tool_calls": 0.0,
            "avg_cost_output_tokens": 0.0,
            "avg_cost_latency_ms": 0.0,
        }

    subset_n = min(len(eval_records), max(1, int(eval_tasks)))
    subset = eval_records[:subset_n]

    successes = 0.0
    successes_under_budget = 0.0
    violation_sum = 0.0
    cost_sum = {"tool_calls": 0.0, "output_tokens": 0.0, "latency_ms": 0.0}

    model.eval()
    with torch.no_grad():
        for i, rec in enumerate(subset):
            gid = f"{rec.get('task_id', 'eval')}::eval::{i}"
            task = clone_task_with_group_id(rec, group_id=gid)
            obs = env.reset(task)
            prompt = _build_prompt(
                env_type=env_type,
                observation=obs.content,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
            )
            enc = tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            )
            in_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            p_len = int(attn.sum(dim=1).item())

            seq = model.generate(
                input_ids=in_ids,
                attention_mask=attn,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
            ids = [int(x) for x in seq.tolist()]
            comp_ids = _extract_completion_ids(ids, prompt_len=p_len, eos_token_id=tokenizer.eos_token_id)
            completion_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
            content = extract_python_code(completion_text) if env_type == "code" else completion_text

            env.reset(task)
            if env_type == "code":
                env.step(Action(ActionType.CODE_WRITE, content, metadata={"logprob": 0.0}))
                env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
            else:
                env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query", metadata={"logprob": 0.0}))

            traj = copy.deepcopy(env.get_trajectory())
            costs = c3.trajectory_costs(traj)
            viol = c3.violation_rate(costs)

            success = float(traj.r_final > 0.5)
            successes += success
            violation_sum += float(viol)
            if success > 0.5 and viol <= 1e-12:
                successes_under_budget += 1.0
            for k in cost_sum:
                cost_sum[k] += float(costs.get(k, 0.0))

    model.train()

    n = float(len(subset))
    return {
        "pass_at_1": successes / n,
        "success_rate": successes / n,
        "success_under_budget": successes_under_budget / n,
        "violation_rate": violation_sum / n,
        "avg_cost_tool_calls": cost_sum["tool_calls"] / n,
        "avg_cost_output_tokens": cost_sum["output_tokens"] / n,
        "avg_cost_latency_ms": cost_sum["latency_ms"] / n,
    }


def run_hf(full_cfg: Dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency `peft` for LoRA. Install requirements first.\n"
            f"Original error: {e}"
        )

    seed = int(full_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    method_cfg = dict(full_cfg.get("method", {}))
    trainer_cfg = dict(full_cfg.get("trainer", {}))
    model_cfg = dict(full_cfg.get("model", {}))
    env_cfg = dict(full_cfg.get("environment", {}))
    constraints_cfg = dict(full_cfg.get("constraints", {}))
    eval_cfg = dict(full_cfg.get("evaluation", {}))

    model_path = os.getenv("MODEL_PATH", model_cfg.get("path") or model_cfg.get("name"))
    if not model_path or str(model_path).strip().lower().endswith("placeholder"):
        raise ValueError(
            "Missing valid model path. Set MODEL_PATH or model.path in config "
            "(current value looks like placeholder)."
        )

    train_dataset = os.getenv("TRAIN_DATA", env_cfg.get("train_dataset"))
    eval_dataset = os.getenv("EVAL_DATA", env_cfg.get("eval_dataset"))
    if not train_dataset:
        raise ValueError("Missing train dataset. Set TRAIN_DATA or environment.train_dataset in config.")

    max_steps = int(os.getenv("MAX_STEPS", trainer_cfg.get("total_updates", 2000)))
    batch_size = int(os.getenv("BATCH_SIZE", trainer_cfg.get("batch_size", 2)))
    eval_interval = int(os.getenv("EVAL_INTERVAL", trainer_cfg.get("eval_interval", 200)))
    log_interval = int(os.getenv("LOG_INTERVAL", trainer_cfg.get("log_interval", 20)))
    save_interval = int(os.getenv("SAVE_INTERVAL", trainer_cfg.get("save_interval", 200)))
    learning_rate = float(os.getenv("LEARNING_RATE", trainer_cfg.get("learning_rate", 5e-6)))

    num_rollouts_per_prompt = int(
        os.getenv("NUM_ROLLOUTS_PER_PROMPT", trainer_cfg.get("num_rollouts_per_prompt", 4))
    )
    temperature = float(os.getenv("TEMPERATURE", trainer_cfg.get("temperature", 1.0)))
    top_p = float(os.getenv("TOP_P", trainer_cfg.get("top_p", 1.0)))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", trainer_cfg.get("max_new_tokens", 192)))
    max_prompt_tokens = int(os.getenv("MAX_PROMPT_TOKENS", trainer_cfg.get("max_prompt_tokens", 1024)))
    max_trajectory_length = int(os.getenv("MAX_TRAJECTORY_LENGTH", env_cfg.get("max_trajectory_length", 8)))
    task_timeout_seconds = float(os.getenv("TASK_TIMEOUT_SECONDS", env_cfg.get("task_timeout_seconds", 8.0)))
    show_tests = _bool_env("SHOW_TESTS", bool(env_cfg.get("show_tests", True)))
    eval_tasks = int(os.getenv("EVAL_TASKS", eval_cfg.get("eval_tasks", 64)))
    max_train_samples = env_cfg.get("max_train_samples")
    max_eval_samples = env_cfg.get("max_eval_samples")

    reward_mode = str(os.getenv("REWARD_MODE", method_cfg.get("reward_mode", "mixed")))
    reward_blend_alpha = float(os.getenv("REWARD_BLEND_ALPHA", method_cfg.get("reward_blend_alpha", 0.7)))
    failure_reward_floor = float(os.getenv("FAILURE_REWARD_FLOOR", method_cfg.get("failure_reward_floor", -0.01)))
    flat_group_fallback = str(method_cfg.get("flat_group_fallback", "batch_centered"))

    use_counterfactual_credit = _bool_env(
        "USE_COUNTERFACTUAL_CREDIT",
        bool(method_cfg.get("use_counterfactual_credit", True)),
    )
    prioritize_high_value_cf = _bool_env(
        "PRIORITIZE_HIGH_VALUE_CF",
        bool(method_cfg.get("prioritize_high_value_cf", False)),
    )
    counterfactual_k = int(os.getenv("COUNTERFACTUAL_K", method_cfg.get("counterfactual_k", 6)))
    intervention_types = [str(x) for x in method_cfg.get("intervention_types", ["delete", "truncate", "swap"])]
    credit_normalization = str(method_cfg.get("credit_normalization", "signed"))
    cost_normalization = str(method_cfg.get("cost_normalization", "signed"))

    dual_lr = float(os.getenv("DUAL_LR", method_cfg.get("dual_lr", 5e-4)))
    dual_warmup_steps = int(os.getenv("DUAL_WARMUP_STEPS", method_cfg.get("dual_warmup_steps", 200)))
    lambda_max = float(os.getenv("LAMBDA_MAX", method_cfg.get("lambda_max", 10.0)))
    fallback_to_adv_when_zero_credit = _bool_env(
        "FALLBACK_TO_ADV_WHEN_ZERO_CREDIT",
        bool(method_cfg.get("fallback_to_adv_when_zero_credit", True)),
    )
    zero_credit_threshold = float(os.getenv("ZERO_CREDIT_THRESHOLD", method_cfg.get("zero_credit_threshold", 1e-8)))

    use_lora = _bool_env("USE_LORA", bool(model_cfg.get("lora", True)))
    lora_rank = int(os.getenv("LORA_RANK", model_cfg.get("lora_rank", 64)))
    dtype_name = str(os.getenv("DTYPE", model_cfg.get("dtype", "bf16")))
    grad_clip = float(os.getenv("GRAD_CLIP", trainer_cfg.get("grad_clip", 1.0)))
    require_cuda = _bool_env("REQUIRE_CUDA", True)
    trust_remote_code = _bool_env("TRUST_REMOTE_CODE", False)
    use_chat_template = _bool_env("USE_CHAT_TEMPLATE", True)
    gradient_checkpointing = _bool_env("GRADIENT_CHECKPOINTING", True)
    sync_eval_and_save = _bool_env("SYNC_EVAL_AND_SAVE", True)
    rl_microbatch_size = int(os.getenv("RL_MICROBATCH_SIZE", trainer_cfg.get("rl_microbatch_size", 0)))

    system_prompt = os.getenv("SYSTEM_PROMPT")
    env_type = str(env_cfg.get("env_type", full_cfg.get("env_type", "code")))
    output_dir = str(full_cfg.get("output_dir", full_cfg.get("logging", {}).get("save_dir", "paper-c3rl/results")))
    experiment_name = str(os.getenv("EXPERIMENT_NAME", full_cfg.get("experiment_name", "c3rl_main")))

    budget_targets = {
        "tool_calls": float(constraints_cfg.get("tool_calls_budget", 8.0)),
        "output_tokens": float(constraints_cfg.get("output_tokens_budget", 1200.0)),
        "latency_ms": float(constraints_cfg.get("latency_ms_budget", 15000.0)),
    }

    dist_info: DistInfo = init_distributed()
    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        if dist_info.is_rank0:
            print("[ERROR] CUDA not available but REQUIRE_CUDA=1", flush=True)
        raise SystemExit(2)
    if has_cuda and dist_info.local_rank >= torch.cuda.device_count():
        raise SystemExit(
            f"[ERROR] local_rank={dist_info.local_rank} but visible CUDA devices={torch.cuda.device_count()}"
        )
    if has_cuda:
        torch.cuda.set_device(dist_info.local_rank)
    device = torch.device(f"cuda:{dist_info.local_rank}" if has_cuda else "cpu")

    tracker = None
    exp_dir = ""
    if dist_info.is_rank0:
        tracker = ExperimentTracker(
            ExperimentConfig(
                experiment_name=experiment_name,
                project="paper_c3rl",
                description="Counterfactual Cost-Credit Constrained RL (HF)",
                model_name=str(model_path),
                use_lora=use_lora,
                lora_rank=lora_rank,
                algorithm="C3_GRPO",
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_steps=max_steps,
                env_type=env_type,
                max_trajectory_length=max_trajectory_length,
                use_counterfactual_credit=use_counterfactual_credit,
                counterfactual_k=counterfactual_k,
                intervention_types=intervention_types,
                seed=seed,
                extra={
                    "backend": "hf",
                    "reward_mode": reward_mode,
                    "reward_blend_alpha": reward_blend_alpha,
                    "failure_reward_floor": failure_reward_floor,
                    "flat_group_fallback": flat_group_fallback,
                    "credit_normalization": credit_normalization,
                    "cost_normalization": cost_normalization,
                    "prioritize_high_value_cf": prioritize_high_value_cf,
                    "dual_lr": dual_lr,
                    "dual_warmup_steps": dual_warmup_steps,
                    "lambda_max": lambda_max,
                    "budget_targets": budget_targets,
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                },
            ),
            base_dir=output_dir,
        )
        exp_dir = str(tracker.experiment_dir)

    exp_dir = str(broadcast_object(exp_dir, src=0, dist_info=dist_info))
    barrier(dist_info=dist_info)

    env = (
        CodeEnv(
            EnvConfig(
                name="code",
                max_steps=max_trajectory_length,
                seed=seed + dist_info.rank,
                extra={
                    "show_tests": bool(show_tests),
                    "default_timeout": float(task_timeout_seconds),
                    "cap_task_timeout": True,
                },
            )
        )
        if env_type == "code"
        else SQLEnv(EnvConfig(name="sql", max_steps=max_trajectory_length, seed=seed + dist_info.rank))
    )

    cf_generator = CounterfactualGenerator(
        intervention_types=intervention_types,
        k=counterfactual_k,
        prioritize_high_value=prioritize_high_value_cf,
        seed=seed + 17 * dist_info.rank,
    )
    cf_executor = CounterfactualExecutor(env, use_cache=True, seed=seed + 19 * dist_info.rank)
    success_credit_estimator = CreditEstimator(normalization=credit_normalization)
    cost_credit_estimator = CostCreditEstimator(normalization=cost_normalization)
    advantage_mapper = AdvantageMapper(level="step")
    c3 = C3Trainer(
        C3Config(
            dual_lr=dual_lr,
            dual_warmup_steps=dual_warmup_steps,
            lambda_max=lambda_max,
            budget_targets=budget_targets,
            credit_normalization=credit_normalization,
            cost_normalization=cost_normalization,
            fallback_to_adv_when_zero_credit=fallback_to_adv_when_zero_credit,
            zero_credit_threshold=zero_credit_threshold,
        )
    )

    train_records = load_jsonl(Path(train_dataset), max_samples=max_train_samples)
    if not train_records:
        raise RuntimeError(f"Empty train dataset: {train_dataset}")
    eval_records = load_jsonl(Path(eval_dataset), max_samples=max_eval_samples) if eval_dataset else []
    if not eval_records:
        eval_records = train_records

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    added_pad = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        added_pad = True

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    if added_pad:
        model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        targets = guess_lora_target_modules(model)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)
        if dist_info.is_rank0 and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    model = model.to(device)

    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "config") and hasattr(raw_model.config, "use_cache"):
        raw_model.config.use_cache = False
    if gradient_checkpointing and hasattr(raw_model, "gradient_checkpointing_enable"):
        raw_model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        if hasattr(raw_model, "enable_input_require_grads"):
            try:
                raw_model.enable_input_require_grads()
            except Exception:
                pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found; check LoRA settings.")
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    sys_prompt = system_prompt or (DEFAULT_SYSTEM_PROMPT if env_type == "code" else DEFAULT_SQL_SYSTEM_PROMPT)
    rng = random.Random(seed + 2026 + 7919 * dist_info.rank)
    wall_start = time.time()

    if dist_info.is_rank0:
        print("=" * 72, flush=True)
        print("C3-RL HF Training", flush=True)
        print("=" * 72, flush=True)
        print(f"Model: {model_path}", flush=True)
        print(f"Train dataset: {train_dataset} ({len(train_records)} samples)", flush=True)
        print(f"Eval dataset: {eval_dataset if eval_dataset else train_dataset} ({len(eval_records)} samples)", flush=True)
        print(f"Max steps: {max_steps}", flush=True)
        print(f"Batch size: {batch_size}, rollouts/prompt: {num_rollouts_per_prompt}", flush=True)
        print(f"World size: {dist_info.world_size}", flush=True)
        print(f"Budgets: {budget_targets}", flush=True)
        print("=" * 72, flush=True)

    for step in range(1, max_steps + 1):
        prompt_tasks: List[Dict[str, Any]] = []
        prompt_texts: List[str] = []

        for i in range(batch_size):
            rec = rng.choice(train_records)
            base_id = str(rec.get("task_id", f"idx{i}"))
            gid = f"{base_id}::s{step}::r{dist_info.rank}::p{i}"
            task = clone_task_with_group_id(rec, group_id=gid)
            prompt_tasks.append(task)

            obs = env.reset(task)
            prompt_texts.append(
                _build_prompt(
                    env_type=env_type,
                    observation=obs.content,
                    tokenizer=tokenizer,
                    use_chat_template=use_chat_template,
                    system_prompt=sys_prompt,
                )
            )

        enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_lens_base = [int(x) for x in attention_mask.sum(dim=1).tolist()]

        model.eval()
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_rollouts_per_prompt,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        model.train()

        r = int(num_rollouts_per_prompt)
        n_samples = len(prompt_tasks) * r
        expanded_tasks = [prompt_tasks[i // r] for i in range(n_samples)]
        prompt_lens = [prompt_lens_base[i // r] for i in range(n_samples)]

        trajectories: List[Trajectory] = []
        for seq, task, p_len in zip(sequences, expanded_tasks, prompt_lens):
            ids = [int(x) for x in seq.tolist()]
            completion_ids = _extract_completion_ids(ids, prompt_len=p_len, eos_token_id=tokenizer.eos_token_id)
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            content = extract_python_code(completion_text) if env_type == "code" else completion_text

            env.reset(task)
            if env_type == "code":
                env.step(Action(ActionType.CODE_WRITE, content, metadata={"logprob": 0.0}))
                env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
            else:
                env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query", metadata={"logprob": 0.0}))
            trajectories.append(copy.deepcopy(env.get_trajectory()))

        reward_values = [
            trajectory_reward(
                t,
                mode=reward_mode,
                blend_alpha=reward_blend_alpha,
                failure_reward_floor=failure_reward_floor,
            )
            for t in trajectories
        ]
        advantages = compute_grpo_advantages(
            trajectories,
            reward_values,
            flat_group_fallback=flat_group_fallback,
        )

        weights: List[float] = []
        credit_weights: List[float] = []
        credit_spreads: List[float] = []
        cf_generated = 0
        cf_valid = 0

        for traj, adv in zip(trajectories, advantages):
            w = float(adv)
            c3_weight = 1.0
            if use_counterfactual_credit and cf_generator.should_generate_counterfactuals(traj):
                interventions = cf_generator.generate(traj)
                cf_results = cf_executor.batch_execute(traj, interventions)
                cf_generated += len(interventions)
                cf_valid += sum(1 for cf in cf_results if cf.is_valid)

                success_map = success_credit_estimator.estimate(
                    traj,
                    cf_results,
                    base_reward=trajectory_reward(
                        traj,
                        mode=reward_mode,
                        blend_alpha=reward_blend_alpha,
                        failure_reward_floor=failure_reward_floor,
                    ),
                    cf_reward_fn=lambda cf: cf_result_reward(
                        cf,
                        mode=reward_mode,
                        blend_alpha=reward_blend_alpha,
                        failure_reward_floor=failure_reward_floor,
                    ),
                )
                credit_spreads.append(float(success_map.spread))

                success_steps = to_float_list(advantage_mapper.map_to_step_advantages(traj, success_map))
                cost_maps = cost_credit_estimator.estimate_multi(
                    traj,
                    cf_results,
                    channels=list(c3.dual_vars.keys()),
                )
                cost_steps = {
                    key: [float(v) for v in cost_map.normalized_credits]
                    for key, cost_map in cost_maps.items()
                }
                c3_steps = c3.combine_step_signals(
                    success_step_credits=success_steps,
                    cost_step_credits=cost_steps,
                )
                lp_steps = [idx for idx, s in enumerate(traj.steps) if s.logprob is not None]
                w, c3_weight = c3.map_to_rollout_weight(
                    trajectory_advantage=w,
                    step_signal=c3_steps,
                    logprob_step_indices=lp_steps,
                )

            weights.append(float(w))
            credit_weights.append(float(c3_weight))

        batch_costs = c3.batch_average_costs(trajectories)
        batch_costs = {
            key: all_reduce_mean(float(value), dist_info=dist_info)
            for key, value in batch_costs.items()
        }
        c3.update_duals(batch_costs)

        loss, rl_meta = weighted_rl_loss(
            model,
            sequences.to(device),
            prompt_lens=prompt_lens,
            weights=weights,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            micro_batch_size=rl_microbatch_size,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if dist_info.distributed:
            import torch.distributed as dist

            for p in trainable_params:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= float(dist_info.world_size)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            success_rate = sum(float(t.r_final) for t in trajectories) / max(1.0, float(len(trajectories)))
            pass_at_1 = pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = pass_at_k_from_groups(trajectories, k=r)
            avg_len = sum(float(t.length) for t in trajectories) / max(1.0, float(len(trajectories)))
            mean_reward = sum(reward_values) / max(1.0, float(len(reward_values)))
            nz_weight_ratio = sum(1 for w in weights if abs(w) > 1e-8) / max(1.0, float(len(weights)))
            mean_weight = sum(weights) / max(1.0, float(len(weights)))
            mean_credit_weight = sum(credit_weights) / max(1.0, float(len(credit_weights)))

            traj_cost_list = [c3.trajectory_costs(t) for t in trajectories]
            mean_violation = sum(c3.violation_rate(c) for c in traj_cost_list) / max(1.0, float(len(traj_cost_list)))
            success_under_budget = sum(
                1.0
                for t, c in zip(trajectories, traj_cost_list)
                if (float(t.r_final) > 0.5 and c3.violation_rate(c) <= 1e-12)
            ) / max(1.0, float(len(traj_cost_list)))

            loss_mean = all_reduce_mean(float(loss.item()), dist_info=dist_info)
            succ_mean = all_reduce_mean(float(success_rate), dist_info=dist_info)
            p1_mean = all_reduce_mean(float(pass_at_1), dist_info=dist_info)
            pk_mean = all_reduce_mean(float(pass_at_k), dist_info=dist_info)
            reward_mean = all_reduce_mean(float(mean_reward), dist_info=dist_info)

            if tracker:
                tracker.log_metrics(
                    RunMetrics(
                        step=step,
                        train_loss=float(loss_mean),
                        success_rate=float(succ_mean),
                        pass_at_1=float(p1_mean),
                        pass_at_k={r: float(pk_mean)},
                        avg_trajectory_length=float(avg_len),
                        wall_time=float(time.time() - wall_start),
                        avg_credit_spread=float(
                            (sum(credit_spreads) / float(len(credit_spreads))) if credit_spreads else 0.0
                        ),
                        extra={
                            "backend": "hf",
                            "world_size": dist_info.world_size,
                            "mean_reward": float(reward_mean),
                            "mean_weight": float(mean_weight),
                            "mean_credit_weight": float(mean_credit_weight),
                            "mean_abs_weight": float(rl_meta.get("mean_abs_weight", 0.0)),
                            "mean_nll": float(rl_meta.get("mean_nll", 0.0)),
                            "rl_microbatch": int(rl_meta.get("rl_microbatch", 0)),
                            "nonzero_weight_ratio": float(nz_weight_ratio),
                            "cf_generated_per_traj": float(cf_generated) / max(1.0, float(len(trajectories))),
                            "cf_valid_per_traj": float(cf_valid) / max(1.0, float(len(trajectories))),
                            "cf_valid_ratio": float(cf_valid) / max(1.0, float(cf_generated)) if cf_generated > 0 else 0.0,
                            "avg_cost_tool_calls": float(batch_costs["tool_calls"]),
                            "avg_cost_output_tokens": float(batch_costs["output_tokens"]),
                            "avg_cost_latency_ms": float(batch_costs["latency_ms"]),
                            "violation_rate": float(mean_violation),
                            "success_under_budget": float(success_under_budget),
                            "lambda_tool_calls": float(c3.dual_vars.get("tool_calls", 0.0)),
                            "lambda_output_tokens": float(c3.dual_vars.get("output_tokens", 0.0)),
                            "lambda_latency_ms": float(c3.dual_vars.get("latency_ms", 0.0)),
                        },
                    )
                )
                tracker.log_event(
                    "c3_train",
                    "logged train metrics",
                    {
                        "step": step,
                        "loss": float(loss_mean),
                        "success_rate": float(succ_mean),
                        "mean_reward": float(reward_mean),
                        "mean_violation": float(mean_violation),
                    },
                )

            if dist_info.is_rank0:
                print(
                    f"[step {step}] loss={loss_mean:.4f} succ={succ_mean:.4f} p@1={p1_mean:.4f} "
                    f"cost(tool/tok/ms)=({batch_costs['tool_calls']:.2f}/{batch_costs['output_tokens']:.1f}/{batch_costs['latency_ms']:.1f}) "
                    f"lambda=({c3.dual_vars.get('tool_calls', 0.0):.3f}/{c3.dual_vars.get('output_tokens', 0.0):.3f}/{c3.dual_vars.get('latency_ms', 0.0):.3f})",
                    flush=True,
                )

        if sync_eval_and_save:
            barrier(dist_info=dist_info)

        if tracker and dist_info.is_rank0 and (step % eval_interval == 0):
            eval_metrics = _evaluate_pass_budget(
                model=model,
                device=device,
                tokenizer=tokenizer,
                env=env,
                eval_records=eval_records,
                env_type=env_type,
                use_chat_template=use_chat_template,
                system_prompt=sys_prompt,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                eval_tasks=eval_tasks,
                c3=c3,
            )
            tracker.log_event("eval", "C3 evaluation", {"step": step, "metrics": eval_metrics})
            print(f"[eval step {step}] {json.dumps(eval_metrics, ensure_ascii=True)}", flush=True)

        if sync_eval_and_save:
            barrier(dist_info=dist_info)

        if tracker and dist_info.is_rank0 and save_interval and (step % save_interval == 0):
            ckpt_dir = Path(exp_dir) / "checkpoints" / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            (ckpt_dir / "c3_state.json").write_text(
                json.dumps(c3.state_dict(), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            tracker.log_event("checkpoint", "saved checkpoint", {"step": step, "path": str(ckpt_dir)})

        if sync_eval_and_save:
            barrier(dist_info=dist_info)

    if tracker and dist_info.is_rank0:
        final_ckpt = Path(exp_dir) / "checkpoints" / "final"
        final_ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_ckpt))
        tokenizer.save_pretrained(str(final_ckpt))
        (final_ckpt / "c3_state.json").write_text(
            json.dumps(c3.state_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        tracker.log_event("checkpoint", "saved final checkpoint", {"path": str(final_ckpt)})
        tracker.finalize()
        summary = {
            "experiment_name": experiment_name,
            "output_dir": exp_dir,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset if eval_dataset else train_dataset,
            "backend": "hf",
            "final_checkpoint": str(final_ckpt),
            "dual_vars": c3.dual_vars,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)

    barrier(dist_info=dist_info)


def run_toy_notice(full_cfg: Dict[str, Any]) -> None:
    experiment_name = str(full_cfg.get("experiment_name", "c3rl_main"))
    print(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "backend": "toy",
                "status": "disabled",
                "message": "Toy backend is intentionally not used for paper-level validation. Use HF backend.",
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    core_cfg_ref = str(cfg.get("core", {}).get("base_config", "../agent-rl-core/configs/base.yaml"))
    core_cfg_path = _resolve_base_config_path(args.config.resolve(), core_cfg_ref)
    base_cfg = _load_yaml(core_cfg_path)
    full_cfg = _merge_dict(base_cfg, cfg)

    backend = str(args.backend).lower()
    if backend == "toy":
        run_toy_notice(full_cfg)
        return
    run_hf(full_cfg)


if __name__ == "__main__":
    main()
