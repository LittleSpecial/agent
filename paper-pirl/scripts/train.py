#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.experiment_tracking import ExperimentConfig, ExperimentTracker, RunMetrics
from shared.hf import (
    DistInfo,
    all_reduce_mean,
    build_attention_and_labels,
    build_code_prompt,
    build_sql_prompt,
    broadcast_object,
    extract_python_code,
    init_distributed,
    load_jsonl,
    per_sample_nll,
)
from shared.hf.distributed import barrier
from shared.hf.prompts import DEFAULT_SQL_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PIRL (HF backend)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "toy"])
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to load config files. Install with `pip install pyyaml`."
        )
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


def _clone_task_with_group_id(task: Dict[str, Any], *, group_id: str) -> Dict[str, Any]:
    cloned = dict(task)
    base_task_id = str(cloned.get("task_id", "unknown"))
    cloned["task_id"] = group_id
    meta = dict(cloned.get("metadata") or {})
    meta.setdefault("base_task_id", base_task_id)
    meta.setdefault("group_id", group_id)
    cloned["metadata"] = meta
    return cloned


def _clamp01(x: float) -> float:
    v = float(x)
    if not (v == v):
        return 0.0
    return max(0.0, min(1.0, v))


def _trajectory_reward(
    trajectory: Trajectory,
    *,
    mode: str,
    blend_alpha: float,
    failure_reward_floor: float,
) -> float:
    binary = _clamp01(float(trajectory.r_final))
    score = _clamp01(float(getattr(trajectory.verifier_info, "score", binary)))
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


def _compute_grpo_advantages(
    trajectories: List[Trajectory],
    rewards: List[float],
    *,
    flat_group_fallback: str = "raw",
) -> List[float]:
    if len(trajectories) != len(rewards):
        raise ValueError("trajectories/rewards length mismatch")
    rewards = [float(r) for r in rewards]
    batch_mean = statistics.fmean(rewards) if rewards else 0.0
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


def _pass_at_k_from_groups(trajectories: List[Trajectory], k: int) -> float:
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


def _guess_lora_target_modules(model: Any) -> List[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    picked = [c for c in candidates if c in names]
    return picked if picked else ["q_proj", "k_proj", "v_proj", "o_proj"]


def _strength_base(strength: str) -> float:
    mapping = {"low": 0.15, "medium": 0.3, "high": 0.5}
    if strength in mapping:
        return mapping[strength]
    try:
        return max(0.0, min(1.0, float(strength)))
    except Exception:
        return 0.3


def _scheduled_strength(*, step: int, max_steps: int, base: float, curriculum: bool) -> float:
    if not curriculum:
        return base
    progress = min(1.0, float(step) / max(1.0, float(max_steps)))
    return base * (0.25 + 0.75 * progress)


def _randomize_interface_text(text: str, *, strength: float, rng: random.Random, variant: int) -> str:
    if strength <= 1e-8:
        return text
    out = text
    substitutions = [
        ("Write", "Implement"),
        ("Return", "Output"),
        ("Python", "py"),
        ("function", "routine"),
        ("tests", "checks"),
        ("code", "solution"),
        ("must", "should"),
    ]
    rng_local = random.Random(rng.randint(0, 10_000_000) + 1337 * variant)
    rng_local.shuffle(substitutions)
    n_subs = max(1, int(round(strength * len(substitutions))))
    for src, dst in substitutions[:n_subs]:
        if rng_local.random() < 0.8:
            out = out.replace(src, dst)

    lines = out.splitlines()
    if len(lines) > 6 and strength >= 0.2:
        head = lines[:2]
        body = lines[2:]
        rng_local.shuffle(body)
        keep = int(max(3, min(len(body), round(len(body) * (0.5 + 0.4 * strength)))))
        lines = head + body[:keep]
        out = "\n".join(lines)

    if strength >= 0.35:
        suffixes = [
            "Interface notes: field names may vary but semantics stay unchanged.",
            "Reminder: rely on behavior, not literal formatting.",
            "API wording can shift across versions; keep logic invariant.",
        ]
        out = out + "\n\n" + suffixes[variant % len(suffixes)]
    return out


def _build_eval_strengths(ood_levels: List[str], base_strength: float) -> Dict[str, float]:
    strengths: Dict[str, float] = {}
    for level in ood_levels:
        if level == "iid":
            strengths[level] = 0.0
        elif level == "ood_easy":
            strengths[level] = max(0.1, 0.6 * base_strength)
        elif level == "ood_hard":
            strengths[level] = min(1.0, 1.2 * base_strength)
        else:
            strengths[level] = base_strength
    return strengths


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


def _tokenize_prompt_ids(tokenizer: Any, text: str, max_length: int) -> List[int]:
    ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
    return [int(x) for x in ids]


def _pad_sequences(token_lists: List[List[int]], *, pad_id: int, device: Any):
    import torch

    max_len = max(len(x) for x in token_lists)
    rows = []
    for x in token_lists:
        row = x + [pad_id] * (max_len - len(x))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


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


def _evaluate_levels(
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
    ood_levels: List[str],
    base_strength: float,
    seed: int,
    eval_tasks_per_level: int,
) -> Dict[str, float]:
    import torch

    strengths = _build_eval_strengths(ood_levels, base_strength)
    rng = random.Random(seed)
    metrics: Dict[str, float] = {}
    if not eval_records:
        for level in ood_levels:
            metrics[level] = 0.0
        return metrics

    subset_n = min(len(eval_records), max(1, eval_tasks_per_level))
    subset = rng.sample(eval_records, subset_n) if len(eval_records) > subset_n else list(eval_records)
    model.eval()
    with torch.no_grad():
        for level in ood_levels:
            strength = strengths[level]
            succ = 0.0
            for i, rec in enumerate(subset):
                gid = f"{rec.get('task_id', 'eval')}::eval::{level}::{i}"
                task = _clone_task_with_group_id(rec, group_id=gid)
                obs = env.reset(task)
                obs_text = _randomize_interface_text(obs.content, strength=strength, rng=rng, variant=i % 3)
                prompt = _build_prompt(
                    env_type=env_type,
                    observation=obs_text,
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
                content = extract_python_code(completion_text)
                env.reset(task)
                if env_type == "code":
                    env.step(Action(ActionType.CODE_WRITE, content))
                    _, r_eval, _, _ = env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
                else:
                    _, r_eval, _, _ = env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query"))
                succ += float(r_eval)
            metrics[level] = succ / max(1.0, float(subset_n))
    model.train()
    return metrics


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
    eval_cfg = dict(full_cfg.get("evaluation", {}))

    model_path = os.getenv("MODEL_PATH", model_cfg.get("path") or model_cfg.get("name"))
    if not model_path:
        raise ValueError("Missing model path. Set MODEL_PATH or model.path in config.")
    train_dataset = os.getenv("TRAIN_DATA", env_cfg.get("train_dataset"))
    eval_dataset = os.getenv("EVAL_DATA", env_cfg.get("eval_dataset"))
    if not train_dataset:
        raise ValueError("Missing train dataset. Set TRAIN_DATA or environment.train_dataset in config.")

    max_steps = int(os.getenv("MAX_STEPS", trainer_cfg.get("total_updates", 200)))
    batch_size = int(os.getenv("BATCH_SIZE", trainer_cfg.get("batch_size", 4)))
    eval_interval = int(os.getenv("EVAL_INTERVAL", trainer_cfg.get("eval_interval", 50)))
    log_interval = int(os.getenv("LOG_INTERVAL", trainer_cfg.get("log_interval", 10)))
    save_interval = int(os.getenv("SAVE_INTERVAL", trainer_cfg.get("save_interval", 100)))
    learning_rate = float(os.getenv("LEARNING_RATE", trainer_cfg.get("learning_rate", 1e-5)))

    num_rollouts_per_prompt = int(os.getenv("NUM_ROLLOUTS_PER_PROMPT", trainer_cfg.get("num_rollouts_per_prompt", 2)))
    temperature = float(os.getenv("TEMPERATURE", trainer_cfg.get("temperature", 1.0)))
    top_p = float(os.getenv("TOP_P", trainer_cfg.get("top_p", 1.0)))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", trainer_cfg.get("max_new_tokens", 192)))
    max_prompt_tokens = int(os.getenv("MAX_PROMPT_TOKENS", trainer_cfg.get("max_prompt_tokens", 1024)))
    max_trajectory_length = int(os.getenv("MAX_TRAJECTORY_LENGTH", env_cfg.get("max_trajectory_length", 8)))
    eval_tasks_per_level = int(os.getenv("EVAL_TASKS_PER_LEVEL", eval_cfg.get("eval_tasks_per_level", 64)))
    max_train_samples = env_cfg.get("max_train_samples")
    max_eval_samples = env_cfg.get("max_eval_samples")

    reward_mode = str(os.getenv("REWARD_MODE", method_cfg.get("reward_mode", "mixed")))
    reward_blend_alpha = float(os.getenv("REWARD_BLEND_ALPHA", method_cfg.get("reward_blend_alpha", 0.7)))
    failure_reward_floor = float(os.getenv("FAILURE_REWARD_FLOOR", method_cfg.get("failure_reward_floor", -0.01)))
    flat_group_fallback = str(method_cfg.get("flat_group_fallback", "raw"))

    invariance_weight = float(os.getenv("INVARIANCE_WEIGHT", method_cfg.get("invariance_weight", 0.1)))
    randomization_strength = str(os.getenv("RANDOMIZATION_STRENGTH", method_cfg.get("randomization_strength", "medium")))
    curriculum = _bool_env("CURRICULUM", bool(method_cfg.get("curriculum", True)))

    use_lora = _bool_env("USE_LORA", bool(model_cfg.get("lora", True)))
    lora_rank = int(os.getenv("LORA_RANK", model_cfg.get("lora_rank", 64)))
    dtype_name = str(os.getenv("DTYPE", model_cfg.get("dtype", "bf16")))
    grad_clip = float(os.getenv("GRAD_CLIP", trainer_cfg.get("grad_clip", 1.0)))
    require_cuda = _bool_env("REQUIRE_CUDA", True)
    trust_remote_code = _bool_env("TRUST_REMOTE_CODE", False)
    use_chat_template = _bool_env("USE_CHAT_TEMPLATE", True)
    gradient_checkpointing = _bool_env("GRADIENT_CHECKPOINTING", True)
    system_prompt = os.getenv("SYSTEM_PROMPT")
    env_type = str(env_cfg.get("env_type", full_cfg.get("env_type", "code")))
    ood_levels = [str(x) for x in eval_cfg.get("ood_levels", ["iid", "ood_easy", "ood_hard"])]
    output_dir = str(full_cfg.get("output_dir", full_cfg.get("logging", {}).get("save_dir", "./experiments")))
    experiment_name = str(os.getenv("EXPERIMENT_NAME", full_cfg.get("experiment_name", "pirl_main")))

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
                project="paper_pirl",
                description="Prompt/Interface Randomized RL (HF)",
                model_name=str(model_path),
                use_lora=use_lora,
                lora_rank=lora_rank,
                algorithm="PIRL_GRPO",
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_steps=max_steps,
                env_type=env_type,
                max_trajectory_length=max_trajectory_length,
                seed=seed,
                extra={
                    "backend": "hf",
                    "randomization_strength": randomization_strength,
                    "invariance_weight": invariance_weight,
                    "curriculum": curriculum,
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
        CodeEnv(EnvConfig(name="code", max_steps=max_trajectory_length, seed=seed + dist_info.rank, extra={"show_tests": True}))
        if env_type == "code"
        else SQLEnv(EnvConfig(name="sql", max_steps=max_trajectory_length, seed=seed + dist_info.rank))
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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        targets = _guess_lora_target_modules(model)
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
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found; check LoRA settings.")
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    sys_prompt = system_prompt or (DEFAULT_SYSTEM_PROMPT if env_type == "code" else DEFAULT_SQL_SYSTEM_PROMPT)
    base_strength = _strength_base(randomization_strength)
    rng = random.Random(seed + 2026 + 7919 * dist_info.rank)
    wall_start = time.time()

    if dist_info.is_rank0:
        print("=" * 60, flush=True)
        print("PIRL HF Training", flush=True)
        print("=" * 60, flush=True)
        print(f"Model: {model_path}", flush=True)
        print(f"Train dataset: {train_dataset} ({len(train_records)} samples)", flush=True)
        print(f"Eval dataset: {eval_dataset if eval_dataset else train_dataset} ({len(eval_records)} samples)", flush=True)
        print(f"Max steps: {max_steps}", flush=True)
        print(f"Batch size: {batch_size}, rollouts/prompt: {num_rollouts_per_prompt}", flush=True)
        print(f"World size: {dist_info.world_size}", flush=True)
        print("=" * 60, flush=True)

    for step in range(1, max_steps + 1):
        strength_now = _scheduled_strength(
            step=step,
            max_steps=max_steps,
            base=base_strength,
            curriculum=curriculum,
        )

        prompt_tasks: List[Dict[str, Any]] = []
        prompt_a_texts: List[str] = []
        prompt_b_texts: List[str] = []

        for i in range(batch_size):
            rec = rng.choice(train_records)
            base_id = str(rec.get("task_id", f"idx{i}"))
            gid = f"{base_id}::s{step}::r{dist_info.rank}::p{i}"
            task = _clone_task_with_group_id(rec, group_id=gid)
            prompt_tasks.append(task)

            obs = env.reset(task)
            obs_a = _randomize_interface_text(obs.content, strength=strength_now, rng=rng, variant=0)
            obs_b = _randomize_interface_text(obs.content, strength=strength_now, rng=rng, variant=1)
            prompt_a_texts.append(
                _build_prompt(
                    env_type=env_type,
                    observation=obs_a,
                    tokenizer=tokenizer,
                    use_chat_template=use_chat_template,
                    system_prompt=sys_prompt,
                )
            )
            prompt_b_texts.append(
                _build_prompt(
                    env_type=env_type,
                    observation=obs_b,
                    tokenizer=tokenizer,
                    use_chat_template=use_chat_template,
                    system_prompt=sys_prompt,
                )
            )

        enc_a = tokenizer(
            prompt_a_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_ids_a = enc_a["input_ids"].to(device)
        attention_mask_a = enc_a["attention_mask"].to(device)
        prompt_lens_a_base = [int(x) for x in attention_mask_a.sum(dim=1).tolist()]

        model.eval()
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
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
        expanded_prompt_lens_a = [prompt_lens_a_base[i // r] for i in range(n_samples)]
        prompt_b_ids_base = [_tokenize_prompt_ids(tokenizer, text, max_prompt_tokens) for text in prompt_b_texts]

        trajectories: List[Trajectory] = []
        completion_token_lists: List[List[int]] = []
        expanded_prompt_lens_b: List[int] = []
        concat_b_token_lists: List[List[int]] = []

        for idx, (seq, task, p_len_a) in enumerate(zip(sequences, expanded_tasks, expanded_prompt_lens_a)):
            ids = [int(x) for x in seq.tolist()]
            completion_ids = _extract_completion_ids(ids, prompt_len=p_len_a, eos_token_id=tokenizer.eos_token_id)
            completion_token_lists.append(completion_ids)

            base_i = idx // r
            p_b_ids = prompt_b_ids_base[base_i]
            expanded_prompt_lens_b.append(len(p_b_ids))
            concat_b_token_lists.append(p_b_ids + completion_ids)

            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            content = extract_python_code(completion_text)

            env.reset(task)
            if env_type == "code":
                env.step(Action(ActionType.CODE_WRITE, content, metadata={"logprob": 0.0}))
                env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
            else:
                env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query", metadata={"logprob": 0.0}))
            trajectories.append(copy.deepcopy(env.get_trajectory()))

        reward_values = [
            _trajectory_reward(
                t,
                mode=reward_mode,
                blend_alpha=reward_blend_alpha,
                failure_reward_floor=failure_reward_floor,
            )
            for t in trajectories
        ]
        advantages = _compute_grpo_advantages(
            trajectories,
            reward_values,
            flat_group_fallback=flat_group_fallback,
        )

        seq_a = sequences.to(device)
        attn_a, labels_a, comp_lens_a = build_attention_and_labels(
            seq_a,
            prompt_lens=expanded_prompt_lens_a,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        out_a = model(input_ids=seq_a, attention_mask=attn_a, use_cache=False)
        nll_sum_a = per_sample_nll(out_a.logits, labels_a)
        nll_mean_a = nll_sum_a / comp_lens_a.to(nll_sum_a.dtype)
        w_t = torch.tensor(advantages, device=device, dtype=nll_mean_a.dtype)
        rl_loss = (w_t * nll_mean_a).mean()

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        seq_b = _pad_sequences(concat_b_token_lists, pad_id=pad_id, device=device)
        attn_b, labels_b, comp_lens_b = build_attention_and_labels(
            seq_b,
            prompt_lens=expanded_prompt_lens_b,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        out_b = model(input_ids=seq_b, attention_mask=attn_b, use_cache=False)
        nll_sum_b = per_sample_nll(out_b.logits, labels_b)
        nll_mean_b = nll_sum_b / comp_lens_b.to(nll_sum_b.dtype)

        inv_loss = ((nll_mean_a.detach() - nll_mean_b) ** 2).mean()
        inv_weight_now = invariance_weight * max(0.1, strength_now)
        total_loss = rl_loss + inv_weight_now * inv_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

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

        del out_a, out_b

        if step % log_interval == 0:
            success_rate = sum(float(t.r_final) for t in trajectories) / max(1.0, float(len(trajectories)))
            pass_at_1 = _pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = _pass_at_k_from_groups(trajectories, k=r)
            avg_len = sum(float(t.length) for t in trajectories) / max(1.0, float(len(trajectories)))
            mean_reward = sum(reward_values) / max(1.0, float(len(reward_values)))
            nz_weight_ratio = sum(1 for w in advantages if abs(w) > 1e-8) / max(1.0, float(len(advantages)))

            total_mean = all_reduce_mean(float(total_loss.item()), dist_info=dist_info)
            rl_mean = all_reduce_mean(float(rl_loss.item()), dist_info=dist_info)
            inv_mean = all_reduce_mean(float(inv_loss.item()), dist_info=dist_info)
            succ_mean = all_reduce_mean(float(success_rate), dist_info=dist_info)
            p1_mean = all_reduce_mean(float(pass_at_1), dist_info=dist_info)
            pk_mean = all_reduce_mean(float(pass_at_k), dist_info=dist_info)
            reward_mean = all_reduce_mean(float(mean_reward), dist_info=dist_info)

            if tracker:
                tracker.log_metrics(
                    RunMetrics(
                        step=step,
                        train_loss=float(total_mean),
                        success_rate=float(succ_mean),
                        pass_at_1=float(p1_mean),
                        pass_at_k={r: float(pk_mean)},
                        avg_trajectory_length=float(avg_len),
                        wall_time=float(time.time() - wall_start),
                        extra={
                            "backend": "hf",
                            "world_size": dist_info.world_size,
                            "mean_reward": float(reward_mean),
                            "rl_loss": float(rl_mean),
                            "invariance_loss": float(inv_mean),
                            "invariance_weight_now": float(inv_weight_now),
                            "randomization_strength_now": float(strength_now),
                            "nonzero_weight_ratio": float(nz_weight_ratio),
                        },
                    )
                )
                tracker.log_event(
                    "pirl",
                    "logged train metrics",
                    {
                        "step": step,
                        "rl_loss": float(rl_mean),
                        "inv_loss": float(inv_mean),
                        "mean_reward": float(reward_mean),
                        "success_rate": float(succ_mean),
                    },
                )

            if dist_info.is_rank0:
                print(
                    f"[step {step}] loss={total_mean:.4f} rl={rl_mean:.4f} inv={inv_mean:.4f} "
                    f"succ={succ_mean:.4f} p@1={p1_mean:.4f} strength={strength_now:.3f}",
                    flush=True,
                )

        if tracker and dist_info.is_rank0 and (step % eval_interval == 0):
            eval_metrics = _evaluate_levels(
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
                ood_levels=ood_levels,
                base_strength=base_strength,
                seed=seed + step * 17,
                eval_tasks_per_level=eval_tasks_per_level,
            )
            robust_gap = float(eval_metrics.get("iid", 0.0) - eval_metrics.get("ood_hard", 0.0))
            tracker.log_event(
                "eval",
                "PIRL OOD evaluation",
                {"step": step, "metrics": eval_metrics, "robust_gap": robust_gap},
            )
            if dist_info.is_rank0:
                print(f"[eval step {step}] {json.dumps({'metrics': eval_metrics, 'robust_gap': robust_gap})}", flush=True)

        if tracker and dist_info.is_rank0 and save_interval and (step % save_interval == 0):
            ckpt_dir = Path(exp_dir) / "checkpoints" / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            tracker.log_event("checkpoint", "saved checkpoint", {"step": step, "path": str(ckpt_dir)})

    if tracker and dist_info.is_rank0:
        final_ckpt = Path(exp_dir) / "checkpoints" / "final"
        final_ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_ckpt))
        tokenizer.save_pretrained(str(final_ckpt))
        tracker.log_event("checkpoint", "saved final checkpoint", {"path": str(final_ckpt)})
        tracker.finalize()
        summary = {
            "experiment_name": experiment_name,
            "output_dir": exp_dir,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset if eval_dataset else train_dataset,
            "backend": "hf",
            "final_checkpoint": str(final_ckpt),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)

    barrier(dist_info=dist_info)


def run_toy_notice(full_cfg: Dict[str, Any]) -> None:
    experiment_name = str(full_cfg.get("experiment_name", "pirl_main"))
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
