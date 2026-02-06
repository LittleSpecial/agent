#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType
from shared.hf import (
    build_code_prompt,
    build_sql_prompt,
    extract_python_code,
    load_jsonl,
)
from shared.hf.prompts import DEFAULT_SQL_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PIRL (HF backend)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint directory or JSON summary path")
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


def _strength_base(strength: str) -> float:
    mapping = {"low": 0.15, "medium": 0.3, "high": 0.5}
    if strength in mapping:
        return mapping[strength]
    try:
        return max(0.0, min(1.0, float(strength)))
    except Exception:
        return 0.3


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
    if comp:
        return comp
    if eos_token_id is not None:
        return [int(eos_token_id)]
    return [seq_ids[-1]]


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


def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "adapter_config.json").exists():
        return True
    if (path / "pytorch_model.bin").exists():
        return True
    if (path / "model.safetensors").exists():
        return True
    if list(path.glob("pytorch_model-*.bin")):
        return True
    if list(path.glob("model-*.safetensors")):
        return True
    return False


def _resolve_checkpoint_dir(ckpt: Path) -> Path:
    path = ckpt.expanduser().resolve()
    if path.is_dir():
        if _is_model_dir(path):
            return path
        final_dir = path / "checkpoints" / "final"
        if _is_model_dir(final_dir):
            return final_dir
        step_dirs = sorted((path / "checkpoints").glob("step_*")) if (path / "checkpoints").is_dir() else []
        step_dirs = [d for d in step_dirs if _is_model_dir(d)]
        if step_dirs:
            return step_dirs[-1]
        raise ValueError(f"No model files found in checkpoint directory: {path}")

    if not path.is_file():
        raise ValueError(f"Checkpoint path does not exist: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError(f"Checkpoint file must be .json summary or model directory: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint summary must be a JSON object: {path}")

    candidates: List[str] = []
    for key in ("final_checkpoint", "checkpoint", "ckpt", "path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    output_dir = payload.get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        candidates.append(str(Path(output_dir) / "checkpoints" / "final"))

    for raw in candidates:
        c_path = Path(raw).expanduser()
        if not c_path.is_absolute():
            c_path = (path.parent / c_path).resolve()
        else:
            c_path = c_path.resolve()
        if _is_model_dir(c_path):
            return c_path

    raise ValueError(
        "Could not resolve a model checkpoint directory from JSON summary. "
        "Pass --ckpt <.../checkpoints/final> instead."
    )


def _load_tokenizer(*, ckpt_dir: Path, model_path: Optional[str], trust_remote_code: bool):
    from transformers import AutoTokenizer

    candidates: List[str] = [str(ckpt_dir)]
    if model_path:
        candidates.append(str(model_path))

    errors: List[str] = []
    tokenizer = None
    for cand in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cand,
                trust_remote_code=trust_remote_code,
                padding_side="left",
            )
            break
        except Exception as e:  # pragma: no cover - surface useful diagnostics
            errors.append(f"{cand}: {type(e).__name__}: {e}")

    if tokenizer is None:
        joined = " | ".join(errors)
        raise RuntimeError(f"Failed to load tokenizer from checkpoint/model path. Details: {joined}")

    return tokenizer


def _load_model(
    *,
    ckpt_dir: Path,
    model_path: Optional[str],
    dtype_name: str,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

    if (ckpt_dir / "adapter_config.json").exists():
        try:
            from peft import AutoPeftModelForCausalLM  # type: ignore
        except ModuleNotFoundError as e:
            raise SystemExit(
                "This checkpoint is a LoRA adapter, but `peft` is missing. "
                f"Install peft first. Original error: {e}"
            )
        try:
            return AutoPeftModelForCausalLM.from_pretrained(
                str(ckpt_dir),
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            if not model_path:
                raise
            from peft import PeftModel  # type: ignore

            base = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            return PeftModel.from_pretrained(base, str(ckpt_dir))

    return AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )


def _evaluate_level(
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
    strength: float,
    seed: int,
    eval_tasks_per_level: int,
    level: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Dict[str, float]:
    import torch

    if not eval_records:
        return {
            "n_tasks": 0.0,
            "success_rate": 0.0,
            "avg_verifier_score": 0.0,
            "avg_output_tokens": 0.0,
            "avg_tool_calls": 0.0,
            "avg_latency_ms": 0.0,
        }

    rng = random.Random(seed)
    subset_n = min(len(eval_records), max(1, eval_tasks_per_level))
    subset = rng.sample(eval_records, subset_n) if len(eval_records) > subset_n else list(eval_records)

    success_sum = 0.0
    score_sum = 0.0
    out_token_sum = 0.0
    tool_call_sum = 0.0
    latency_sum = 0.0

    model.eval()
    with torch.no_grad():
        for i, rec in enumerate(subset):
            gid = f"{rec.get('task_id', 'eval')}::pirl::{level}::{i}"
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
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_len = int(attention_mask.sum(dim=1).item())

            gen_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = max(1e-5, float(temperature))
                gen_kwargs["top_p"] = max(1e-5, min(1.0, float(top_p)))
            else:
                gen_kwargs["do_sample"] = False

            seq = model.generate(**gen_kwargs)[0]
            ids = [int(x) for x in seq.tolist()]
            comp_ids = _extract_completion_ids(ids, prompt_len=prompt_len, eos_token_id=tokenizer.eos_token_id)
            completion_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
            content = extract_python_code(completion_text)

            env.reset(task)
            t0 = time.time()
            if env_type == "code":
                env.step(Action(ActionType.CODE_WRITE, content))
                _, reward, _, _ = env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
            else:
                _, reward, _, _ = env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query"))
            latency_ms = (time.time() - t0) * 1000.0
            traj = env.get_trajectory()

            success_sum += float(traj.r_final if traj is not None else reward)
            score_sum += float(
                getattr(traj.verifier_info, "score", reward) if traj is not None else reward
            )
            out_token_sum += float(len(comp_ids))
            tool_call_sum += float(getattr(traj, "total_tool_calls", 0.0) if traj is not None else 0.0)
            latency_sum += float(latency_ms)

    denom = float(max(1, subset_n))
    return {
        "n_tasks": float(subset_n),
        "success_rate": success_sum / denom,
        "avg_verifier_score": score_sum / denom,
        "avg_output_tokens": out_token_sum / denom,
        "avg_tool_calls": tool_call_sum / denom,
        "avg_latency_ms": latency_sum / denom,
    }


def run_hf(full_cfg: Dict[str, Any], ckpt_arg: Path) -> None:
    import torch

    seed = int(full_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    model_cfg = dict(full_cfg.get("model", {}))
    trainer_cfg = dict(full_cfg.get("trainer", {}))
    method_cfg = dict(full_cfg.get("method", {}))
    env_cfg = dict(full_cfg.get("environment", {}))
    eval_cfg = dict(full_cfg.get("evaluation", {}))

    model_path = os.getenv("MODEL_PATH", model_cfg.get("path") or model_cfg.get("name"))
    eval_dataset = os.getenv("EVAL_DATA", env_cfg.get("eval_dataset") or env_cfg.get("train_dataset"))
    if not eval_dataset:
        raise ValueError("Missing eval dataset. Set EVAL_DATA or environment.eval_dataset in config.")

    env_type = str(env_cfg.get("env_type", full_cfg.get("env_type", "code")))
    max_trajectory_length = int(os.getenv("MAX_TRAJECTORY_LENGTH", env_cfg.get("max_trajectory_length", 8)))
    max_prompt_tokens = int(os.getenv("MAX_PROMPT_TOKENS", trainer_cfg.get("max_prompt_tokens", 1024)))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", trainer_cfg.get("max_new_tokens", 192)))
    eval_tasks_per_level = int(os.getenv("EVAL_TASKS_PER_LEVEL", eval_cfg.get("eval_tasks_per_level", 128)))
    max_eval_samples = (
        int(os.getenv("MAX_EVAL_SAMPLES"))
        if os.getenv("MAX_EVAL_SAMPLES") is not None
        else env_cfg.get("max_eval_samples")
    )
    dtype_name = str(os.getenv("DTYPE", model_cfg.get("dtype", "bf16")))
    require_cuda = _bool_env("REQUIRE_CUDA", True)
    trust_remote_code = _bool_env("TRUST_REMOTE_CODE", False)
    use_chat_template = _bool_env("USE_CHAT_TEMPLATE", True)
    do_sample = _bool_env("EVAL_DO_SAMPLE", False)
    temperature = float(os.getenv("EVAL_TEMPERATURE", trainer_cfg.get("temperature", 1.0)))
    top_p = float(os.getenv("EVAL_TOP_P", trainer_cfg.get("top_p", 1.0)))
    ood_levels = [str(x) for x in eval_cfg.get("ood_levels", ["iid", "ood_easy", "ood_hard"])]
    randomization_strength = str(os.getenv("RANDOMIZATION_STRENGTH", method_cfg.get("randomization_strength", "medium")))
    base_strength = _strength_base(randomization_strength)
    strengths = _build_eval_strengths(ood_levels, base_strength)
    experiment_name = str(os.getenv("EXPERIMENT_NAME", full_cfg.get("experiment_name", "pirl_main")))
    sys_prompt = os.getenv("SYSTEM_PROMPT") or (
        DEFAULT_SYSTEM_PROMPT if env_type == "code" else DEFAULT_SQL_SYSTEM_PROMPT
    )

    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise SystemExit("[ERROR] CUDA is required for HF eval, but no CUDA device is available.")
    device = torch.device("cuda:0" if has_cuda else "cpu")

    ckpt_dir = _resolve_checkpoint_dir(ckpt_arg)
    eval_records = load_jsonl(Path(eval_dataset), max_samples=max_eval_samples)
    if not eval_records:
        raise RuntimeError(f"Empty eval dataset: {eval_dataset}")

    tokenizer = _load_tokenizer(
        ckpt_dir=ckpt_dir,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
    )
    model = _load_model(
        ckpt_dir=ckpt_dir,
        model_path=model_path,
        dtype_name=dtype_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if has_cuda:
        model = model.to(device)
    model.eval()

    env = (
        CodeEnv(
            EnvConfig(
                name="code",
                max_steps=max_trajectory_length,
                seed=seed,
                extra={"show_tests": True},
            )
        )
        if env_type == "code"
        else SQLEnv(
            EnvConfig(
                name="sql",
                max_steps=max_trajectory_length,
                seed=seed,
            )
        )
    )

    level_metrics: Dict[str, Dict[str, float]] = {}
    for idx, level in enumerate(ood_levels):
        level_metrics[level] = _evaluate_level(
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
            strength=float(strengths.get(level, base_strength)),
            seed=seed + 31 * (idx + 1),
            eval_tasks_per_level=eval_tasks_per_level,
            level=level,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    iid_success = float(level_metrics.get("iid", {}).get("success_rate", 0.0))
    hard_success = float(level_metrics.get("ood_hard", {}).get("success_rate", 0.0))
    robust_gap = iid_success - hard_success

    summary = {
        "experiment_name": experiment_name,
        "backend": "hf",
        "checkpoint_dir": str(ckpt_dir),
        "model_path": model_path,
        "eval_dataset": eval_dataset,
        "eval_records": len(eval_records),
        "eval_tasks_per_level": eval_tasks_per_level,
        "ood_levels": ood_levels,
        "level_metrics": level_metrics,
        "robust_gap": robust_gap,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def run_toy_notice(full_cfg: Dict[str, Any]) -> None:
    experiment_name = str(full_cfg.get("experiment_name", "pirl_main"))
    print(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "backend": "toy",
                "status": "disabled",
                "message": "Toy backend is intentionally disabled. Use --backend hf for paper-level evaluation.",
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
    run_hf(full_cfg, args.ckpt)


if __name__ == "__main__":
    main()
