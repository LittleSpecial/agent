#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
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
from shared.envs.base import Action, ActionType
from shared.hf import build_code_prompt, build_sql_prompt, extract_python_code, load_jsonl
from shared.hf.prompts import DEFAULT_SQL_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

from c3rl import C3Config, C3Trainer
from c3rl.train_utils import clone_task_with_group_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate C3-RL checkpoint (HF)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True, help="checkpoint directory (step_x / final)")
    parser.add_argument("--eval_tasks", type=int, default=64)
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


def _load_model_and_tokenizer(ckpt_dir: Path, *, dtype_name: str, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=trust_remote_code, padding_side="left")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

    # Try loading as PEFT adapter first, then fall back to full model checkpoint.
    model = None
    try:
        from peft import PeftConfig, PeftModel  # type: ignore

        peft_cfg = PeftConfig.from_pretrained(str(ckpt_dir))
        base = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(base, str(ckpt_dir))
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

    return model, tokenizer


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt}")

    cfg = _load_yaml(args.config)
    core_cfg_ref = str(cfg.get("core", {}).get("base_config", "../agent-rl-core/configs/base.yaml"))
    core_cfg_path = _resolve_base_config_path(args.config.resolve(), core_cfg_ref)
    base_cfg = _load_yaml(core_cfg_path)
    full_cfg = _merge_dict(base_cfg, cfg)

    method_cfg = dict(full_cfg.get("method", {}))
    model_cfg = dict(full_cfg.get("model", {}))
    env_cfg = dict(full_cfg.get("environment", {}))
    constraints_cfg = dict(full_cfg.get("constraints", {}))

    env_type = str(env_cfg.get("env_type", "code"))
    eval_dataset = env_cfg.get("eval_dataset")
    if not eval_dataset:
        raise ValueError("Missing environment.eval_dataset in config for evaluation.")

    eval_records = load_jsonl(Path(eval_dataset), max_samples=env_cfg.get("max_eval_samples"))
    if not eval_records:
        raise RuntimeError(f"Empty eval dataset: {eval_dataset}")

    import torch

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    dtype_name = str(model_cfg.get("dtype", "bf16"))
    trust_remote_code = bool(method_cfg.get("trust_remote_code", False))

    model, tokenizer = _load_model_and_tokenizer(args.ckpt, dtype_name=dtype_name, trust_remote_code=trust_remote_code)
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    system_prompt = DEFAULT_SYSTEM_PROMPT if env_type == "code" else DEFAULT_SQL_SYSTEM_PROMPT
    use_chat_template = True
    max_prompt_tokens = int(full_cfg.get("trainer", {}).get("max_prompt_tokens", 1024))
    max_new_tokens = int(full_cfg.get("trainer", {}).get("max_new_tokens", 192))
    max_steps = int(env_cfg.get("max_trajectory_length", 8))

    env = (
        CodeEnv(
            EnvConfig(
                name="code",
                max_steps=max_steps,
                seed=int(full_cfg.get("seed", 42)) + 123,
                extra={
                    "show_tests": bool(env_cfg.get("show_tests", True)),
                    "default_timeout": float(env_cfg.get("task_timeout_seconds", 8.0)),
                    "cap_task_timeout": True,
                },
            )
        )
        if env_type == "code"
        else SQLEnv(EnvConfig(name="sql", max_steps=max_steps, seed=int(full_cfg.get("seed", 42)) + 123))
    )

    c3 = C3Trainer(
        C3Config(
            dual_lr=float(method_cfg.get("dual_lr", 5e-4)),
            dual_warmup_steps=int(method_cfg.get("dual_warmup_steps", 200)),
            lambda_max=float(method_cfg.get("lambda_max", 10.0)),
            budget_targets={
                "tool_calls": float(constraints_cfg.get("tool_calls_budget", 8.0)),
                "output_tokens": float(constraints_cfg.get("output_tokens_budget", 1200.0)),
                "latency_ms": float(constraints_cfg.get("latency_ms_budget", 15000.0)),
            },
            credit_normalization=str(method_cfg.get("credit_normalization", "signed")),
            cost_normalization=str(method_cfg.get("cost_normalization", "signed")),
            fallback_to_adv_when_zero_credit=bool(method_cfg.get("fallback_to_adv_when_zero_credit", True)),
            zero_credit_threshold=float(method_cfg.get("zero_credit_threshold", 1e-8)),
        )
    )

    c3_state_path = args.ckpt / "c3_state.json"
    if c3_state_path.exists():
        c3.load_state_dict(json.loads(c3_state_path.read_text(encoding="utf-8")))

    subset_n = min(len(eval_records), max(1, int(args.eval_tasks)))
    subset = eval_records[:subset_n]

    successes = 0.0
    success_under_budget = 0.0
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
                success_under_budget += 1.0
            for k in cost_sum:
                cost_sum[k] += float(costs.get(k, 0.0))

    n = float(len(subset))
    summary = {
        "experiment_name": str(full_cfg.get("experiment_name", "c3rl_main")),
        "eval_checkpoint": str(args.ckpt),
        "eval_dataset": str(eval_dataset),
        "eval_tasks": int(len(subset)),
        "pass_at_1": successes / n,
        "success_rate": successes / n,
        "success_under_budget": success_under_budget / n,
        "violation_rate": violation_sum / n,
        "avg_cost_tool_calls": cost_sum["tool_calls"] / n,
        "avg_cost_output_tokens": cost_sum["output_tokens"] / n,
        "avg_cost_latency_ms": cost_sum["latency_ms"] / n,
        "dual_vars": c3.dual_vars,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
