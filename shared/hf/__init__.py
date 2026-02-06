"""
HuggingFace (Transformers) utilities.

This subpackage is only imported by the real LLM backends (HF / torch).
Keep the toy backend free from heavy deps by not importing this from `shared/__init__.py`.
"""

from .distributed import DistInfo, init_distributed, broadcast_object, all_reduce_mean
from .jsonl import load_jsonl
from .prompts import build_code_prompt, build_sql_prompt
from .code_extract import extract_python_code
from .rl_loss import build_attention_and_labels, per_sample_nll, weighted_rl_loss

__all__ = [
    "DistInfo",
    "init_distributed",
    "broadcast_object",
    "all_reduce_mean",
    "load_jsonl",
    "build_code_prompt",
    "build_sql_prompt",
    "extract_python_code",
    "build_attention_and_labels",
    "per_sample_nll",
    "weighted_rl_loss",
]
