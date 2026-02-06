#!/usr/bin/env python3
"""
Export code benchmarks from HuggingFace datasets into JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _jsonl_write(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mbpp_to_records(ds: Any, *, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_samples is not None and len(records) >= max_samples:
            break
        prompt = item.get("text") or item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        task_id = item.get("task_id", idx)
        setup = item.get("test_setup_code") or item.get("test_setup") or ""
        tests = item.get("test_list") or item.get("tests") or []
        if not isinstance(tests, list):
            tests = []
        test_lines = [t for t in tests if isinstance(t, str)]
        parts: List[str] = []
        if isinstance(setup, str) and setup.strip():
            parts.append(setup.rstrip() + "\n")
        if test_lines:
            parts.append("\n".join(test_lines).rstrip() + "\n")
        test_code = "\n".join(parts).strip()
        records.append(
            {
                "task_id": f"mbpp/{task_id}",
                "prompt": prompt,
                "initial_code": item.get("starter_code") or "",
                "test_code": test_code,
                "language": "python",
                "timeout": 30.0,
                "metadata": {"source": "mbpp", "split": split, "task_type": "code_gen"},
            }
        )
    return records


def _humaneval_to_records(ds: Any, *, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_samples is not None and len(records) >= max_samples:
            break
        task_id = item.get("task_id", f"HumanEval/{idx}")
        prompt = item.get("prompt")
        test = item.get("test")
        entry_point = item.get("entry_point")
        if not (isinstance(prompt, str) and isinstance(test, str) and isinstance(entry_point, str)):
            continue
        test_code = test.rstrip() + "\n\n" + (
            "if __name__ == '__main__':\n"
            f"    check(globals()[{entry_point!r}])\n"
        )
        records.append(
            {
                "task_id": str(task_id),
                "prompt": prompt,
                "initial_code": "",
                "test_code": test_code,
                "language": "python",
                "timeout": 30.0,
                "metadata": {"source": "openai_humaneval", "split": split, "task_type": "code_gen"},
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mbpp", "openai_humaneval"])
    parser.add_argument("--split", required=True, type=str)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: datasets. Install with `python -m pip install datasets`.\n"
            f"Original error: {e}"
        )

    if args.dataset == "mbpp":
        ds = load_dataset("mbpp", args.config, split=args.split) if args.config else load_dataset("mbpp", split=args.split)
        rows = _mbpp_to_records(ds, split=args.split, max_samples=args.max_samples)
    else:
        ds = (
            load_dataset("openai_humaneval", args.config, split=args.split)
            if args.config
            else load_dataset("openai_humaneval", split=args.split)
        )
        rows = _humaneval_to_records(ds, split=args.split, max_samples=args.max_samples)

    _jsonl_write(rows, args.out)
    print(f"Wrote {len(rows)} records to {args.out}")


if __name__ == "__main__":
    main()
