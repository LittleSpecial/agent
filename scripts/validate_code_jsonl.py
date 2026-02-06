#!/usr/bin/env python3
"""
Validate JSONL task files used by the toy/data-backed pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _validate_record(record: Dict[str, Any], *, path: Path, line_no: int) -> List[str]:
    errors: List[str] = []
    if "task_id" not in record or not isinstance(record.get("task_id"), str):
        errors.append("task_id must be str")
    if "prompt" not in record or not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        errors.append("prompt must be non-empty str")
    if "metadata" in record and record["metadata"] is not None and not isinstance(record["metadata"], dict):
        errors.append("metadata must be dict")
    if "target_plan" in record:
        tp = record["target_plan"]
        if not isinstance(tp, list) or not tp or not all(isinstance(x, str) for x in tp):
            errors.append("target_plan must be non-empty list[str]")
    if "test_code" in record and isinstance(record["test_code"], str) and record["test_code"].strip():
        try:
            compile(record["test_code"], f"{path}:{line_no}", "exec")
        except SyntaxError as e:
            errors.append(f"test_code syntax error: {e.msg} (line {e.lineno})")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--max_errors", type=int, default=50)
    args = parser.parse_args()

    path = args.jsonl
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    seen = set()
    total = 0
    n_errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                n_errors += 1
                print(f"[ERROR] {path}:{line_no} invalid json: {e}")
                continue
            if not isinstance(row, dict):
                n_errors += 1
                print(f"[ERROR] {path}:{line_no} record must be object")
                continue

            task_id = row.get("task_id")
            if isinstance(task_id, str):
                if task_id in seen:
                    n_errors += 1
                    print(f"[ERROR] {path}:{line_no} duplicated task_id: {task_id}")
                seen.add(task_id)

            errs = _validate_record(row, path=path, line_no=line_no)
            if errs:
                n_errors += len(errs)
                for err in errs:
                    print(f"[ERROR] {path}:{line_no} {err}")

            if n_errors >= args.max_errors:
                raise SystemExit(1)

    if n_errors:
        print(f"Found {n_errors} errors in {total} rows: {path}")
        raise SystemExit(1)
    print(f"OK: {total} rows validated: {path}")


if __name__ == "__main__":
    main()
