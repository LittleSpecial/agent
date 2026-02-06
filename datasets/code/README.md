# Code Datasets (JSONL)

Use JSONL files with one task per line.

Required fields:

- `task_id` (str)
- `prompt` (str)

Optional fields:

- `target_plan` (list[str]) for deterministic toy target actions
- `test_code` (str)
- `initial_code` (str)
- `metadata` (dict), e.g. `{"task_type":"type0","skill_id":"skill_0"}`

## Prepare from HuggingFace

```bash
python -m pip install datasets
python scripts/prepare_code_dataset_hf.py --dataset mbpp --split train --out datasets/code/mbpp_train.jsonl
python scripts/prepare_code_dataset_hf.py --dataset openai_humaneval --split test --out datasets/code/humaneval_test.jsonl
python scripts/validate_code_jsonl.py datasets/code/mbpp_train.jsonl
```
