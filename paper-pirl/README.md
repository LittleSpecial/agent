# paper-pirl

Prompt/Interface Randomized RL (PIRL) for robust tool-use agents.

## Scope
- HF-based PIRL training (`paper-pirl/scripts/train.py`)
- OOD robustness evaluation (`paper-pirl/scripts/eval.py`)
- Slurm entrypoint for server runs (`scripts/slurm/run_pirl_main.sh`)

## Install
From repo root:

```bash
pip install -e ./agent-rl-core
pip install -e ./paper-pirl
```

## Required inputs
- `MODEL_PATH`: local HF model path (for example local Qwen2.5 checkpoint)
- `datasets/code/mbpp_train.jsonl`
- `datasets/code/humaneval_test.jsonl`

You can prepare datasets with:

```bash
python scripts/prepare_code_dataset_hf.py
python scripts/validate_code_jsonl.py --path datasets/code/mbpp_train.jsonl
python scripts/validate_code_jsonl.py --path datasets/code/humaneval_test.jsonl
```

## Local train (single process)
```bash
MODEL_PATH=/path/to/local/model \
TRAIN_DATA=datasets/code/mbpp_train.jsonl \
EVAL_DATA=datasets/code/humaneval_test.jsonl \
MAX_STEPS=200 \
python paper-pirl/scripts/train.py --config paper-pirl/configs/train_pirl.yaml --backend hf
```

## Slurm train
```bash
CONDA_ENV=$HOME/.conda/envs/rlvr \
MODEL_PATH=/path/to/local/model \
TRAIN_DATA=datasets/code/mbpp_train.jsonl \
EVAL_DATA=datasets/code/humaneval_test.jsonl \
MAX_STEPS=2000 \
sbatch scripts/slurm/run_pirl_main.sh
```

## Evaluate checkpoint
`train.py` prints final checkpoint path such as:
`paper-pirl/results/.../checkpoints/final`

```bash
MODEL_PATH=/path/to/local/model \
EVAL_DATA=datasets/code/humaneval_test.jsonl \
python paper-pirl/scripts/eval.py \
  --config paper-pirl/configs/train_pirl.yaml \
  --ckpt paper-pirl/results/<exp_id>/checkpoints/final \
  --backend hf
```

Output includes `iid/ood_easy/ood_hard` and `robust_gap = iid_success - ood_hard_success`.
