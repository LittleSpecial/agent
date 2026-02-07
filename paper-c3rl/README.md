# paper-c3rl

Counterfactual Cost-Credit Constrained RL for tool-use agents.

## Scope
- Real HF/LoRA training backend (CodeEnv / SQLEnv)
- Counterfactual success credit + counterfactual cost credit
- Multi-budget dual-variable constrained optimization
- Unified A -> C3 training pipeline with checkpoint/eval tooling

## Install
```bash
pip install -e ../agent-rl-core
pip install -e .
```

## Train
```bash
python paper-c3rl/scripts/train.py --config paper-c3rl/configs/train_c3rl.yaml --backend hf
```

Recommended runtime overrides:
```bash
MODEL_PATH=/path/to/local/model \
TRAIN_DATA=datasets/code/mbpp_train.jsonl \
EVAL_DATA=datasets/code/humaneval_test.jsonl \
python paper-c3rl/scripts/train.py --config paper-c3rl/configs/train_c3rl_strict.yaml --backend hf
```

## Eval
```bash
python paper-c3rl/scripts/eval.py \
  --config paper-c3rl/configs/train_c3rl.yaml \
  --ckpt paper-c3rl/results/<exp>/checkpoints/final
```

## Notes
- `--backend toy` is kept only as an explicit disabled placeholder.
- Core C3 method code is under `paper-c3rl/methods/c3rl/`.
