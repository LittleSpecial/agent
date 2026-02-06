# paper-c3rl

Counterfactual Cost-Credit Constrained RL for tool-use agents.

## Dependency
This repo depends on `agent-rl-core`.

Recommended local setup:
```bash
pip install -e ../agent-rl-core
pip install -e .
```

## Scope
- C3-RL method implementation
- A baseline -> C3 upgrade path (same pipeline)
- C3-specific configs and experiments
- C3 ablation scripts and analysis

## Quick start
```bash
python scripts/train.py --config configs/train_c3rl.yaml
python scripts/eval.py --config configs/train_c3rl.yaml --ckpt path/to/ckpt
```

## Structure
- `methods/c3rl/`: algorithm code
- `configs/`: experiment configs
- `scripts/`: train/eval entrypoints
- `references/`: related work and experiment checklist
- `results/`: output artifacts (ignored)
