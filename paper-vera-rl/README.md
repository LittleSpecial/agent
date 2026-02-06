# paper-vera-rl

Verifier-Uncertainty Aware RL with selective rechecking.

## Dependency
This repo depends on `agent-rl-core`.

```bash
pip install -e ../agent-rl-core
pip install -e .
```

## Scope
- Verifier uncertainty scoring
- Budgeted selective rechecking
- Reward-noise calibration and corrected training
- Optional auxiliary module on top of core/C3/PIRL pipelines

## Quick start
```bash
python scripts/train.py --config configs/train_vera.yaml
python scripts/eval.py --config configs/train_vera.yaml --ckpt path/to/ckpt
```
