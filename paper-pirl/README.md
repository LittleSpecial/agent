# paper-pirl

Prompt/Interface Randomized RL for robust tool-use agents.

## Dependency
This repo depends on `agent-rl-core`.

```bash
pip install -e ../agent-rl-core
pip install -e .
```

## Scope
- Interface randomization operators
- Invariance objective integration
- Multi-skill retention-aware robustness training
- OOD robustness benchmark suite

## Quick start
```bash
python scripts/train.py --config configs/train_pirl.yaml
python scripts/eval.py --config configs/train_pirl.yaml --ckpt path/to/ckpt
```
