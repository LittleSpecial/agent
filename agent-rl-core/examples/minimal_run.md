# Minimal run

```bash
cd agent-rl-core
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train_baseline.py --config configs/base.yaml
```
