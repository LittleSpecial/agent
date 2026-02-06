# agent-rl-core

Shared infrastructure for long-horizon, verifier-based agent RL projects.

## Scope
- Environment and tool wrappers
- Rollout and replay interfaces
- Verifier abstraction
- Common training/evaluation loop hooks
- Shared logging/metrics definitions

## Why this repo exists
All paper repos (`paper-c3rl`, `paper-pirl`, `paper-vera-rl`) depend on the same baseline stack. This keeps comparisons fair and avoids duplicated bug fixes.

## Recommended workflow
1. Implement common components here first.
2. Expose stable interfaces in `src/agent_rl_core/interfaces.py`.
3. Keep paper-specific logic in each paper repo.
4. Pin a commit hash from each paper repo.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train_baseline.py --config configs/base.yaml
```

## Structure
- `src/agent_rl_core/`: shared python package
- `configs/`: baseline and shared config templates
- `scripts/`: baseline train/eval entry points
- `docs/`: architecture and interface notes
- `tests/`: smoke tests for core abstractions

## Stability contract
Anything exported in `interfaces.py` should be treated as a public API.
If you need to break it, bump version and update all paper repos.
