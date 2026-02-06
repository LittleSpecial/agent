# Architecture

## Layering
1. `interfaces.py`: stable contracts (env, policy, verifier, trajectory)
2. `runner.py`: rollout and replay orchestration
3. `trainer.py`: optimization loop and logging hooks
4. Paper repos: algorithm-specific modules only

## Rules
- Keep core algorithm-agnostic.
- Add feature flags for optional behavior.
- Avoid paper-specific metrics in core unless reused by >=2 projects.
