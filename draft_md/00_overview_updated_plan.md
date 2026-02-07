# NeurIPS 2026 Ideas - Updated Overview (for sharing)

Last updated: 2026-02-07

## 1) Current final prioritization
1. Main paper: **C3-RL** (A baseline -> C3 upgrade)
2. Second paper: **PIRL** (prompt/interface shift robustness)
3. Module-only (not standalone paper): **VERA** (as C3 plugin/ablation)

## 2) Why this changed
- Old A (step credit only) is in a crowded lane; keep it as a strong baseline.
- C3 is stronger because it unifies:
  - sparse-reward step credit,
  - step-level cost credit,
  - explicit budget constraints.
- PIRL is a different bottleneck (distribution shift at deployment), so the two-paper story is complementary.
- VERA is useful, but too close to existing verifier-noise lines as a standalone; better as a C3 component.

## 3) One-line paper story
- **Paper 1 (C3-RL)**: under the same budget, learn to succeed more with fewer wasteful steps.
- **Paper 2 (PIRL)**: keep performance when prompt/schema/interface wording shifts.

## 4) Collision-risk positioning
- C3-RL: medium collision risk (must win by sharper definition + stronger experiments)
- PIRL: medium-low collision risk (if OOD protocol is rigorous)
- VERA standalone: high-ish overlap risk

## 5) Execution order
1. Freeze core baseline stack (`train/eval/checkpoint/metrics`).
2. Reproduce A baseline cleanly.
3. Implement C3 upgrade on top of A.
4. Add VERA as optional C3 module and ablation.
5. Develop PIRL as independent second paper track.
