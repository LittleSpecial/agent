# Paper 2 (Second): PIRL
# Prompt/Interface Randomized RL for Robust Tool-Use Agents

## 1. Core problem
Agent policies overfit fixed prompt/interface forms. At deployment, small wording/schema shifts cause performance collapse.

This is prompt/interface shift, not task difficulty increase.

## 2. Goal
Learn policies invariant to surface form changes while preserving in-domain performance.

## 3. Method
Training-time randomization operators:
- tool description paraphrase/reorder
- schema/field renaming with semantic-preserving mapping
- system prompt template variation
- mild distractor text injection

Sample transformation `T ~ Q(T)` each episode.

Base objective:
- `L_rl = E[R(tau; E_T)]`

Invariance regularization (behavior-level preferred first):
- `L_inv = KL(pi(.|s,T1) || pi(.|s,T2))`

Total objective:
- `L = L_rl - alpha * L_inv`

Use curriculum randomization strength (easy -> medium -> hard).

## 4. Evaluation protocol
Define three splits explicitly:
- IID (original format)
- OOD-Easy (light paraphrase/reorder)
- OOD-Hard (schema rename + template shift)

Main outcome is OOD-Hard robustness.

## 5. Baselines
- no randomization RL
- randomization only (no invariance loss)
- test-time correction baseline

## 6. Metrics
- IID / OOD success
- Robust gap = IID - OOD
- cost changes (tool/tokens)
- seed variance

## 7. Key claims
1. Better OOD robustness at similar IID level.
2. Invariance term contributes beyond plain augmentation.
3. Curriculum stabilizes training under strong shifts.

## 8. Risk and mitigation
Risk: reviewer says "just data augmentation".
Mitigation: strong controlled ablations + invariance-loss evidence + distribution-shift-specific benchmark design.
