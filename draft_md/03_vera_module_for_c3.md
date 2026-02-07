# VERA as C3 Module (not standalone paper)
# Verifier-Uncertainty Aware Reward Correction

## 1. Positioning
Use VERA as a pluggable component inside C3.

Do not position VERA as an independent primary paper line.

## 2. Why keep it
Verifier noise (FN/FP) can bias policy gradients. A lightweight uncertainty-aware recheck improves reward quality.

## 3. Module design
- uncertainty score `q(tau)` from policy/verifier signals
- selective recheck if `q(tau) > gamma`
- fixed recheck budget per batch
- estimate FN/FP statistics online
- corrected reward `r_corr` for advantage computation

## 4. Minimal interfaces
Inputs:
- trajectory record
- raw verifier result
- optional confidence signals

Outputs:
- corrected reward
- recheck flag
- diagnostics (estimated FN/FP, budget use)

## 5. C3 ablation table recommendation
At least include:
- C3 without VERA
- C3 with VERA (same compute budget)
- random recheck baseline

Report:
- success
- violation rate
- training stability
- recheck cost ratio

## 6. Decision rule
Keep VERA enabled only if it gives measurable gains under fixed recheck budget. Otherwise keep it as negative/neutral ablation evidence.
