# Paper 1 (Main): C3-RL
# Counterfactual Cost-Credit Constrained RL for Tool-Use Agents

## 1. Core problem
Long-horizon agents often train with only final verifiable reward. This causes:
- sparse and weak credit assignment,
- wasteful behavior (too many tool calls/tokens/latency),
- poor deployment efficiency.

Target is not only higher success, but higher success **under budgets**.

## 2. A -> C3 upgrade
A baseline (old idea): counterfactual step success credit only.

C3 upgrade adds two pieces:
1. counterfactual step cost credit,
2. constrained optimization with adaptive dual variables.

## 3. Method
For a trajectory `tau = (s_1,a_1,...,s_T,a_T)`:
- success contribution:
  - `u_t = E_I [R(tau) - R(tau_t^I)]`
- cost contribution for each channel `k`:
  - `v_{t,k} = E_I [C_k(tau) - C_k(tau_t^I)]`

`tau_t^I` is intervention replay (delete/truncate/replace at step `t`).

Step objective:
- `A_t^c3 = norm(u_t) - sum_k lambda_k * norm(v_{t,k})`

Policy update:
- PPO/GRPO style with `A_t^c3`

Dual update:
- `lambda_k <- [lambda_k + eta_lambda * (E[C_k]-B_k)]_+`

Interpretation:
- reward high-utility low-cost actions,
- suppress low-utility high-cost actions,
- satisfy multi-budget constraints.

## 4. Costs tracked
- tool call count
- output token count
- latency (ms)
(optional: API dollar cost)

## 5. VERA integration (module, not paper)
C3 can optionally enable verifier uncertainty correction:
- estimate uncertain samples,
- selective recheck under a fixed budget,
- use corrected reward in training.

Report VERA as plugin/ablation in C3.

## 6. Required experiments
Baselines:
- A baseline (step success credit only)
- reward-only PPO/GRPO
- fixed penalty (`R - alpha*C`)
- hard budget truncation

Main metrics:
- success/pass@1
- average costs per channel
- violation rate `Pr(C_k > B_k)`
- success under fixed budget
- Pareto frontier / area

Ablations:
- remove cost credit
- remove dual update
- single vs multi-cost
- with vs without VERA plugin
- loose/medium/strict budgets

## 7. Key claims to prove
1. Better success at equal budget.
2. Lower violation rates than fixed-penalty methods.
3. Step-level counterfactual cost credit adds value beyond global penalties.

## 8. Risk and mitigation
Risk: "old constrained RL repackaging".
Mitigation: emphasize step-level counterfactual cost credit + verifier-only long-horizon setting + hard empirical analysis.
