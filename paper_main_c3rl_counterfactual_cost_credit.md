# NeurIPS 2026 主推方案 1：C3-RL
# Counterfactual Cost-Credit Constrained RL for Tool-Use Agents

> 定位：把“稀疏成功奖励 + 预算约束”统一起来。相比只优化成功率，C3-RL 明确学习“哪一步对成功最关键、哪一步最烧预算、单位成本收益是否值得”。

## 1. 问题定义

## 1.1 背景
在 agent RL 中，常见训练目标是最终成功率（success / pass@1）。
但线上部署更关心：
- 成功率不能低；
- tool 调用次数、token 成本、延迟都要受控；
- 不能靠“多试几次 + 超长链路”换分数。

现有做法通常是：
- 全局 reward 加固定 penalty：`R - alpha * Cost`；
- 或简单硬截断（超过预算就停）。

问题在于：
- 固定 `alpha` 很难迁移；
- 没有 step-level 信用分配，学不到“具体哪一步值得花预算”；
- 易出现训练不稳定或策略塌缩。

## 1.2 目标
构建一个约束优化框架，联合解决：
1. 稀疏成功奖励下的步骤信用分配；
2. 多预算约束（tool/token/latency）的稳定满足；
3. 在相同预算下提高成功率。

## 1.3 形式化（CMDP + 反事实步骤信用）
轨迹：`tau = (s_1,a_1,...,s_T,a_T)`

- 最终可验证奖励：`R(tau) in {0,1}` 或 `[0,1]`
- 多成本通道：`C_k(tau)`，`k=1..K`
- 预算：`B_k`

主问题：
`max_pi E[R(tau)]  s.t. E[C_k(tau)] <= B_k`

新增步骤信用：
- 成功边际贡献（反事实）
  `u_t = E_I [ R(tau) - R(tau^I_t) ]`
- 成本边际贡献（反事实）
  `v_{t,k} = E_I [ C_k(tau) - C_k(tau^I_t) ]`

其中 `tau^I_t` 是对第 `t` 步做干预后的重放轨迹（delete / truncate / replace）。

## 2. 核心方法

## 2.1 Lagrangian 化的步骤优势
构造步骤级拉格朗日优势：

`A_t^lag = norm(u_t) - sum_k lambda_k * norm(v_{t,k})`

- `lambda_k >= 0` 为对偶变量；
- `norm()` 采用组内标准化（如 z-score + clip）；
- 直觉：鼓励“高成功贡献、低成本贡献”的动作。

## 2.2 策略更新
可接 PPO/GRPO 结构，推荐 PPO-clip 形式：

`L_policy = E[min(r_t * A_t^lag, clip(r_t,1-eps,1+eps)*A_t^lag)] - beta*KL`

其中 `r_t = pi_theta(a_t|s_t)/pi_old(a_t|s_t)`。

## 2.3 对偶更新（预算约束）
每个 batch 结束：

`lambda_k <- [lambda_k + eta_lambda * (E_batch[C_k]-B_k)]_+`

稳定化：
- dual warmup：前 `N` step 固定 `lambda_k=0`；
- violation EMA：用 `m_k <- rho*m_k + (1-rho)*(E[C_k]-B_k)` 更新；
- dual clip：限制 `lambda_k <= lambda_max`。

## 2.4 反事实干预策略（降开销版）
不是每步都做干预，采用“选择性干预”：
- 仅对 top-p 不确定步骤（高熵动作、critic 方差高）执行干预；
- 每条轨迹最多干预 `M` 步（如 2~4）；
- 优先在成功轨迹 + 临界失败轨迹采样。

这样把反事实成本控制在可训练范围内。

## 3. 与现有工作差异（写作要点）

1. 与全局成本约束方法差异：
- 它们主要在 trajectory/global 层做 trade-off；
- C3-RL 在 step 层识别“预算花在哪一步才有效”。

2. 与单纯步骤信用方法差异：
- 仅步骤 credit 不处理部署预算；
- C3-RL 同时解决“信用来源 + 预算约束”。

3. 与 test-time budget tuning 差异：
- C3-RL 是训练期内生约束学习，不依赖后处理阈值。

## 4. 实验框架

## 4.1 任务与环境
最小建议三类：
- `CodeAgent`：代码修复/补全，reward 为单测通过；
- `APIAgent`：工具链任务，reward 为结构化验收器通过；
- `Web-lite`（可选）：简化网页任务，reward 为可验证目标达成。

每个任务都记录成本通道：
- tool_calls
- output_tokens
- latency_ms

## 4.2 Baselines
- SFT / RFT
- PPO/GRPO (reward-only)
- `R - alpha*C` 固定惩罚
- hard truncation
- global constrained RL（如果能复现）

## 4.3 主指标
- Success@1 / pass@1
- Avg tool_calls / tokens / latency
- 约束违例率：`Pr(C_k > B_k)`
- 同预算下成功率提升
- Pareto AUC（success-cost 前沿面积）

## 4.4 关键消融
- 去掉反事实成本信用（只留全局成本）
- 去掉对偶更新（固定 lambda）
- 单成本 vs 多成本
- 全步干预 vs 选择性干预
- 不同预算强度（松/中/紧）

## 4.5 机制分析
- credit map 可视化：成功贡献/成本贡献热力图
- 行为层分析：冗余工具调用减少比例
- 训练动态：`lambda_k` 曲线与违例率联动

## 5. 工程实现建议（4 卡 A100）

## 5.1 模型与训练配置
- 阶段 1：Qwen 7B + LoRA（快速迭代）
- 阶段 2：Qwen 14B + LoRA（主表）

建议：
- bf16, gradient checkpointing
- rollout 与训练异步队列化
- 成本统计在 runner 端直接记录

## 5.2 最小模块
1. rollout_runner（含成本日志）
2. verifier_service
3. intervention_replay（反事实）
4. credit_estimator（u_t, v_tk）
5. constrained_trainer（policy + dual）
6. dashboard（success/cost/violation/pareto）

## 6. 风险与备选

风险 A：反事实太慢
- 方案：选择性干预 + 并行重放 + 缓存环境状态

风险 B：双目标冲突导致成功率掉太多
- 方案：预算 curriculum（先松后紧）

风险 C：审稿人认为“旧 constrained RL 套壳”
- 方案：重点展示 step-level 反事实成本信用带来的额外收益（而不是仅换目标函数）。

## 7. 论文主叙事模板

- 我们提出 C3-RL：把反事实步骤信用分配与多预算约束优化统一。
- 方法能在不依赖 LLM judge 的前提下，学习“哪一步值得花预算”。
- 在多个长轨迹 agent 任务上，实现：
  1) 同预算更高成功率；
  2) 更低违例率；
  3) 更可解释的步骤决策。

## 8. 8 周执行计划

第 1-2 周：
- 打通 reward-only 和 fixed-penalty baseline
- 完成成本日志与约束评估脚本

第 3-4 周：
- 接入 dual 约束训练
- 单成本版本跑通 + 初步曲线

第 5-6 周：
- 反事实成本信用（选择性干预）
- 多成本版本与主消融

第 7-8 周：
- 机制分析图（credit/pareto/lambda dynamics）
- 写作与 rebuttal 材料预案

## 9. 最小可行结果（MVP）标准

满足以下三项即可进入写作期：
1. 在至少一个任务上，同预算 success 提升 >= 2~3 个点；
2. 约束违例率下降 >= 20%（相对下降）；
3. 消融显示反事实成本信用优于全局成本项。
