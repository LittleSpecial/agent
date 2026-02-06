# NeurIPS 2026 备选方案：VERA-RL
# Verifier-Uncertainty Aware RL with Selective Rechecking

> 定位：当 verifier 不完美（假阳性/假阴性）时，RL 会学到偏策略。VERA-RL 通过在线估计 verifier 误差并触发“选择性复核”，降低噪声奖励带来的训练偏置。

## 1. 问题定义

## 1.1 现象
RLVR 假设 verifier 给出的 reward 足够可靠，但实际中常见：
- checker 漏检（FN）：正确轨迹被判错；
- checker 误判（FP）：错误轨迹被判对；
- 某些困难样本判定不稳定。

这些噪声会被策略梯度放大，导致：
- 梯度方向偏移；
- 训练曲线抖动；
- 策略过拟合 verifier loophole。

## 1.2 目标
在不大幅增加评估开销的前提下，提高 reward 质量与训练稳定性。

## 1.3 研究问题
是否可以：
1. 在线估计 verifier 可靠度；
2. 只对高风险样本做二级复核；
3. 用噪声校正后的 reward 做 RL 更新。

## 2. 核心方法

## 2.1 两层判定机制
- 一级：主 verifier（便宜、快速）
- 二级：高精 verifier / 重检流程（昂贵、准确）

目标不是全量二级评估，而是 selective rechecking。

## 2.2 不确定性评分
为每条轨迹定义不确定性分数 `q(tau)`，可由以下信号组合：
- policy 置信度（logprob margin）
- verifier 边界分数（若 checker 输出连续置信）
- 自一致性信号（多采样判定分歧）
- 历史误判先验（同类型任务的 FN/FP 统计）

触发规则：
`recheck if q(tau) > gamma`

`gamma` 可动态调节以满足复核预算。

## 2.3 噪声模型与奖励校正
设观测奖励 `r_obs`，真实奖励 `r_true`。
在线估计：
- `p_fp = P(r_obs=1 | r_true=0)`
- `p_fn = P(r_obs=0 | r_true=1)`

可用 EM/贝叶斯更新或滑动窗口频率估计。

构造校正奖励：
`r_corr = E[r_true | r_obs, q, p_fp, p_fn]`

然后用 `r_corr` 替代 `r_obs` 训练。

## 2.4 更新目标
在 PPO/GRPO 框架中，仅替换优势来源：
- 原：`A_t <- GAE(r_obs)`
- 新：`A_t <- GAE(r_corr)`

同时加入轻量正则：
- 限制高不确定样本占比；
- 控制复核预算。

## 2.5 成本控制
定义复核预算 `B_recheck`（每 batch 最大复核比例）。
策略：
- top-k 不确定样本复核；
- 自适应阈值 `gamma`：若超预算则提高阈值。

## 3. 差异化点（与已有噪声工作区分）

1. 面向长轨迹 tool-agent，而非仅单轮 reasoning。
2. 强调“复核预算-准确率-性能”三者联合优化。
3. 可直接插到现有 verifier-only pipeline，工程落地成本低。

## 4. 实验框架

## 4.1 任务
优先选择 verifier 有已知噪声来源的任务：
- code generation + flaky tests
- API workflow + partial checker
- optional web-like deterministic checker with noisy proxy

## 4.2 Baselines
- 原始 RLVR（无复核、无校正）
- 全量二级复核（上界，成本高）
- 随机复核（等预算）
- 只做触发不做校正

## 4.3 指标
性能：
- success/pass@1

噪声鲁棒：
- 对噪声注入强度的性能退化曲线
- reward-label 一致性提升

效率：
- 复核比例
- 每 1% 性能提升对应复核成本

稳定性：
- seed 方差
- 训练曲线振荡程度

## 4.4 消融
- 无不确定性估计（随机复核）
- 不同触发特征组合
- 不同预算 `B_recheck`
- 无噪声校正、仅重标注

## 4.5 分析
- 触发样本类型统计
- FP/FN 在线估计收敛行为
- “复核预算 vs 成果提升”收益曲线

## 5. 工程实现建议（4 卡 A100）

## 5.1 模型
- 7B LoRA 即可完成大部分验证
- 如需主结果再迁移到 14B

## 5.2 必要模块
1. uncertainty scorer
2. recheck scheduler (budgeted)
3. noise calibrator (fp/fn estimator)
4. corrected-reward adapter

## 5.3 实施顺序
1. 先做离线噪声注入实验验证思路
2. 再接在线复核触发
3. 最后接入真实二级 verifier

## 6. 风险与规避

风险 A：二级 verifier 也不稳定
- 规避：多源复核 + 保守融合（多数投票/阈值）

风险 B：复核成本过高
- 规避：严格预算 + 动态阈值 + 只查高价值样本

风险 C：贡献被认为是工程技巧
- 规避：突出噪声模型、预算理论与鲁棒性实验系统性。

## 7. 论文叙事模板

- 我们指出 verifier 噪声是 agent RLVR 的关键瓶颈。
- 提出 VERA-RL：不确定性感知 + 预算化选择性复核 + 奖励校正。
- 在相同复核预算下，显著提升训练稳定性和最终成功率。

## 8. 8 周执行计划

第 1-2 周：
- 噪声注入基线 + 无复核训练曲线

第 3-4 周：
- 不确定性触发 + 随机复核对照

第 5-6 周：
- 奖励校正模块 + 预算调度

第 7-8 周：
- 机制分析图 + 主表 + 写作

## 9. MVP 成功标准

1. 同复核预算下优于随机复核 >= 2~3 点；
2. 训练波动显著下降；
3. 在中高噪声设定下退化更慢；
4. 触发器与校正模块均有独立增益。
