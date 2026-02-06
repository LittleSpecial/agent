# NeurIPS 2026 主推方案 2：PIRL
# Prompt/Interface Randomized RL for Robust Tool-Use Agents

> 定位：这篇不卷更高的单一 pass@1，而是解决 agent 在真实环境中的核心痛点：接口描述一改、工具文档换版本、prompt 模板变化后性能断崖。

## 1. 问题定义

## 1.1 现象
许多 agent 在训练环境表现不错，但遇到以下变化就崩：
- tool 描述文本改写（同义改写/顺序变化）；
- API 参数字段重命名；
- 系统提示模板变化；
- UI/接口文档小幅漂移。

这类问题本质是 **prompt/interface overfitting**。

## 1.2 目标
学习一个对接口表述变化更稳健的策略，而不是记忆某一份固定 prompt/schema。

## 1.3 研究问题
在 verifier-only 的 RL setting 下，是否可以通过“训练期随机化 + 不变性约束”获得：
- 更高的分布外鲁棒性；
- 不明显牺牲 in-domain 成绩；
- 更可控的鲁棒-效率折中。

## 2. 核心方法

## 2.1 随机化算子族（Domain Randomization for Agent Interface）
定义随机化变换 `T ~ Q(T)`，应用于每个 episode：
- `T_desc`：tool 描述同义改写/结构重排；
- `T_schema`：字段名映射（保持语义不变）；
- `T_prompt`：系统模板改写；
- `T_noise`：注入轻量无关信息（防脆弱匹配）。

得到随机化后的环境 `E_T`，训练在 `E_T` 上进行。

## 2.2 训练目标
基础 RL 目标：
`L_rl = E_{T,tau~pi}[ R(tau; E_T) ]`

加入不变性约束（两种可选）

### 方案 A：行为分布一致性
对同一状态在两个随机化视图 `T1, T2`，约束策略分布一致：
`L_inv = KL( pi(.|s,T1) || pi(.|s,T2) )`

总损失：
`L = L_rl - alpha * L_inv`

### 方案 B：表示不变性
约束中间表示 `h(s,T)` 跨视图一致：
`L_inv = || h(s,T1) - h(s,T2) ||_2^2`

实践建议先用方案 A，落地更直接。

## 2.3 难度调度（Curriculum Randomization）
训练初期扰动过强会学不动，建议调度：
- Stage 1：轻扰动（同义改写、顺序变化）；
- Stage 2：中扰动（字段重命名映射）；
- Stage 3：强扰动（组合变换 + 干扰信息）。

## 2.4 评测分层
- IID：原始接口分布
- OOD-Easy：轻改写
- OOD-Hard：字段重命名 + 模板变化

论文主结果应以 OOD-Hard 为核心。

## 3. 差异化点（避免撞车）

1. 不是做静态 prompt engineering，而是 **训练期鲁棒策略学习**。
2. 不是单轮 QA，而是多步工具交互 agent。
3. 不依赖 LLM judge，用可验证 reward，结果可复现。

## 4. 实验设计

## 4.1 任务集
推荐 2~3 类：
- code tool-use（函数调用/测试执行）
- API workflow（多工具串联）
- 可选 web-lite（页面操作动作）

每类都构造 interface-shift benchmark。

## 4.2 Baselines
- 常规 PPO/GRPO（无随机化）
- 只做数据增强（无不变性项）
- test-time self-correction（无训练期鲁棒化）
- 可能的话加一个 prompt-overfit 缓解 baseline

## 4.3 指标
主指标：
- IID success
- OOD-Easy success
- OOD-Hard success
- Robust gap = IID - OOD

次指标：
- 平均 tool 调用数
- token 成本
- 训练稳定性（seed 方差）

## 4.4 消融
- 去掉 `L_inv`
- 去掉 curriculum
- 仅 `T_desc` / 仅 `T_schema` / 组合
- 不同 `alpha` 强度

## 4.5 机制分析
- 哪类变换最影响性能
- 策略熵与鲁棒性的关系
- OOD 失败案例分类（schema mismatch / planning error / execution error）

## 5. 工程实现建议（4 卡 A100）

## 5.1 模型与配置
- 7B LoRA 起步，14B LoRA 做主结果
- 与现有 RL pipeline 兼容：只需在环境输入层接随机化模块

## 5.2 必要模块
1. interface randomizer
2. schema mapper（可逆映射）
3. invariance loss hook
4. OOD evaluation harness

## 5.3 数据与缓存
- 随机化模板预先生成并缓存，减少在线改写成本
- 字段映射表固定 seed 可复现

## 6. 风险与规避

风险 A：IID 下降明显
- 规避：多目标平衡（调小 alpha）+ curriculum

风险 B：OOD 提升来自额外数据量而非方法
- 规避：严格控制样本预算，对比“等样本增强”基线

风险 C：审稿人认为是 data augmentation
- 规避：强调不变性约束和跨层级接口变换设计，并给机制分析。

## 7. 论文叙事模板

- 我们提出 PIRL：面向 tool-use agent 的接口随机化鲁棒 RL 框架。
- 方法在训练期同时优化任务成功与表示/行为不变性。
- 在多类 interface shift 上显著提升 OOD 成功率，并减少 robust gap。

## 8. 8 周执行计划

第 1-2 周：
- 实现 `T_desc/T_schema` + 基础 OOD 评测器

第 3-4 周：
- 跑通无不变性和有不变性两条线

第 5-6 周：
- 加 curriculum + 主消融 + 多 seed

第 7-8 周：
- 失败案例分析 + 写作图表

## 9. MVP 成功标准

1. OOD-Hard 提升 >= 4~6 点；
2. IID 下降 <= 1~2 点；
3. robust gap 显著收窄；
4. 消融证明 `L_inv` 与 curriculum 都有独立贡献。
