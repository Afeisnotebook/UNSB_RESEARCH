# 统一方法叙事与 novelty 边界（投稿级）

> **历史叙事草案：** DT/HJ 在 clean deterministic 口径下没有稳定正收益，因此本文不得作为当前论文主线。见 [FINAL_STATUS.md](./FINAL_STATUS.md) 和 [../../CURRENT_STATE_CN.md](../../CURRENT_STATE_CN.md)。

## 统一介入原则

把 DT 与 HJ 统一为同一原则的两个实例：

> 在 unpaired UNSB 的 amortized 端点条件律上，对“局部响应统计”施加**最小、前向不变、可由诊断量自适应调节**的修正。

- DT：修正对象是局部响应的**分布尺度**（端点 proposal 分歧 `U`），用有界函数正则把当前律约束在固定参考律的标准化响应图表内。
- HJ：修正对象是局部响应的**更新方向**（PatchNCE 梯度），用前向不变的梯度投影只去除“结构有害方向”上的冲突分量。

两者共同点：不改 forward / 不加参数 / 只在 gate 命中的局部做小强度修正 / 用同一套诊断量（teacher-student 距离、SB 熵梯度范数、响应漂移）决定“何时、多强”介入。

## 贡献表述（允许）

> 我们提出一种针对 amortized Schrödinger bridge 恢复的统一介入原则：以可观测诊断量驱动的、前向不变的、对端点局部响应统计的最小修正。其两个实例分别约束响应分布尺度（DT）与响应更新方向（HJ），并在 unpaired EROT 设定下用可验证 gate 与消融检验其必要性。

## 禁止表述

- 首次 preconditioning；
- 首次 gradient surgery；
- 首次/校准 predictive uncertainty；
- 新 reference 设计 / 新噪声 schedule；
- 严格求得了完整 Schrödinger Bridge 或 Markov path law；
- “teacher 是 clean oracle” / “估计真实后验协方差”。

## novelty 边界如何守住

- 每个方法都显式锚定在 UNSB 的“受限端点律 + unpaired EROT”这一具体对象上；
- gate 与诊断量都可从 raw 特征纯函数重算；
- 消融用 roll / 随机方向 / 去掉归一化等对照，证明增益来自具体方向/尺度，而非通用正则；
- Related Work 主动承认 bridge preconditioning、gradient surgery、uncertainty 校准等先例。

## 与数理 grounding 的关系

本文件是叙事层；数学对象与 gate 定义见 `METHOD_GROUNDING.md`；机制证据诊断见 `refactor/_runs/diagnostics/`。

## 经验证据状态（诚实，单/3-seed）

- DT：3-seed 配对 delta +0.70 dB，95% CI [0.17, 1.23]（不含 0）→ 相对 plain 方向稳健；
  但各 trick 的 necessity 仍是单 seed，需 ≥5 seed。
- HJ：3-seed true−plain / true−roll 的 95% CI 均含 0，roll 对照跨 seed 极不稳定 →
  “结构方向特异”归因**未跨 seed 复现**，当前只能作方向参考，投稿级结论需 ≥5 seed。
- 因此“证明必要性”只允许写为“在单 seed 上检验、多 seed 待确认”；不得把 3-seed
  结果表述为已证实的投稿结论。
