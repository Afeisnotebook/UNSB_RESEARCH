# L2 medium-4090

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 4090 中型/长程实验层，用于把 L1 信号放大到更完整轨迹或更大 held-out 统计。 |
| 当前结论 | HNEK `gamma=0.25` 单 seed e200 开发冻结；六域 seed=2051 支持 shared-clock regret。两者均未进入训练 seed 级确认。 |
| 时间线位置 | T5 HNEK 长程开发、T6 动机六域放大。 |
| 先看哪里 | [HNEK e200](./EXP-L2-HNEK-SEARCH-E200-20260824/README.md)、[六域动机](./EXP-L2-MOTIVATION-SIXDOMAIN-20260824/README.md)。 |

L2 的图像 bootstrap 不能代替 L4 的训练 seed 统计；开发集搜索后的最佳分支也不能称为 confirmatory。
