# Search controllers

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 冻结多 lane 的训练、恢复、评估、排序与合成规则；它是选择控制器，不是算法结论。 |
| 当前结论 | SEARCH-001 本地完成，HNEK 以 `positive_but_fragile` 冻结为唯一候选；下一步 4090 matched 验证。 |
| 时间线位置 | T8 建立并过工程门，T9 完成本地筛选、合成、复赛和延长。 |
| 先看哪里 | [SEARCH-001](./SEARCH-001-clean-directional/README.md) → [L1 完整结果](../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/README.md) → [当前决策](../../decisions/CURRENT.md)。 |

`SEARCH` 是冻结的多 lane 选择/合成控制器，不是候选状态，也不是实验结论。代码、排序规则和数据访问边界在这里冻结；每次真实运行必须在 `experiments/` 创建独立记录，再由 `decisions/` 裁决。

当前搜索：

- [SEARCH-001 clean directional](./SEARCH-001-clean-directional/README.md)：本地已完成；唯一候选 HNEK，下一门禁为 seed=2026 的 4090 30k/60k/120k matched 验证。
