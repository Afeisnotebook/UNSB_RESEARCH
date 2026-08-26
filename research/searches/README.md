# Search controllers

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 冻结多 lane 的训练、恢复、评估、排序与合成规则；它是选择控制器，不是算法结论。 |
| 当前结论 | SEARCH-002 已冻结 finite-horizon HJ 为 `positive_but_fragile` 第一候选；下一步 4090 matched 验证。 |
| 时间线位置 | T8–T9 完成 SEARCH-001；T10 重推导 DT/HJ 并冻结有限期 HJ。 |
| 先看哪里 | [SEARCH-002](./SEARCH-002-dthj-rederivation/README.md) → [L1 结果](../../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/README.md) → [当前决策](../../decisions/CURRENT.md)。 |

`SEARCH` 是冻结的多 lane 选择/合成控制器，不是候选状态，也不是实验结论。代码、排序规则和数据访问边界在这里冻结；每次真实运行必须在 `experiments/` 创建独立记录，再由 `decisions/` 裁决。

当前搜索：

- [SEARCH-001 clean directional](./SEARCH-001-clean-directional/README.md)：历史完整本地竞争；HNEK 为当时总冠军，现为递补一。
- [SEARCH-002 DT/HJ re-derivation](./SEARCH-002-dthj-rederivation/README.md)：输出空间 LTTR 被证伪后，有限期 HJ 在独立 discovery70 上保住正收益并成为当前第一候选。
