# Search controllers

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 冻结多 lane 的训练、恢复、评估、排序与合成规则；它是选择控制器，不是算法结论。 |
| 当前结论 | SEARCH-001 已通过 L0 full-state/resume/评估/PTQ 工程门，下一步是 stage1；尚未产生效果裁决。 |
| 时间线位置 | T8 建立，位于 deterministic canonical 验收之后、L1 方向筛选之前。 |
| 先看哪里 | [SEARCH-001](./SEARCH-001-clean-directional/README.md) → [L0 门禁](../../experiments/L0-contract/EXP-L0-SEARCH-001-GATE-20260826/README.md) → [当前决策](../../decisions/CURRENT.md)。 |

`SEARCH` 是冻结的多 lane 选择/合成控制器，不是候选状态，也不是实验结论。代码、排序规则和数据访问边界在这里冻结；每次真实运行必须在 `experiments/` 创建独立记录，再由 `decisions/` 裁决。

当前搜索：

- [SEARCH-001 clean directional](./SEARCH-001-clean-directional/README.md)：plain 与 DT/HJ/HNEK anchors，加 DCUM/LBST/PTQ/AEB 新机制；L0 工程门禁已通过，下一步为 stage1。
