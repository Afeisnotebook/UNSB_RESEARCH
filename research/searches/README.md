# Search controllers

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 冻结多 lane 的训练、恢复、评估、排序与合成规则；它是选择控制器，不是算法结论。 |
| 当前结论 | SEARCH-005 的 6 类初始机制与 4 次因果修订均未通过持续收益门禁；PCOA 只是唯一 `weak_fallback`，不是已确认算法。 |
| 时间线位置 | T8–T10 完成 SEARCH-001/002；SEARCH-003 暴露目标漂移，T11 由 SEARCH-005 完成真正的路线一算子发现。 |
| 先看哪里 | [SEARCH-005](./SEARCH-005-long-horizon-operator-discovery/README.md) → [最终结果](./SEARCH-005-long-horizon-operator-discovery/RESULTS.md) → [当前决策](../../decisions/CURRENT.md)。 |

`SEARCH` 是冻结的多 lane 选择/合成控制器，不是候选状态，也不是实验结论。代码、排序规则和数据访问边界在这里冻结；每次真实运行必须在 `experiments/` 创建独立记录，再由 `decisions/` 裁决。

当前搜索：

- [SEARCH-005 long-horizon operator discovery](./SEARCH-005-long-horizon-operator-discovery/README.md)：当前权威路线一结果；没有持续候选，PCOA 为弱递补。
- [SEARCH-003 evidence-guided discovery](./SEARCH-003-evidence-guided-discovery/README.md)：保留反转证据；其 whole-branch controller 子路线已关闭，不代表路线一算子发现已结束。
- [SEARCH-001 clean directional](./SEARCH-001-clean-directional/README.md)：历史完整本地竞争；HNEK 为当时总冠军，现为递补一。
- [SEARCH-002 DT/HJ re-derivation](./SEARCH-002-dthj-rederivation/README.md)：历史 finite-horizon HJ 正窗口；窗口/接手协议不再是路线一当前答案。
