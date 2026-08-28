# Search controllers

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 冻结多 lane 的训练、恢复、评估、排序与合成规则；它是选择控制器，不是算法结论。 |
| 当前结论 | SEARCH-005 路线一无持续候选；SEARCH-004 路线二找到通过本地长程门禁的 `HJ1200-NATIVE-HANDOFF`，但尚未 full100/多 seed。 |
| 时间线位置 | T8–T11 完成路线一；T12 由 SEARCH-004 独立完成状态交接因果审计。 |
| 先看哪里 | [SEARCH-004](./SEARCH-004-gap-aware-handoff/README.md) → [最终结果](./SEARCH-004-gap-aware-handoff/RESULTS.md) → [当前决策](../../decisions/CURRENT.md)。 |

`SEARCH` 是冻结的多 lane 选择/合成控制器，不是候选状态，也不是实验结论。代码、排序规则和数据访问边界在这里冻结；每次真实运行必须在 `experiments/` 创建独立记录，再由 `decisions/` 裁决。

当前搜索：

- [SEARCH-004 gap-aware handoff](./SEARCH-004-gap-aware-handoff/README.md)：当前权威路线二结果；HJ complete-state native handoff 为唯一 4090 第一候选。
- [SEARCH-005 long-horizon operator discovery](./SEARCH-005-long-horizon-operator-discovery/README.md)：当前权威路线一结果；没有持续候选，PCOA 为弱递补。
- [SEARCH-003 evidence-guided discovery](./SEARCH-003-evidence-guided-discovery/README.md)：保留反转证据；其 whole-branch controller 子路线已关闭，不代表路线一算子发现已结束。
- [SEARCH-001 clean directional](./SEARCH-001-clean-directional/README.md)：历史完整本地竞争；HNEK 为当时总冠军，现为递补一。
- [SEARCH-002 DT/HJ re-derivation](./SEARCH-002-dthj-rederivation/README.md)：历史 finite-horizon HJ 正窗口；窗口/接手协议不再是路线一当前答案。
