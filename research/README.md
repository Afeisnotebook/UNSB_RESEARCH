# Research entities

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 保存研究问题、候选身份、搜索控制器与跨路线综合；回答“为什么做、做的是什么、如何演进”。 |
| 当前结论 | SEARCH-005 路线一没有持续候选；独立的 SEARCH-004 路线二证明 HJ 正状态可由 native UNSB 继承，并冻结 `HJ1200-NATIVE-HANDOFF` 为 sustained-local 第一候选。 |
| 时间线位置 | 覆盖 T1–T12；SEARCH-003/005 完成路线一，SEARCH-004 完成路线二状态交接因果审计。 |
| 先看哪里 | [动机](./motivations/README.md) → [候选](./candidates/README.md) → [搜索](./searches/README.md) → [综合](./synthesis/README.md)。 |

- `motivations/` 保存为什么值得研究以及当前允许的 claim。
- `candidates/` 保存具体候选、迭代和实现。
- `searches/` 保存跨候选的冻结搜索控制器；搜索输出必须另行登记为实验，不能直接改变候选状态。
- `synthesis/` 保存跨候选的数学、叙事和研究过程综合。

结果本体属于 `experiments/`，状态变化属于 `decisions/`。
