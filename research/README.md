# Research entities

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 保存研究问题、候选身份、搜索控制器与跨路线综合；回答“为什么做、做的是什么、如何演进”。 |
| 当前结论 | MOT-001 有限支持；SEARCH-001 本地完成，HNEK 为脆弱正向总冠军，四个新机制当前实现关闭为负。 |
| 时间线位置 | 覆盖 T1–T9；动机重建集中在 T4–T6，T9 完成本地方向搜索。 |
| 先看哪里 | [动机](./motivations/README.md) → [候选](./candidates/README.md) → [搜索](./searches/README.md) → [综合](./synthesis/README.md)。 |

- `motivations/` 保存为什么值得研究以及当前允许的 claim。
- `candidates/` 保存具体候选、迭代和实现。
- `searches/` 保存跨候选的冻结搜索控制器；搜索输出必须另行登记为实验，不能直接改变候选状态。
- `synthesis/` 保存跨候选的数学、叙事和研究过程综合。

结果本体属于 `experiments/`，状态变化属于 `decisions/`。
