# Foundation

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 所有候选共同依赖的 deterministic UNSB canonical、数据/评估/复现契约和公共审计工具。 |
| 当前结论 | canonical 已通过本地真实数据 sampler、padding、推理重放和一步 full-state twin gate，可作为新实验父节点。 |
| 时间线位置 | T5 暴露原源码不确定性；T7 完成新基座验收；T8 搜索从此基座出发。 |
| 先看哪里 | [canonical](./canonical/README.md) → [验收契约](./canonical/CANONICAL_BASELINE.md) → [harness](./harness/README.md)。 |

`canonical/` 是所有候选共同使用的新确定性 UNSB 基座；`harness/` 提供数据身份、配置、checkpoint、确定性和统计工具。候选不得把自己的收益写回 canonical 定义。
