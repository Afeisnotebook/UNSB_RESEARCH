# Canonical contracts

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 把“同一实验”拆成数据身份、确定性、评估与复现四类可检查契约。 |
| 当前结论 | 新 canonical 已通过 L0 微验证；后续每次正式运行仍必须重新冻结 commit、manifest、配置、评估 split 和完整训练状态。 |
| 时间线位置 | T5–T7 建立，用来防止原实现随机方差、数据漂移或评估器差异覆盖小消融。 |
| 先看哪里 | [data identity](./data_identity/README.md)、[determinism](./determinism/DETERMINISM_FIX.md)、[evaluation](./evaluation/README.md)、[reproducibility](./reproducibility/README.md)。 |

这些契约通过只表示运行身份可信，不表示算法收益成立。
