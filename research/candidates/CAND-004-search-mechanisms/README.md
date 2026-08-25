# CAND-004: Search mechanisms incubation

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | DCUM、LBST、PTQ、AEB 四条新机制及候选完整状态恢复接口的孵化身份。 |
| 当前结论 | `IMPLEMENTED`，SEARCH-001 L0 工程门已通过；没有 PSNR/SSIM 效果裁决，不能进入论文方法结论。 |
| 时间线位置 | T8，在新 canonical 接受后登记并接入最后一轮分级搜索。 |
| 先看哪里 | [形成史](./LINEAGE.md) → [SEARCH-001](../../searches/SEARCH-001-clean-directional/README.md) → [L0 工程门](../../../experiments/L0-contract/EXP-L0-SEARCH-001-GATE-20260826/README.md)。 |

这是对 commit `495a092` 中、生命周期重构开始前已存在代码的登记，不是算法有效性裁决。

当前包含：DCUM 同域不同 stem unpaired marginal、LBST rollout EMA generator、PTQ physical-time quadrature、AEB antithetic endpoint averaging，以及 DT/HJ/HNEK 的额外训练状态恢复接口。

实现位置：DCUM 在 `foundation/canonical/src/data/unaligned_dataset.py`；LBST/PTQ/AEB 在 `foundation/canonical/src/models/sb_model.py`；lane、排序与状态编排在 `research/searches/SEARCH-001-clean-directional/`。

当前状态：`IMPLEMENTED`。四条新 lane 及 plain、DT、HJ、HNEK anchors 已在 [SEARCH-001](../../searches/SEARCH-001-clean-directional/README.md) 中冻结；在 L0 工程门禁、正式实验记录和决策完成前，不进入 HNEK/DT/HJ 的现有裁决，也不写入论文输出。
