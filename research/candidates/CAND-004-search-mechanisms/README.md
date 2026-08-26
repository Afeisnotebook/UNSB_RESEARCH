# CAND-004: Search mechanisms incubation

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | DCUM、LBST、PTQ、AEB 四条新机制及候选完整状态恢复接口的孵化身份。 |
| 当前结论 | `CLOSED_NEGATIVE`：四条 standalone lane 均未保持正的晚期轨迹，DCUM 合成也未通过完整视图复赛。 |
| 时间线位置 | T8 接入搜索，T9 由 SEARCH-001 本地实验裁决。 |
| 先看哪里 | [SEARCH-001 报告](../../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/REPORT.md) → [完整轨迹](../../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/RESULTS.json) → [形成史](./LINEAGE.md)。 |

这是对 commit `495a092` 中、生命周期重构开始前已存在代码的登记，不是算法有效性裁决。

当前包含：DCUM 同域不同 stem unpaired marginal、LBST rollout EMA generator、PTQ physical-time quadrature、AEB antithetic endpoint averaging，以及 DT/HJ/HNEK 的额外训练状态恢复接口。

实现位置：DCUM 在 `foundation/canonical/src/data/unaligned_dataset.py`；LBST/PTQ/AEB 在 `foundation/canonical/src/models/sb_model.py`；lane、排序与状态编排在 `research/searches/SEARCH-001-clean-directional/`。

当前状态：`CLOSED_NEGATIVE`。DCUM 有短程正信号，但单独晚期反转，HNEK+DCUM 在 stage2 的最后三点均值为 `-0.381743 dB`；LBST、PTQ、AEB 及 DCUM+AEB 均为负。实现保留作可追溯研究资产，但不作为 4090 首发候选。
