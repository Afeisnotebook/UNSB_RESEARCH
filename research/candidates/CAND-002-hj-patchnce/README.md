# CAND-002: HJ-PatchNCE

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 围绕 harmful-joint patch 更新逐步收缩得到的 PatchNCE 候选家族。 |
| 当前结论 | `ROUTE2_SUSTAINED_LOCAL`：有限期 HJ 后完整状态交给 native UNSB，total step3200 为 `+0.871 dB`、6/6 域正，晚三点平均 `+1.180 dB`。 |
| 时间线位置 | T2 多轮形成；T5 关闭 continuous 版本；T10 重开有限期导航；T12 通过路线二长程交接门禁。 |
| 先看哪里 | [SEARCH-004 报告](../../searches/SEARCH-004-gap-aware-handoff/RESULTS.md) → [规格](./SPEC.md) → [形成史](./LINEAGE.md) → [决策索引](./DECISION_INDEX.md)。 |

来源：高不确定 patch 应谨慎更新的 PatchNCE 动机。具体实现经过降权、gradient risk、structure harm、relational routing 和 layer-0 harmful-joint 多轮收缩。

当前状态：`ROUTE2_SUSTAINED_LOCAL`。旧 continuous HJ 的负结果仍成立；晋级的是一个固定有限数据曝光窗口后、从完整方法状态永久切回 native UNSB 的新 iteration，不是对旧强度/layer 的网格搜索。它仍待 full100、跨 seed 与 confirmation 验证。

- 规格：[SPEC.md](./SPEC.md)
- 报告：[REPORT.md](./REPORT.md)
- clean-core 实验：[EXP-L1-DT-HJ-CLEAN-CORE-20260824](../../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/README.md)
- finite-horizon 实验：[EXP-L1-SEARCH-002-DTHJ-20260827](../../../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/README.md)
- route-2 handoff 实验：[EXP-L1-SEARCH-004-GAP-HANDOFF-20260828](../../../experiments/L1-local/EXP-L1-SEARCH-004-GAP-HANDOFF-20260828/README.md)
- 形成史：[DT/HJ/HNEK 互动史](../../synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md)
