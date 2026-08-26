# CAND-002: HJ-PatchNCE

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 围绕 harmful-joint patch 更新逐步收缩得到的 PatchNCE 候选家族。 |
| 当前结论 | `DEVELOPMENT_FROZEN / positive_but_fragile`：有限期 HJ 在 discovery70 的 1200 步为 `+0.710548 dB`、6/6 域正；handoff 后 1600 仍 `+0.660975 dB`。 |
| 时间线位置 | T2 多轮形成；T5 统一 clean rerun 后关闭；T10 以有限期方向导航重新打开。 |
| 先看哪里 | [SEARCH-002 报告](../../../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/REPORT.md) → [规格](./SPEC.md) → [形成史](./LINEAGE.md) → [决策索引](./DECISION_INDEX.md)。 |

来源：高不确定 patch 应谨慎更新的 PatchNCE 动机。具体实现经过降权、gradient risk、structure harm、relational routing 和 layer-0 harmful-joint 多轮收缩。

当前状态：`DEVELOPMENT_FROZEN / positive_but_fragile`。旧 continuous HJ 的负结果仍成立；重新打开的是一个固定有限数据曝光窗口后永久 handoff 给 plain 的新 iteration，不是对旧强度/layer 的网格搜索。

- 规格：[SPEC.md](./SPEC.md)
- 报告：[REPORT.md](./REPORT.md)
- clean-core 实验：[EXP-L1-DT-HJ-CLEAN-CORE-20260824](../../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/README.md)
- finite-horizon 实验：[EXP-L1-SEARCH-002-DTHJ-20260827](../../../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/README.md)
- 形成史：[DT/HJ/HNEK 互动史](../../synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md)
