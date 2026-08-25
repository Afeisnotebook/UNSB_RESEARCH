# CAND-002: HJ-PatchNCE

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 围绕 harmful-joint patch 更新逐步收缩得到的 PatchNCE 候选家族。 |
| 当前结论 | `CLOSED_NEGATIVE`：clean seed=2026 下 true 分支相对 plain 仅 `+0.0381 dB`，视为收益基本消失；`true−roll` 不能替代方法对 plain 的比较。 |
| 时间线位置 | T2 多轮形成；T5 统一 clean rerun 后关闭。 |
| 先看哪里 | [规格](./SPEC.md) → [形成史](./LINEAGE.md) → [clean 实验](../../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/README.md) → [决策索引](./DECISION_INDEX.md)。 |

来源：高不确定 patch 应谨慎更新的 PatchNCE 动机。具体实现经过降权、gradient risk、structure harm、relational routing 和 layer-0 harmful-joint 多轮收缩。

当前状态：`CLOSED_NEGATIVE`。实现和负结果保留用于追溯。

- 规格：[SPEC.md](./SPEC.md)
- 报告：[REPORT.md](./REPORT.md)
- clean-core 实验：[EXP-L1-DT-HJ-CLEAN-CORE-20260824](../../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/README.md)
- 形成史：[DT/HJ/HNEK 互动史](../../synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md)
