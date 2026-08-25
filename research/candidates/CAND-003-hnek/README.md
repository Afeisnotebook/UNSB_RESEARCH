# CAND-003: HNEK

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 回到 Schrödinger Bridge 物理 horizon 与 endpoint transition 的桥原生 kernel 候选。 |
| 当前结论 | `DEVELOPMENT_FROZEN`：`gamma=0.25 + residual + physical + all` 在 seed=2026 e200 开发集为 `+0.7884 dB`、4/5 域正；LowLight `−0.1813 dB`，未做训练 seed/未触碰确认。 |
| 时间线位置 | T3 从通用机制退回 SB-native 对象形成；T5 冻结开发候选。 |
| 先看哪里 | [桥原生重估](./docs/BRIDGE_NATIVE_REASSESSMENT.md) → [形成史](./LINEAGE.md) → [e200 实验](../../../experiments/L2-medium-4090/EXP-L2-HNEK-SEARCH-E200-20260824/README.md) → [决策索引](./DECISION_INDEX.md)。 |

桥原生 horizon-normalized endpoint-kernel 候选。当前冻结 iteration：`gamma=0.25 + residual coordinate + physical horizon + all`。

当前状态：`DEVELOPMENT_FROZEN`，仅 single-seed paired-development，不是 confirmatory。

- 代码接入：`foundation/canonical/src/models/hnek/` 与 `hnek_search_model.py`
- 搜索证据：[EXP-L2-HNEK-SEARCH-E200-20260824](../../../experiments/L2-medium-4090/EXP-L2-HNEK-SEARCH-E200-20260824/README.md)
- 桥原生重估：[docs/BRIDGE_NATIVE_REASSESSMENT.md](./docs/BRIDGE_NATIVE_REASSESSMENT.md)
- 形成史：[DT/HJ/HNEK 互动史](../../synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md)
