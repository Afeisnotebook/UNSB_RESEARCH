# CAND-001: DT-CovMatch

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 将 endpoint disagreement 的绝对尺度改为 domain-time 校准、冻结 teacher 与短窗口一致性的候选家族。 |
| 当前结论 | `CLOSED_NEGATIVE`：clean deterministic seed=2026 下最佳分支相对自身 plain 为 `−0.2677 dB`。历史正收益不能作为当前证据。 |
| 时间线位置 | T1 由早期 covariance/u_match 主线形成；T5 统一 clean rerun 后关闭。 |
| 先看哪里 | [规格](./SPEC.md) → [形成史](./LINEAGE.md) → [clean 实验](../../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/README.md) → [决策索引](./DECISION_INDEX.md)。 |

来源：MOT-001 的 endpoint disagreement / covariance 主线。用户固定协方差研究轴，Codex根据跨域 U 尺度冲突形成 domain-time calibration、frozen teacher 和短时正则实现。

当前状态：`CLOSED_NEGATIVE`。实现和历史规格保留用于追溯，不作为下一轮默认方法。

- 规格：[SPEC.md](./SPEC.md)
- 报告：[REPORT.md](./REPORT.md)
- clean-core 实验：[EXP-L1-DT-HJ-CLEAN-CORE-20260824](../../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/README.md)
- 形成史：[DT/HJ/HNEK 互动史](../../synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md)
