# Candidate registry

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 保存每个算法家族的稳定身份、源码、形成史、迭代、实验索引和决策索引。 |
| 当前结论 | SEARCH-005 后没有持续收益晋级候选；finite-horizon HJ/HNEK 保留为历史正窗口，PCOA 仅是搜索级 weak fallback。 |
| 时间线位置 | DT=T1，HJ=T2/T10，HNEK=T3，clean 统一裁决=T5，新机制=T8，算子发现=T11。 |
| 先看哪里 | 先看下表状态，再进入 candidate README；跨候选形成过程见 [互动史](../synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md)。 |

| ID | 候选 | 当前状态 | 当前 iteration |
|---|---|---|---|
| CAND-001 | DT-CovMatch | CLOSED_NEGATIVE | clean deterministic core |
| CAND-002 | HJ-PatchNCE | HISTORICAL POSITIVE WINDOW | finite-horizon HJ → plain handoff；不再自动晋级 |
| CAND-003 | HNEK | HISTORICAL OSCILLATORY FALLBACK | gamma=0.25 physical residual |
| CAND-004 | Search mechanisms | CLOSED_NEGATIVE | DCUM/LBST/PTQ/AEB 当前实现未保持长程收益 |
| CAND-005 | LTTR | CLOSED_NEGATIVE | tangent/pulse/direction 均在 800 步反转 |

状态来自 [决策账本](../../decisions/DECISION_LEDGER.json)，不是由历史最好数字自动产生。

源码位置和接任顺序也汇总在 [根接任指南](../../TAKEOVER_GUIDE_CN.md#源码在哪里)。新候选应复制 `_template/`，不得直接把实验临时代码写入 canonical 后再补身份。
