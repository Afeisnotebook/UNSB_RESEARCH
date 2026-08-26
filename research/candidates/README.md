# Candidate registry

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 保存每个算法家族的稳定身份、源码、形成史、迭代、实验索引和决策索引。 |
| 当前结论 | SEARCH-001 本地完成：HNEK 为 `positive_but_fragile` 总冠军；DT/HJ 未获保护名额；四个新机制当前实现关闭为负。 |
| 时间线位置 | DT=T1，HJ=T2，HNEK=T3，clean 统一裁决=T5，新机制=T8。 |
| 先看哪里 | 先看下表状态，再进入 candidate README；跨候选形成过程见 [互动史](../synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md)。 |

| ID | 候选 | 当前状态 | 当前 iteration |
|---|---|---|---|
| CAND-001 | DT-CovMatch | CLOSED_NEGATIVE | clean deterministic core |
| CAND-002 | HJ-PatchNCE | CLOSED_NEGATIVE | layer-0 harmful-joint core |
| CAND-003 | HNEK | DEVELOPMENT_FROZEN / positive_but_fragile | gamma=0.25 physical residual；待 4090 |
| CAND-004 | Search mechanisms | CLOSED_NEGATIVE | DCUM/LBST/PTQ/AEB 当前实现未保持长程收益 |

状态来自 [决策账本](../../decisions/DECISION_LEDGER.json)，不是由历史最好数字自动产生。

源码位置和接任顺序也汇总在 [根接任指南](../../TAKEOVER_GUIDE_CN.md#源码在哪里)。新候选应复制 `_template/`，不得直接把实验临时代码写入 canonical 后再补身份。
