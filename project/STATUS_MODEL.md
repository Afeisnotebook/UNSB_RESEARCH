# Status model

## Motivation

`PROPOSED`、`SUPPORTED_WITH_LIMITS`、`SUPPORTED`、`REJECTED`、`ARCHIVED`。

## Candidate

`INCUBATING`、`IMPLEMENTED`、`DEVELOPMENT_FROZEN`、`HELD`、`CLOSED_NEGATIVE`、`CONFIRMED`、`ARCHIVED`。

## Local result classification

`STRONG_LOCAL_SIGNAL`、`POSITIVE_BUT_FRAGILE`、`WEAK_FALLBACK`、`COMPUTE_ONLY_SIGNAL`。

结果分类描述一次冻结搜索的证据强度，不替代候选生命周期状态。例如本地总冠军可以同时是候选状态 `DEVELOPMENT_FROZEN` 和结果分类 `POSITIVE_BUT_FRAGILE`。

## Decision outcome

`PROMOTE`、`REVISE`、`HOLD`、`REJECT`、`FREEZE_DEVELOPMENT`、`CONFIRM`、`SUPERSEDE`。

状态只能由 `decisions/DECISION_LEDGER.json` 中的记录改变，不能从文件夹名称、最好 checkpoint 或单个数字推测。
