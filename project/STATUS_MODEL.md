# Status model

## Motivation

`PROPOSED`、`SUPPORTED_WITH_LIMITS`、`SUPPORTED`、`REJECTED`、`ARCHIVED`。

## Candidate

`INCUBATING`、`IMPLEMENTED`、`DEVELOPMENT_FROZEN`、`HELD`、`CLOSED_NEGATIVE`、`CONFIRMED`、`ARCHIVED`。

`CLOSED_NEGATIVE` 仅为旧版兼容状态，不能继续承担科学结论。SEARCH-003
起采用三个正交证据字段：

- `protocol_status`: `active`、`closed_current_protocol`、`superseded`；
- `mechanism_status`: `open`、`supported_with_limits`、`mechanism_falsified`；
- `trajectory_status`: `not_audited`、`reversal_observed`、`locally_sustained`、`seed_sustained`。

例如，一个固定实现可以同时是 `closed_current_protocol` 和
`reversal_observed`，但其父机制仍为 `open`。只有针对机制本身的判死实验通过，才允许
写 `mechanism_falsified`；“长程相对收益反转”本身不足以推出该结论。

## Local result classification

`STRONG_LOCAL_SIGNAL`、`POSITIVE_BUT_FRAGILE`、`WEAK_FALLBACK`、`COMPUTE_ONLY_SIGNAL`。

结果分类描述一次冻结搜索的证据强度，不替代候选生命周期状态。例如本地总冠军可以同时是候选状态 `DEVELOPMENT_FROZEN` 和结果分类 `POSITIVE_BUT_FRAGILE`。

## Decision outcome

`PROMOTE`、`REVISE`、`HOLD`、`REJECT`、`FREEZE_DEVELOPMENT`、`CONFIRM`、`SUPERSEDE`。

状态只能由 `decisions/DECISION_LEDGER.json` 中的记录改变，不能从文件夹名称、最好 checkpoint 或单个数字推测。
