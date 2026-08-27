# Experiment scale-up path

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 按成本与证据强度保存不可变实验身份、冻结协议、运行证据和裁决输入。 |
| 当前结论 | L1 已冻结 SEARCH-005 无持续候选结果；L2 保存历史中型证据；尚无 L3/L4 算法确认。 |
| 时间线位置 | T4–T11 形成现有记录；没有对象自动进入确认层。 |
| 先看哪里 | 先按下表选等级，再进入该层 README；当前状态以 [决策](../decisions/CURRENT.md) 为准。 |

| Level | 职责 | 当前记录 |
|---|---|---|
| L0 contract | 单测、确定性、默认等价、数据身份 | [canonical micro / SEARCH gate](./L0-contract/README.md) |
| L1 local | 本地机制筛选与 clean core | [motivation、DT/HJ](./L1-local/README.md) |
| L2 medium-4090 | 中型判别、长程轨迹 | [motivation six-domain、HNEK e200](./L2-medium-4090/README.md) |
| L3 scale-5090 | 大规模迁移 | [当前无 layout-v2 冻结记录](./L3-scale-5090/README.md) |
| L4 confirmation | 多 seed、未触碰确认 | [当前为空](./L4-confirmation/README.md) |

每个新实验应包含 `experiment.json`、冻结 protocol/identity、evidence、`adjudication.json` 和外部 artifacts 清单。

正在运行的 `runs/` 目录不是实验记录；只有运行完成、身份核验并以新 EXP ID 冻结后才可提交。模板见 [`_template/`](./_template/README.md)。
