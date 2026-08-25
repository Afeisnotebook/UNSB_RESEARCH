# Schemas

## 接任摘要

| 是什么 | 当前结论 | 时间线位置 | 下一步 |
|---|---|---|---|
| registry/state/identity 的机器契约说明区 | 当前采用轻量 JSON + 测试约束，尚未引入正式 JSON Schema | T8 仓库治理层 | 自动化增加且字段稳定后再补 schema，不为形式化而提前锁死研究字段 |

当前 registry 使用轻量 JSON 契约。新增自动化前可在此补正式 JSON Schema；在此之前由 pytest、JSON parse 和 link check 保证基本一致性。
