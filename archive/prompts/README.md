# Historical prompts

## 接任摘要

| 是什么 | 当前结论 | 时间线位置 | 使用方式 |
|---|---|---|---|
| 历史服务器执行 prompt | 只记录当时任务边界，不是当前 runner 或状态真源 | T5–T7 前后 | 只用于还原上下文；复用前按当前路径、candidate spec 和决策重写 |

这些 prompt 冻结了当时的路径、环境和研究口径，不得直接作为当前 runner。需要复用时先按 `project/LEGACY_PATH_MAP.json` 和当前 candidate spec 重建。
