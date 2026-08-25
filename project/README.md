# Project governance

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 定义仓库生命周期、稳定 ID、状态词汇、注册表和旧路径迁移规则的治理层。 |
| 当前结论 | 研究对象必须通过 `MOT → CAND/ITER → SEARCH → EXP → DEC → OUTPUT` 连接；文件存在或一次运行不能自动改变状态。 |
| 时间线位置 | T7–T8：在确定性基座复核后固化，用来承接最后一轮探索。 |
| 先看哪里 | [总时间线](./TIMELINE.md) → [生命周期](./LIFECYCLE.md) → [状态模型](./STATUS_MODEL.md) → [命名与 ID](./NAMING_AND_IDS.md)。 |

其它入口：注册表规则见 [REGISTRIES.md](./REGISTRIES.md)，历史路径见 [LEGACY_PATH_MAP.json](./LEGACY_PATH_MAP.json)，机器文件结构见 [schemas/](./schemas/README.md)。

## 模块说明规范

每个新生命周期模块或实体 README 至少回答：

1. **是什么**：职责和不负责的内容；
2. **当前结论**：允许怎样理解，不能怎样理解；
3. **时间线位置**：对应 [TIMELINE.md](./TIMELINE.md) 的阶段；
4. **阅读/行动入口**：真源、源码、实验、决策和下一步。

结论必须链接 EXP/DEC，不从目录名、旧报告或输出图反推。历史内容需要可追溯但不得重新进入当前注册表，除非通过新的实验和决策重新激活。
