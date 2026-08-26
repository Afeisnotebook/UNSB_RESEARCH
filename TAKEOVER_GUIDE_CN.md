# UNSB 接任指南

> 更新：2026-08-26。本文回答新同事最先需要知道的三件事：仓库里有什么、已经得出了什么结论、这些内容位于研究时间线的哪一段。

## 先记住当前结论

1. 新 deterministic canonical 已通过本地真实数据微验证，是后续实验的共同父节点。
2. 动机已从“固定 Epoch 4–5 过度压缩”修正为：AIO 共享训练伴随域、bridge time、训练阶段和 seed 依赖的路径几何不同步。六域单训练种子实验进一步观察到正的 held-out shared-clock regret；它不是因果伤害或算法有效性证明。
3. DT 与 HJ 在 clean deterministic 单 seed 对比中没有保住历史收益，当前均为 `CLOSED_NEGATIVE`。
4. SEARCH-001 已完成本地统一竞争；HNEK `gamma=0.25` 是唯一候选，但最后三点均值仅 `+0.006322 dB`，分类为 `positive_but_fragile`，尚未确认。
5. DCUM/LBST/PTQ/AEB 当前固定实现没有保持长程正收益；HNEK+DCUM 与 DT 是递补，HJ 因轨迹振荡未晋级。

完整边界以 [当前科学裁决](./decisions/CURRENT.md) 为准。

## 生命周期地图

| 模块 | 它是什么 | 当前结论 | 时间线 | 接任入口 |
|---|---|---|---|---|
| `project/` | 生命周期、ID、状态和迁移规则 | 研究对象按 MOT→CAND→SEARCH→EXP→DEC→OUTPUT 流转 | T7–T8 固化 | [项目治理](./project/README.md) |
| `foundation/` | 全候选共享的 canonical 和 harness | 新确定性基座 READY | T5 发现问题，T7 接受新基座 | [Foundation](./foundation/README.md) |
| `research/motivations/` | 为什么值得继续研究 | 固定窗口关闭；路径几何不同步获有限支持 | T0 原始动机，T4–T6 重建 | [动机索引](./research/motivations/README.md) |
| `research/candidates/` | DT/HJ/HNEK 与新机制的身份、代码和演进 | HNEK 为本地脆弱总冠军；新机制当前实现关闭 | T1–T3、T5、T8–T9 | [候选索引](./research/candidates/README.md) |
| `research/searches/` | 冻结的筛选/合成控制器 | SEARCH-001 本地完成，待 4090 | T8–T9→下一阶段 | [搜索索引](./research/searches/README.md) |
| `research/synthesis/` | 跨候选的形成史与叙事综合 | 用于解释思路，不直接改变科学状态 | 覆盖 T0–T5 | [综合说明](./research/synthesis/README.md) |
| `experiments/` | 不可变协议、运行身份、证据和统计 | 已有 L0–L2；L3/L4 尚未形成新记录 | T4–T8 | [实验路径](./experiments/README.md) |
| `decisions/` | 当前裁决和不可变决策记录 | 是“能否继续、以何种身份继续”的真源 | 全时段，当前截至 T8 | [决策入口](./decisions/README.md) |
| `outputs/` | 经裁决允许的论文、图、release、handoff | 六域头图已有稳定副本；无最终方法 release | T6 后逐步形成 | [产出索引](./outputs/README.md) |
| `archive/` | 被取代、证伪或仅供考古的历史 | 可追溯，不参与当前状态计算 | 主要 T0–T4 | [历史边界](./archive/README.md) |

T0–T8 的日期、转折和证据链接见 [项目总时间线](./project/TIMELINE.md)。时间线表示**研究裁决顺序**，不保证与文件提交时间完全相同。

## 源码在哪里

| 对象 | 科学身份/说明 | 可执行源码 |
|---|---|---|
| plain deterministic UNSB | [canonical](./foundation/canonical/README.md) | `foundation/canonical/src/` |
| DT-CovMatch | [CAND-001](./research/candidates/CAND-001-dt-covmatch/README.md) | `research/candidates/CAND-001-dt-covmatch/dtcov/`；canonical 薄接入在 `foundation/canonical/src/models/dtcov_model.py` |
| HJ-PatchNCE | [CAND-002](./research/candidates/CAND-002-hj-patchnce/README.md) | `research/candidates/CAND-002-hj-patchnce/hj/`；canonical 薄接入在 `foundation/canonical/src/models/hj_model.py` |
| HNEK | [CAND-003](./research/candidates/CAND-003-hnek/README.md) | `foundation/canonical/src/models/hnek/` 与 `hnek_search_model.py` |
| DCUM/LBST/PTQ/AEB | [CAND-004](./research/candidates/CAND-004-search-mechanisms/README.md) | DCUM：`foundation/canonical/src/data/unaligned_dataset.py`；其余三者：`foundation/canonical/src/models/sb_model.py`；控制器：`research/searches/SEARCH-001-clean-directional/` |

候选目录定义“它是谁、为什么出现、现状如何”；canonical 中的 model 文件定义“当前基座如何执行它”。不要仅凭文件存在判断候选有效。

## 新同事第一天建议

1. 读本文、[当前裁决](./decisions/CURRENT.md) 和 [MOT-001 阅读指南](./research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md)。
2. 运行根 README 的三项基座检查；这验证工程契约，不代替 GPU 实验。
3. 若接算法：先读对应 candidate 的 README、lineage、experiment index、decision index，再看源码。
4. 若开实验：从 [实验模板](./experiments/_template/README.md) 建新 ID；输出写入新的实验目录或 Git 忽略的 `runs/`，不得覆盖冻结证据。
5. 若要改变状态：先登记 EXP，再新增 DEC；SEARCH 运行结果和论文图都不能自行改变候选状态。

## 冲突时相信谁

机器可读实验裁决 → `decisions/CURRENT.md` / decision record → 冻结报告 → 模块 README → 历史叙事与计划。历史文档中的正收益、旧路径和旧术语都不能越过这条权威顺序。
