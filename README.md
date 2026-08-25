# UNSB Research Lifecycle Repository

本仓库按研究生命周期组织 UNSB 无配对 All-in-One 多天气恢复工作。默认入口只呈现当前基座、活跃动机、候选状态和下一门禁；历史材料仍可追溯，但位于 `archive/`，不会干扰当前判断。

```text
canonical foundation
        ↓
motivation (MOT)
        ↓
candidate / iteration (CAND / ITER)
        ↓
frozen search controller (SEARCH)
        ↓
experiment L0 → L1 → L2 → L3 → L4
        ↓
decision (DEC)
        ↓
paper / figure / release / handoff
```

## 当前快照

- 新 deterministic canonical 已可作为后续探索基座。
- `MOT-001` 支持“共享训练改变条件路径几何”的现象级动机；六域单 seed 进一步观察到正的 shared-clock regret，但不支持固定普适窗口、因果伤害或算法有效性。
- DT、HJ 已在 clean deterministic 口径下关闭为当前主方法。
- HNEK `gamma=0.25` 是单 seed paired-development 候选，尚未确认。
- DCUM/LBST/PTQ/AEB 与状态恢复代码已由 `SEARCH-001` 接入分级搜索；L0 工程门禁已通过，但尚未产生效果裁决，不改变上述科学状态。

机器状态见 [PROJECT_STATE.json](./PROJECT_STATE.json)，人类可读摘要见 [CURRENT_STATE_CN.md](./CURRENT_STATE_CN.md)。

## 五分钟入口

1. [接任指南：先看什么、结论是什么、代码在哪里](./TAKEOVER_GUIDE_CN.md)
2. [项目总时间线](./project/TIMELINE.md)
3. [当前裁决全文](./decisions/CURRENT.md)
4. [确定性 canonical](./foundation/canonical/README.md)
5. [核心动机 MOT-001](./research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md)
6. [候选注册表](./research/candidates/README.md)
7. [当前搜索控制器](./research/searches/README.md)
8. [实验放大路径](./experiments/README.md)

## 目录职责

| 目录 | 职责 |
|---|---|
| `project/` | 生命周期、状态词汇、稳定 ID、旧路径映射 |
| `foundation/` | canonical baseline、确定性/数据/评估契约、公共 harness |
| `research/` | 动机实体、候选算法及其演进、跨候选研究综合 |
| `experiments/` | L0–L4 实验记录、协议、证据和裁决输入 |
| `decisions/` | 当前状态和不可变决策记录 |
| `outputs/` | 论文、最终图、发布包和 handoff 索引 |
| `archive/` | 早期探索、旧 prompt、被取代计划和旧布局说明 |

每个一级模块和当前研究实体的 README 都采用同一套接任摘要：**是什么、当前结论、时间线位置、阅读/行动入口**。新增模块时请沿用 [说明规范](./project/README.md#模块说明规范)。

## 基座验证

```bash
python -m pytest -q
python foundation/harness/self_test.py
python -m compileall -q foundation research
```

迁移前的中文路径和服务器路径保留在历史协议中作为 provenance。新代码和新文档应使用当前 ASCII 路径；映射见 [LEGACY_PATH_MAP.json](./project/LEGACY_PATH_MAP.json)。
