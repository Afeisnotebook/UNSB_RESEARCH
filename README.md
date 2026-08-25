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
- `MOT-001` 支持“共享训练改变条件路径几何”的现象级动机，不支持固定普适窗口。
- DT、HJ 已在 clean deterministic 口径下关闭为当前主方法。
- HNEK `gamma=0.25` 是单 seed paired-development 候选，尚未确认。
- commit `495a092` 保存了新一轮 DCUM/LBST/PTQ/AEB 与状态恢复代码；它们已由 `SEARCH-001` 接入分级搜索，但尚未产生实验裁决，不改变上述科学状态。

机器状态见 [PROJECT_STATE.json](./PROJECT_STATE.json)，人类可读摘要见 [CURRENT_STATE_CN.md](./CURRENT_STATE_CN.md)。

## 五分钟入口

1. [当前裁决全文](./decisions/CURRENT.md)
2. [确定性 canonical](./foundation/canonical/README.md)
3. [核心动机 MOT-001](./research/motivations/MOT-001-aio-path-geometry/README.md)
4. [候选注册表](./research/candidates/README.md)
5. [当前搜索控制器](./research/searches/README.md)
6. [实验放大路径](./experiments/README.md)
7. [决策账本](./decisions/README.md)
8. [最终产出](./outputs/README.md)

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

## 基座验证

```bash
python -m pytest -q
python foundation/harness/self_test.py
python -m compileall -q foundation research
```

迁移前的中文路径和服务器路径保留在历史协议中作为 provenance。新代码和新文档应使用当前 ASCII 路径；映射见 [LEGACY_PATH_MAP.json](./project/LEGACY_PATH_MAP.json)。
