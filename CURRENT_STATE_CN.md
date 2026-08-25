# UNSB 当前状态入口

> 更新：2026-08-26

这是根目录的一页式入口。完整科学边界、门禁和禁止表述见 [decisions/CURRENT.md](./decisions/CURRENT.md)，机器可读状态见 [PROJECT_STATE.json](./PROJECT_STATE.json)。

## 当前状态

| 对象 | 状态 | 当前含义 |
|---|---|---|
| canonical baseline | READY | 确定性实现已接受为新的实验基座 |
| MOT-001 | SUPPORTED_WITH_LIMITS | 共享训练改变路径几何；六域观察到正的 shared-clock regret；固定窗口不成立 |
| CAND-001 DT-CovMatch | CLOSED_NEGATIVE | clean deterministic rerun 未保住历史收益 |
| CAND-002 HJ-PatchNCE | CLOSED_NEGATIVE | 相对 plain 的 clean 收益基本消失 |
| CAND-003 HNEK | DEVELOPMENT_FROZEN | `gamma=0.25` 单 seed 开发候选，未确认 |
| CAND-004 search mechanisms | IMPLEMENTED | DCUM/LBST/PTQ/AEB 已接入 SEARCH-001，尚无实验裁决 |
| SEARCH-001 clean directional | ENGINEERING_GATE_PASS | full-state/resume/评估/PTQ 门禁已登记，下一步为 stage1 |

## 下一门禁

`SEARCH-001 --stage gate` 已在本地真实数据上通过并登记为 L0 实验；现在进入 stage1 方向筛选。历史 PNG 字节重放不是新 canonical 的前置条件。

六域动机结果使用单一新训练 seed=2051：三个 bridge time 的 cross-fit regret 分别为 `0.0201 / 0.0369 / 0.0164`，图像级 bootstrap 区间均在零以上。它支持域依赖有效相位和共享时钟错配的观察性表述，不支持跨训练 seed 稳定、因果恢复伤害或任何候选算法已经解决该问题。入口见 [MOT-001 阅读指南](./research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md)。
