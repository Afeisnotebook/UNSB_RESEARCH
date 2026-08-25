# UNSB 当前状态入口

> 更新：2026-08-26

这是根目录的一页式入口。完整科学边界、门禁和禁止表述见 [decisions/CURRENT.md](./decisions/CURRENT.md)，机器可读状态见 [PROJECT_STATE.json](./PROJECT_STATE.json)。

## 当前状态

| 对象 | 状态 | 当前含义 |
|---|---|---|
| canonical baseline | READY | 确定性实现已接受为新的实验基座 |
| MOT-001 | SUPPORTED_WITH_LIMITS | 共享训练改变路径几何；固定窗口不成立 |
| CAND-001 DT-CovMatch | CLOSED_NEGATIVE | clean deterministic rerun 未保住历史收益 |
| CAND-002 HJ-PatchNCE | CLOSED_NEGATIVE | 相对 plain 的 clean 收益基本消失 |
| CAND-003 HNEK | DEVELOPMENT_FROZEN | `gamma=0.25` 单 seed 开发候选，未确认 |
| CAND-004 search mechanisms | IMPLEMENTED | DCUM/LBST/PTQ/AEB 已接入 SEARCH-001，尚无实验裁决 |
| SEARCH-001 clean directional | IMPLEMENTED_UNRUN | 搜索协议与工程门禁已实现，尚未登记运行证据 |

## 下一门禁

先在本地真实数据上运行 `SEARCH-001 --stage gate`，登记为新的 L0 实验；通过后才进入 stage1 方向筛选。昂贵训练必须由显式决策推进，历史 PNG 字节重放不是新 canonical 的前置条件。
