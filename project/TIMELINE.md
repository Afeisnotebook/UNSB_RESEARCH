# UNSB 项目总时间线

> 这是一条研究裁决时间线：它记录为什么转向、什么被证伪、什么仍存活。部分工作并行发生，因此不等同于严格的 Git commit 顺序。

| 阶段 | 日期（2026） | 做了什么 | 到今天的结论 | 主要入口 |
|---|---|---|---|---|
| T0 原始问题 | 06-22–06-26 | 复现原 UNSB，做 AIO joint 与 single 严格消融 | AIO 退化构成原始研究动机；旧数值只作历史定性证据 | [早期搜寻](../archive/early-search/README.md) |
| T1 covariance 主线 | 06-27–07-10 | 收敛 uncertainty 术语、排除 TTO/UA 伪增益，形成 u_match 与 DT-CovMatch | proxy 口径保留；DT 思路进入候选，早期收益需确定性复核 | [早期时间线](../archive/early-search/docs/TIMELINE.md)、[CAND-001](../research/candidates/CAND-001-dt-covmatch/README.md) |
| T2 HJ 演进 | 07-15–07-26 | 从 patch 降权、gradient risk、structure harm 收缩到 layer-0 harmful-joint PatchNCE | 多轮设计未形成干净稳定收益 | [CAND-002](../research/candidates/CAND-002-hj-patchnce/README.md)、[互动史](../research/synthesis/DT_HJ_HNEK_CODEX_INTERACTION_HISTORY_CN.md) |
| T3 返回 SB 原生对象 | 07 月末–08 月中 | 排除通用 routing/梯度手术式包装，提出 horizon-normalized endpoint kernel | HNEK 成为唯一进入长期开发延伸的老候选 | [CAND-003](../research/candidates/CAND-003-hnek/README.md) |
| T4 动机重启 | 08-18–08-24 | 用 plain Single/AIO 和 6 seeds 重建五域路径几何证据 | 固定 Epoch 4–5 窗口关闭；阶段/seed/域依赖差异保留 | [MOT-001](../research/motivations/MOT-001-aio-path-geometry/README.md)、[窗口实验](../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/README.md) |
| T5 确定性 clean core | 08-24 | 去除原实现不确定来源，在同一 deterministic core 上复核 DT/HJ/HNEK | DT `−0.2677`、HJ 相对 plain `+0.0381`，均关闭；HNEK e200 `+0.7884` 仅开发冻结 | [算法裁决](../decisions/records/DEC-20260824-ALGORITHM-STATUS.md) |
| T6 动机放大 | 08-24–08-26 | 五域三 split 复核，并恢复 RainDS 做六域 held-out phase/shared-clock 统计 | `SUPPORTED_SIXDOMAIN_SHARED_CLOCK_REGRET`；仍只有训练 seed=2051，不是因果或方法证明 | [阅读指南](../research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md)、[六域实验](../experiments/L2-medium-4090/EXP-L2-MOTIVATION-SIXDOMAIN-20260824/README.md) |
| T7 新基座验收 | 08-25–08-26 | 审核 clean re-exploration 包，拒绝门禁不全的运行；对新 canonical 做本地真实数据微验证 | 包内结果不更新科学结论；新 deterministic canonical 被接受 | [包审计](../decisions/records/DEC-20260826-CLEAN-REEXPLORATION-AUDIT.md)、[基座验收](../decisions/records/DEC-20260826-NEW-CANONICAL-ACCEPTANCE.md) |
| T8 生命周期化与新搜索 | 08-26 | 仓库重构；接入 DCUM/LBST/PTQ/AEB 和 SEARCH-001；完成 L0 工程门 | 新机制 `IMPLEMENTED`、搜索 `ENGINEERING_GATE_PASS`，均无效果裁决 | [SEARCH-001](../research/searches/SEARCH-001-clean-directional/README.md)、[L0 gate](../experiments/L0-contract/EXP-L0-SEARCH-001-GATE-20260826/README.md) |
| T9 本地方向再搜索 | 08-26 | 八 lane、两条合成、完整视图复赛；首名与 plain 等量延长至 12k | HNEK 以最后三点 `+0.006322 dB` 冻结为 `positive_but_fragile`；新机制当前实现关闭 | [L1 完整实验](../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/README.md)、[本地裁决](../decisions/records/DEC-20260826-SEARCH-001-LOCAL-WINNER.md) |
| T10 DT/HJ 方向重推导 | 08-27 | 证伪输出空间 LTTR；用独立 discovery70 扩展 HJ 强 checkpoint；实现有限期 HJ→plain handoff | HJ discovery70 `+0.710548 dB`、6/6 域正，handoff 后仍正；finite-horizon HJ 取代 HNEK 成为脆弱第一候选 | [SEARCH-002 实验](../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/README.md)、[候选裁决](../decisions/records/DEC-20260827-HJ-FINITE-HORIZON-LOCAL-CANDIDATE.md) |
| T11 反转因果与数学算子发现 | 08-27–08-28 | 修正 SEARCH-003 的 controller 目标漂移；SEARCH-005 运行 6 类 G1 机制、4 次因果修订和独立 2400-step 轨迹 | 正窗口可复现，但没有持续候选；PCOA 仅 weak fallback，full100/多 seed/confirmation20 未启动 | [SEARCH-005](../research/searches/SEARCH-005-long-horizon-operator-discovery/RESULTS.md)、[路线一裁决](../decisions/records/DEC-20260828-SEARCH005-ROUTE1-STOP.md) |
| NEXT 显式路线选择 | T11 之后 | 二选一：接受 weak-fallback 风险后高算力证伪 PCOA，或另立 route-2 gap-aware handoff | 不允许把 route 2、窗口阈值或 paired 控制偷偷写回 SEARCH-005 | [当前门禁](../decisions/CURRENT.md) |

## 三条主线如何汇合

```text
T0–T1 原始 AIO 退化与 covariance proxy ──→ DT
T2 patch harmful-joint 探索              ──→ HJ
T3 SB-native 重估                         ──→ HNEK
                         ↓ T5 clean deterministic 统一裁决
T4–T6 plain-only 动机重建 ───────────────→ 候选应解释的新验收对象
                         ↓ T7 新 canonical
                         ↓ T8 SEARCH-001 与最后一轮分级探索
                         ↓ T9 HNEK 本地脆弱正向候选
                         ↓ T10 finite-horizon HJ 形成历史强窗口
                         ↓ T11 SEARCH-005 路线一无持续候选，PCOA 仅 weak fallback
```

“动机成立”不推出“候选有效”；“工程门通过”也不推出“效果成立”。两者都必须在独立冻结实验中连接到决策。
