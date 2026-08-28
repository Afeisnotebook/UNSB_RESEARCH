# UNSB 当前状态入口

> 更新：2026-08-28

这是根目录的一页式入口。完整科学边界、门禁和禁止表述见 [decisions/CURRENT.md](./decisions/CURRENT.md)，机器可读状态见 [PROJECT_STATE.json](./PROJECT_STATE.json)。

## 当前状态

| 对象 | 状态 | 当前含义 |
|---|---|---|
| canonical baseline | READY | 确定性实现已接受为新的实验基座 |
| MOT-001 | SUPPORTED_WITH_LIMITS | 共享训练改变路径几何；六域观察到正的 shared-clock regret；固定窗口不成立 |
| CAND-001 DT-CovMatch | CLOSED_NEGATIVE | clean deterministic rerun 未保住历史收益 |
| CAND-002 HJ finite-horizon | ROUTE2_SUSTAINED_LOCAL | HJ `[240,1200)` 后完整状态交给 native UNSB；到 total step 3200 的晚三点 `+1.180 dB`，待 full100/多 seed |
| CAND-003 HNEK | HISTORICAL OSCILLATORY FALLBACK | 8k/10k/12k 均值仅 `+0.006322 dB`，且长程反复换号 |
| CAND-004 search mechanisms | CLOSED_NEGATIVE | DCUM/LBST/PTQ/AEB 当前固定实现未形成稳定长程收益 |
| CAND-005 LTTR | CLOSED_NEGATIVE | tangent/pulse/direction 三个输出空间重推导均在 800 步反转 |
| SEARCH-001 clean directional | LOCAL_COMPLETE_HISTORICAL | 当时冻结 HNEK；已被 SEARCH-002 当前裁决取代 |
| SEARCH-002 DT/HJ re-derivation | HISTORICAL SUBROUTE COMPLETE | 当时 finite-horizon HJ 取代 HNEK；已由 SEARCH-005 当前裁决覆盖 |
| SEARCH-003 evidence-guided controller | SUBROUTE_COMPLETE | whole-branch/controller 子路线关闭；不代表数学算子发现已穷尽 |
| SEARCH-005 operator discovery | ROUTE1_COMPLETE / NO SUSTAINED CANDIDATE | PCOA 仅为 weak fallback；没有候选获 full100/多 seed/confirmation20 资格 |
| SEARCH-004 gap-aware handoff | ROUTE2_COMPLETE / SUSTAINED LOCAL | HJ native handoff 为唯一 4090 第一候选；LCNMP/VCMR 均未胜过直接完整状态交接 |

## 下一门禁

当前有且只有一个自动进入下一门禁的第一候选：`HJ1200-NATIVE-HANDOFF`。它在 small25、seed=2026 上经过 2000 次 native 接续更新后仍为 `+0.871 dB`、6/6 域正，晚三点平均 `+1.180 dB`，并通过 SSIM/LPIPS 与回撤护栏。下一步必须用冻结的 exposure-normalized 配置从 e0 做 full100 4090 matched 验证；不得根据 30k/60k 中间结果改算法或 handoff 点。它尚未跨 seed，confirmation20 继续封存。完整结果见 [SEARCH-004 报告](./research/searches/SEARCH-004-gap-aware-handoff/RESULTS.md)。

六域动机结果使用单一新训练 seed=2051：三个 bridge time 的 cross-fit regret 分别为 `0.0201 / 0.0369 / 0.0164`，图像级 bootstrap 区间均在零以上。它支持域依赖有效相位和共享时钟错配的观察性表述，不支持跨训练 seed 稳定、因果恢复伤害或任何候选算法已经解决该问题。入口见 [MOT-001 阅读指南](./research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md)。
