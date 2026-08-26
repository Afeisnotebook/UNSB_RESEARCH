# UNSB 当前状态入口

> 更新：2026-08-27

这是根目录的一页式入口。完整科学边界、门禁和禁止表述见 [decisions/CURRENT.md](./decisions/CURRENT.md)，机器可读状态见 [PROJECT_STATE.json](./PROJECT_STATE.json)。

## 当前状态

| 对象 | 状态 | 当前含义 |
|---|---|---|
| canonical baseline | READY | 确定性实现已接受为新的实验基座 |
| MOT-001 | SUPPORTED_WITH_LIMITS | 共享训练改变路径几何；六域观察到正的 shared-clock regret；固定窗口不成立 |
| CAND-001 DT-CovMatch | CLOSED_NEGATIVE | clean deterministic rerun 未保住历史收益 |
| CAND-002 HJ finite-horizon | DEVELOPMENT_FROZEN / positive_but_fragile | discovery70 step1200 `+0.710548 dB`、6/6 域正；handoff 后 step1600 仍 `+0.660975 dB` |
| CAND-003 HNEK | FALLBACK-1 / positive_but_fragile | SEARCH-001 旧总冠军；最后三点均值仅 `+0.006322 dB`，现为递补 |
| CAND-004 search mechanisms | CLOSED_NEGATIVE | DCUM/LBST/PTQ/AEB 当前固定实现未形成稳定长程收益 |
| CAND-005 LTTR | CLOSED_NEGATIVE | tangent/pulse/direction 三个输出空间重推导均在 800 步反转 |
| SEARCH-001 clean directional | LOCAL_COMPLETE_HISTORICAL | 当时冻结 HNEK；已被 SEARCH-002 当前裁决取代 |
| SEARCH-002 DT/HJ re-derivation | LOCAL_COMPLETE_CANDIDATE_FROZEN | finite-horizon HJ 取代 HNEK 成为当前第一候选 |

## 下一门禁

SEARCH-002 已把当前第一候选冻结为 `finite-horizon HJ → plain handoff`。full-view 每域 100 张时固定 warmup 960 updates、HJ active `[960,4800)`、之后永久 plain；接下来只与 matched plain 在 4090 上运行 seed=2026 的 30k/60k/120k 固定里程碑，不根据中间结果改算法。confirmation20 继续封存。完整结果见 [SEARCH-002 报告](./experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/REPORT.md)。

六域动机结果使用单一新训练 seed=2051：三个 bridge time 的 cross-fit regret 分别为 `0.0201 / 0.0369 / 0.0164`，图像级 bootstrap 区间均在零以上。它支持域依赖有效相位和共享时钟错配的观察性表述，不支持跨训练 seed 稳定、因果恢复伤害或任何候选算法已经解决该问题。入口见 [MOT-001 阅读指南](./research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md)。
