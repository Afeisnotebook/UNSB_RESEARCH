# UNSB 当前状态入口

> 更新：2026-08-28

这是根目录的一页式入口。完整科学边界、门禁和禁止表述见 [decisions/CURRENT.md](./decisions/CURRENT.md)，机器可读状态见 [PROJECT_STATE.json](./PROJECT_STATE.json)。

## 当前状态

| 对象 | 状态 | 当前含义 |
|---|---|---|
| canonical baseline | READY | 确定性实现已接受为新的实验基座 |
| MOT-001 | SUPPORTED_WITH_LIMITS | 共享训练改变路径几何；六域观察到正的 shared-clock regret；固定窗口不成立 |
| CAND-001 DT-CovMatch | CLOSED_NEGATIVE | clean deterministic rerun 未保住历史收益 |
| CAND-002 HJ finite-horizon | HISTORICAL POSITIVE WINDOW | discovery70 有强窗口，但属于 handoff 子路线；SEARCH-005 后不再自动晋级 |
| CAND-003 HNEK | HISTORICAL OSCILLATORY FALLBACK | 8k/10k/12k 均值仅 `+0.006322 dB`，且长程反复换号 |
| CAND-004 search mechanisms | CLOSED_NEGATIVE | DCUM/LBST/PTQ/AEB 当前固定实现未形成稳定长程收益 |
| CAND-005 LTTR | CLOSED_NEGATIVE | tangent/pulse/direction 三个输出空间重推导均在 800 步反转 |
| SEARCH-001 clean directional | LOCAL_COMPLETE_HISTORICAL | 当时冻结 HNEK；已被 SEARCH-002 当前裁决取代 |
| SEARCH-002 DT/HJ re-derivation | HISTORICAL SUBROUTE COMPLETE | 当时 finite-horizon HJ 取代 HNEK；已由 SEARCH-005 当前裁决覆盖 |
| SEARCH-003 evidence-guided controller | SUBROUTE_COMPLETE | whole-branch/controller 子路线关闭；不代表数学算子发现已穷尽 |
| SEARCH-005 operator discovery | ROUTE1_COMPLETE / NO SUSTAINED CANDIDATE | PCOA 仅为 weak fallback；没有候选获 full100/多 seed/confirmation20 资格 |

## 下一门禁

当前没有自动进入 4090 的第一候选。SEARCH-005 的 PCOA 在 400/800/1200 为正，但 1600/2400 反转，只能作为 weak fallback；其 full100 命令仅供明确接受风险后的高算力证伪。若转向“算法撤出后如何让 plain 接手”，必须另立 route-2 gap-aware handoff 计划，不能把它写成 SEARCH-005 的成功延续。confirmation20 继续封存。完整结果见 [SEARCH-005 报告](./research/searches/SEARCH-005-long-horizon-operator-discovery/RESULTS.md)。

六域动机结果使用单一新训练 seed=2051：三个 bridge time 的 cross-fit regret 分别为 `0.0201 / 0.0369 / 0.0164`，图像级 bootstrap 区间均在零以上。它支持域依赖有效相位和共享时钟错配的观察性表述，不支持跨训练 seed 稳定、因果恢复伤害或任何候选算法已经解决该问题。入口见 [MOT-001 阅读指南](./research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md)。
