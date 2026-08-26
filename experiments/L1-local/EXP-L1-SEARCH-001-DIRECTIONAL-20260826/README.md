# EXP-L1-SEARCH-001-DIRECTIONAL-20260826

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | seed=2026、六域真实数据、matched plain 下的 SEARCH-001 完整本地方向筛选。 |
| 当前结论 | HNEK 为唯一冻结候选，分类为 `positive_but_fragile`；最后三个 checkpoint 的宏平均 delta 仅 `+0.006322 dB`。 |
| 数据边界 | 使用 discovery；`confirmation20_opened=false`。 |
| 先看哪里 | [中文报告](./REPORT.md) → [唯一候选](./CANDIDATE.json) → [完整轨迹](./RESULTS.json) → [协议锁](./PROTOCOL_LOCK.json)。 |

该实验包含工程门禁后的全部 stage1、两个 synthesis、stage2 全视图复赛以及 HNEK/plain 到 12k updates 的等量延长。`RESULTS.json` 保存每个 checkpoint 的宏平均与六域 delta；大型 checkpoint 和 raw 日志留在运行机，不纳入 Git。
