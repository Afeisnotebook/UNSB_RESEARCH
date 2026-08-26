# EXP-L1-SEARCH-002-DTHJ-20260827

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | seed=2026 下重新提炼 DT/HJ 思想，并对 HJ 有限期导航做 matched 本地验证。 |
| 当前结论 | `finite-horizon HJ → plain handoff` 是当前唯一候选，分类 `positive_but_fragile`。 |
| 关键证据 | discovery70（420 张）在 1200 步为 `+0.710548 dB`、6/6 域正；关闭 HJ 后到 1600 仍为 `+0.660975 dB`。 |
| 数据边界 | 训练视图每域 25 张；评估只用 discovery；`confirmation20_opened=false`。 |
| 先看哪里 | [报告](./REPORT.md) → [唯一候选](./CANDIDATE.json) → [机器结果](./RESULTS.json) → [协议锁](./PROTOCOL_LOCK.json)。 |

大型 checkpoint 与逐图 raw metrics 留在本机 `E:\UNSB_Expl\runs\dthj_rederivation_20260826`，Git 只保存可审计汇总。
