> **历史/已废弃**：本文是早期“最小 GPU 验证 + 成本门”计划，已被 `refactor/EXPERIMENT_PLAN.md` 取代；成本门也已被“成本不用考虑”的口径取代。仅作历史参考。

# 最小 GPU 验证计划（预注册，控制成本）

目标：确认重构后的干净实现保住或超过已知最好收益，并确定哪些 trick 是必要的。

## 前置

- 两个 subagent 的 SPEC / TRICK_LEDGER / 最小实现 / CPU 测试先通过。
- 只用 `seed=2026`、`128×128`，优先复用已有 warmup / plain / 冻结 checkpoint。

## 第一步：复现锚点（推理即可，不训练）

用冻结 checkpoint 重新跑 eval-off 推理与 metrics，确认 harness 能复现报告数字：

- DT：`plain@200` 与 best `dtcov_grouped_ramp5hold15decay25_l001_all6@200`，目标复现 18.7360 / 19.7800。
- HJ：`plain@200`、`true_constant@200`、`roll_constant@200` 在 val-O，目标复现 +1.4729 和归因 SSIM -0.0202。

这一步不训练，主要验证数据/指标/配对 bootstrap 的接线正确。

## 第二步：干净实现复现（每个算法 1 次训练）

- DT：从已有 warmup/plain prefix 出发，用干净实现跑一次“u_match 窗口 + 继续 plain 到 200”，eval-off 对比 best。
- HJ：从已有 `roll_ancestor@100`（或 plain prefix）出发，用干净实现跑一次 continuous layer0-HJ 到 200，val-O 对比。

## 第三步：knock-out 消融（每个算法最多 2 次）

每次只去掉/替换一个可疑 trick，看收益是否掉，逐步把 `unknown` 分类收敛成“必要 / 不必要”：

- DT 候选：`domain_time_ema` vs 全局 norm、`grouped_domain` vs none、冻结 teacher vs self。
- HJ 候选：`boundary_scale`、`min_risk`、`central_consensus` vs `onesided`、`no_flip`/确定性等。

具体消融顺序由 subagent 的 TRICK_LEDGER 决定，我最终拍板。

## 预算与停止

- 每个算法新训练 ≤ 3 次，总 GPU 约 6 小时封顶。
- 超预算、或收益掉了但原因查不清，就停并写进 PROGRESS。

## 判据

- 干净实现 eval-off 收益 >= 已知最好（容差内）→ 重构成立，继续写作。
- 干净实现掉收益 → 看消融定位是丢了必要 trick，还是原收益本身不稳定。
