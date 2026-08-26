# DEC-20260827: freeze finite-horizon HJ as the next local candidate

## 决定

将 CAND-002 的 `ITER-007-finite-horizon-handoff` 冻结为当前本地第一候选，分类 `positive_but_fragile`；CAND-003 HNEK 下调为递补一。下一步只允许 full-view matched 4090 验证，不再进行本地 HJ 强度、layer、方向或窗口搜索。

## 依据

- SEARCH-001 原 HJ step1200 在 discovery10 为 `+0.804544 dB`、6/6 域正。
- 未参与 screen 的 discovery70 共 420 张上，matched delta 为 `+0.710548 dB`，6/6 域正，SSIM `+0.020316`，LPIPS `-0.034900`，最差域仍 `+0.174754 dB`。
- HJ 在 step1200 永久关闭后，继续 400 次完全 plain 更新，step1600 仍为 `+0.660975 dB`；这支持“早期方向导航改变可延续优化状态”。
- step2000 为 `+3.791830 dB`，但主要由 matched plain 后期坍塌放大，只作为抗坍塌证据，不作为预期平均收益。
- 输出空间 LTTR tangent、one-epoch pulse、direction barrier 均在 800 步反转，已关闭。

## 冻结配置

HJ 核心仍为 layer0、joint structure direction、central consensus、strength 0.5、boundary scale 0.001、min risk 0.05。唯一结构变化是按真实数据曝光定义的有限窗口：warmup 1.6 epochs、HJ steering 6.4 epochs、第 8.0 epoch handoff 给 plain。full view 每域 100 张对应 `[960,4800)` updates。

## 风险与边界

只有 seed=2026；本地训练仅每域 25 张；discovery 已用于选择；LowLight 在 handoff 后转负；confirmation20 未打开。因此不得称为 confirmed/stable/generalized，也不得把 2000 步 `+3.79 dB` 当作跨环境预期收益。

证据：[SEARCH-002 实验](../../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/README.md)；[唯一候选](../../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/CANDIDATE.json)。
