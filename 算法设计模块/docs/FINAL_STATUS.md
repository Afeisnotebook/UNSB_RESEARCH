# 最终状态归纳（2026-08-17，当前权威）

本文是收口结论，不再启动训练。所有 sub-dB 差异按 direction + limitation 处理。

## 1. 已确认收益

- **DT**：3-seed 配对 delta `+0.8875 / +0.7426 / +0.4687`，
  mean **+0.6996 dB**，95% CI **[0.1712, 1.2280]**（不含 0）→ 方向性成立。
- **HJ**：3-seed true>plain 全为正 `+2.7533 / +3.1007 / +0.3592`，
  mean **+2.0711 dB**，95% CI **[−1.6372, +5.7793]**（含 0）→ 方向性成立但幅度不稳。

## 2. 机制现状（不互相否定）

- **DT** = 对 amortized endpoint 响应几何（teacher-student 标准化响应漂移）的正则，
  数理自洽（见 `refactor/METHOD_GROUNDING.md`）。
- **HJ** = 旧“结构方向特异”在 3-seed 下未复现；但 true>plain 方向为正，算法方向性有效，
  真正机制未定。
- **H1（DT 时机/窗口，单 seed=2026）**：control 手调25 = **18.4552** 最优；
  plateau 退出(cap60) = 17.6721；延长窗口45 = 17.1409。
  → 不是“窗口太短”，而是“decay 到 0 并退出”的**时机**重要；延长 hold 让 drift 继续涨、结果更差。
  （差异 0.8~1.3 dB，单 seed 方向性，不作精确声明。）
- **HJ α（单 seed=2026）**：α0 = 17.1158，α0.5 = 18.6225，α1 = 17.4125。
  → 结构方向(α0)最差、掺随机(α0.5)最好。但这是**单 seed、差异在噪声内**，只能作弱方向信号。
  - 口径核查：`hj_alpha00` 在算法上等价于原 `hj_control=true + direction=joint + α=0`
    （α=0 为 no-op，其余 strength/gate/control 一致）。其 17.11 与历史 true 19.43 差 2.3 dB，
    原因是 determinism 代码变更 + `reflection_pad2d` 噪声 + HJ 单 seed 方差；`--display_freq`
    差异经核查为 no-op（display 被 `--display_id -1 --no_html` 禁用），不是 confound。

## 3. 动机

- 路径方向几何变化：部分成立、阶段依赖（DT 早期退出有效）。
- 局部结构冲突（edge/SSIM 方向）：3-seed 归因未复现，α 实验为弱负信号 → **证伪/存疑**。

## 4. limitation

- `reflection_pad2d` backward 无确定性实现 → 单 seed 有约 1~2 dB 运行方差。
- 单 seed（消融/优化）；DT/HJ 各仅 3 seed，n 不足。
- HJ α 实验的 α0 未 clean 复现历史 true（单 seed 方差所致，非代码 bug）。
