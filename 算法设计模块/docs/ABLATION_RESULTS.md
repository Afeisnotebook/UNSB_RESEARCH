# Knock-out 消融结果（单 seed=2026，eval-off PSNR）

> **历史/已被后续口径覆盖：** 本文记录确定性修复前的消融。其 DT/HJ 正收益与“不可约 reflection-pad 方差”不是当前结论。当前权威状态见 [FINAL_STATUS.md](./FINAL_STATUS.md)。

> ⚠️ **确定性边界**：以下均为单 seed=2026 结果。已确认生成器 `reflection_pad2d`
> 的 backward 无确定性实现，单 seed 运行间方差约 1 dB；因此 DT 侧 +0.35~+1.07 dB
> 的差距**在噪声量级内，不能单独作为 necessity 结论**。HJ 侧 +2.75 dB 效应较大、
> 更稳健，但同样需多 seed 确认。投稿级结论以用户侧多 seed（均值 + CI）为准。

**3-seed 试点（配对）**：DT-vs-plain delta 为 +0.8875 / +0.7426 / +0.4687，
mean +0.6996，95% CI [0.1712, 1.2280]（不含 0）。即 DT 相对 plain 的方向稳健，
但各 trick 的必要性仍需更多 seed。详见 `DT_MULTISEED_PILOT.md`。

## 基线

- DT：plain 17.9578，best 18.8453（**+0.8875 dB**，test40）。
- HJ：plain 16.6755，true 19.4287（**+2.7533 dB**，val-O）。

## DT 消融

| 改动 | PSNR | delta vs plain | 结论 |
|---|---:|---:|---|
| 基线（grouped_domain） | 18.8453 | +0.8875 | — |
| A1 grouped→equal | 18.3070 | +0.3492 | grouped_domain 必要（贡献 ~0.54 dB） |
| A4 schedule 固定 λ | 18.6955 | +0.7377 | ramp-hold-decay 有贡献（~0.15 dB），固定 schedule 仍保留大部分收益 |
| A2 frozen teacher→self | 19.0276 | +1.0698 | **噪声证据**：`U_match` 全程为 0（应等价 plain），却高于 DT best，判定为运行间方差，不当作“frozen teacher 不必要” |
| A3 domain×time EMA→global | 未跑 | — | 因发现单 seed 非确定而停止，待多 seed 复跑 |
| A5 signal norm→raw U | 未跑 | — | 同上 |

## HJ 消融

| 改动 | PSNR | delta vs plain | 结论 |
|---|---:|---:|---|
| 基线（central_consensus, boundary 0.001, min_risk 0.05, strength 0.5） | 19.4287 | +2.7533 | — |
| A1 central→onesided | 19.0098 | +2.3343 | central_consensus 必要（~0.42 dB） |
| A2 去 boundary | 17.7957 | +1.1202 | boundary 很关键（~1.63 dB） |
| A3 去 min_risk | 18.8557 | +2.1803 | min_risk 必要（~0.57 dB） |
| A4 strength 1.0 | 18.0831 | +1.4077 | strength 0.5 明显优于 1.0（过投影有害，~1.35 dB） |

> 上述 HJ knock-out 为单 seed=2026。HJ 的 3-seed 归因试点（`HJ_MULTISEED_PILOT.md`）
> 显示 true−plain 与 true−roll 的 95% CI 均含 0，roll 对照跨 seed 极不稳定；
> 因此 HJ 的 necessity / 归因结论也需 ≥5 seed 后才能写死，当前只作方向参考。

## 总体结论

单 seed 方向性观察：HJ 的 boundary_scale 与 strength 最敏感，central_consensus / min_risk 也有明显贡献；DT 侧 grouped_domain 与 schedule 有方向性贡献，但差距处于噪声量级。**最终“每项 trick 必要”的结论需多 seed 复现后给出**，此处不作过度声明。
