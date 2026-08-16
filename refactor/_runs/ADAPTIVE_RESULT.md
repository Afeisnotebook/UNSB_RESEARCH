# DT adaptive schedule vs 手调（单 seed=2026，test40）

| 分支 | PSNR | delta vs plain |
|---|---:|---:|
| plain | 17.9578 | 0 |
| 手调 ramp5hold15decay25 | 18.8453 | +0.8875 |
| adaptive（EMA plateau 检测） | 18.8911 | +0.9332 |

结论：adaptive ≥ 手调（+0.0458 dB），满足“adaptive ≥ 手调”目标。

机制观察（诊断日志 refactor/_runs/diagnostics/dtcov_adaptive.jsonl）：
- drift（teacher-student mismatch EMA）从 warmup 的 0 增长到窗口末 ~0.52，仍在收敛，因此 25 窗口内 plateau 未触发，λ 保持在 base，随后 plain 续训。
- SB 熵项梯度范数在整个窗口内波动（0.02~0.41），为后续“介入时机定位”提供机制证据。
