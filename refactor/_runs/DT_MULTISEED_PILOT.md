# DT 多 seed 试点（3 seeds，配对 plain-vs-DT）

本试点回答“DT +0.8875 dB 是否跨 seed 稳健”。每个 seed 共享同一 warmup，
在同一 seed 内跑 plain 与 DT best，做**配对 delta**，避免绝对基线的运行间方差。

| seed | plain PSNR | DT best PSNR | delta |
|---|---:|---:|---:|
| 2026（旧确定性代码，同协议） | 17.9578 | 18.8453 | +0.8875 |
| 2027（修复后确定性代码） | 17.6587 | 18.4013 | +0.7426 |
| 2028（修复后确定性代码） | 19.2021 | 19.6708 | +0.4687 |

跨 seed（n=3）：mean **+0.6996**，std 0.2127，sem 0.1228，
95% CI（Student-t）**[0.1712, 1.2280]**。

## 结论

- plain 绝对 PSNR 跨 seed 波动约 1.5 dB（17.66~19.20），印证了 `reflection_pad2d`
  backward 带来的运行间方差；**不能用绝对 baseline 比较**。
- 但**同 seed 配对 delta 三个 seed 全部为正**，均值 +0.70 dB，95% CI 不含 0，
  说明 DT 相对 plain 的收益方向稳健，且在这 3 个 seed 上可区分于 0。
- 这是“本机 3-seed 初步证据”，不是最终投稿级结论；最终应在更好服务器上扩大到
  更多 seed（建议 ≥5），并同样做 HJ 与各 trick 的多 seed 消融。

复现脚本：`refactor/_runs/run_dt_pilot_multiseed.sh`；聚合：
`python aggregate_multiseed.py 0.8875 0.7426 0.4687`。
