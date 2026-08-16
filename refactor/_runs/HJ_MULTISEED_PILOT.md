# HJ 多 seed 归因试点（3 seeds，val-O）

同 warmup、同 seed 内配对跑 plain / true / roll，验证 HJ 的性能与归因是否跨 seed 稳健。

| seed | plain PSNR | true PSNR | roll PSNR | true−plain | true−roll | roll−plain |
|---|---:|---:|---:|---:|---:|---:|
| 2026（旧确定性代码，同协议） | 16.6755 | 19.4287 | 16.6676 | +2.7533 | +2.7612 | −0.008 |
| 2027（修复后代码） | 15.4467 | 18.5474 | 18.0231 | +3.1007 | +0.5243 | +2.576 |
| 2028（修复后代码） | 18.8758 | 19.2350 | 18.0588 | +0.3592 | +1.1762 | −0.817 |

## 跨 seed 汇总（Student-t，n=3）

- true−plain：mean **+2.0711**，std 1.4927，95% CI **[−1.6372, +5.7793]**（含 0）。
- true−roll：mean **+1.4872**，std 1.1504，95% CI **[−1.3708, +4.3453]**（含 0）。
- roll−plain：mean +0.5838，std 1.7724，95% CI [−3.8194, +4.9871]（含 0）。

## 诚实结论

- HJ true 相对 plain 的方向在 3 个 seed 都为正，但幅度从 +0.36 到 +3.10 波动很大，
  且 n=3 的 95% CI 包含 0，**尚不能支持“HJ 稳定增益”的投稿级结论**。
- “结构方向特异”归因也不稳健：true−roll 三个 seed 都为正（方向特异成分存在），但
  roll 对照本身在 seed 2027 相对 plain 大涨 +2.58 dB，说明可能还存在通用投影/稳健化
  成分；且 true−roll 的 CI 含 0。
- 这与原先单 seed=2026 的“roll≈plain、归因四项 PASS”并不矛盾，但表明该归因
  **未跨 seed 复现**，必须扩大到 ≥5 seed 后才能定性。当前 HJ 单 seed 结果只能作
  “方向参考”，不能写死为投稿结论。

复现脚本：`refactor/_runs/run_hj_pilot_multiseed.sh`；聚合：
`python aggregate_multiseed.py <deltas>`。
