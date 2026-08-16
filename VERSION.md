# v1.0.0 — Clean-room DT-CovMatch / HJ-PatchNCE（阶段性冻结）

## 范围

冻结的是重构后的 clean-room 实现与其证据，不包括原始 legacy 代码、checkpoint、
生成图片、训练日志与数据集：

- clean baseline：`refactor/baseline/`
- DT-CovMatch：`refactor/dt_covmatch/`
- HJ-PatchNCE：`refactor/hj_patchnce/`
- 统一 harness：`refactor/harness/`
- 测试、核心文档、复现/聚合/诊断脚本：`refactor/_runs/*.sh|py|md`、`*.md`、`logs/CHANGELOG.md`
- 小体积结果证据：各 `metrics_summary.json`、`HJ_ATTRIBUTION_RESULT.json`、`dt_clean_comparison.json`、`diagnostics/*.jsonl`

完整文件清单与校验和见同目录 `MANIFEST.sha256`。

## 当前可复现数字（按 mean±CI 报告）

### DT（3-seed 配对，test40）

| seed | plain | DT best | delta |
|---|---:|---:|---:|
| 2026 | 17.9578 | 18.8453 | +0.8875 |
| 2027 | 17.6587 | 18.4013 | +0.7426 |
| 2028 | 19.2021 | 19.6708 | +0.4687 |

- 跨 seed：mean **+0.6996 dB**，95% CI **[0.1712, 1.2280]**。

### HJ（3-seed，val-O）

| seed | plain | true | roll | true−plain | true−roll |
|---|---:|---:|---:|---:|---:|
| 2026 | 16.6755 | 19.4287 | 16.6676 | +2.7533 | +2.7612 |
| 2027 | 15.4467 | 18.5474 | 18.0231 | +3.1007 | +0.5243 |
| 2028 | 18.8758 | 19.2350 | 18.0588 | +0.3592 | +1.1762 |

- true−plain：mean +2.0711，95% CI [−1.6372, +5.7793]。
- true−roll：mean +1.4872，95% CI [−1.3708, +4.3453]。

### 其它（单 seed，仅方向参考）

- DT adaptive（EMA plateau）：PSNR 18.8911（+0.9332）≥ 手调 18.8453（+0.8875）。
- HJ adaptive / per-location risk amplitude 均劣于固定 strength=0.5（单 seed）。

## 已知 limitation

1. **非确定性**：生成器 `reflection_pad2d` 的 backward 在 PyTorch 中无确定性实现；
   即便 `cudnn.deterministic=True` + 关 TF32，同一 seed 两次训练也不逐位一致。
   单 seed 存在约 1~1.5 dB 运行方差。因此所有 sub-dB 差异不作精确声明。
2. **多 seed 未完成**：DT/HJ 各只有 3 seed；HJ 的 true−plain / true−roll 在 n=3 下
   95% CI 含 0，`roll` 对照跨 seed 不稳定。HJ 归因在当前阶段只能作方向参考。
3. **消融/自适应均为单 seed**：DT A1–A5、HJ A1–A4、adaptive 对比均未多 seed。
4. **试点复用共享 warmup**：3-seed 试点同 warmup、同 seed 内配对，是“给定 warmup 的条件
   seed 研究”，不是完整的从头 seed 研究。

## 环境

- Python：`/home/yc/anaconda3/envs/unsb_cov/bin/python`（torch 2.5.1+cu121）
- GPU：单卡 RTX 4090；复现顺序见 `refactor/_runs/REPRODUCE.md`
