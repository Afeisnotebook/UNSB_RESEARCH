# 关键证据汇总

> 这里只保留能支撑结论的硬数字。来源路径是服务器上的原始文件，作为 provenance 保留；仓库里不复制这些服务器文件。

## 1. AIO 联合训练退化（动机起源）

来源：`experiments/unsb_joint_vs_single/metrics/summary.md`（seed=2026，NFE=5，20 张/域）。

| 任务 | single PSNR | joint PSNR | ΔPSNR | single SSIM | joint SSIM | ΔLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| rain | 15.576 | 15.073 | −0.503 | 0.4306 | 0.4177 | −0.0188 |
| snow | 23.391 | 22.770 | −0.620 | 0.6797 | 0.6862 | −0.0219 |
| lowlight | 16.190 | 13.130 | −3.060 | 0.4952 | 0.4241 | +0.0589 |

注：Δ 定义为 `joint - single`。负 PSNR / 正 LPIPS 表示联合训练退化。此表是**定性动机证据**，最终动机模块使用 5 域干净实现重新测量，不要引用这里的绝对数字作为最终值。

## 2. test-time covariance calibration（早期主线）

来源：`UNSB_Cov_Project/01_research_status_and_evidence_summary.md` 与 `runs/20260627_final_candidate_validation/REPORT.md`。

- mean-only self-ensemble：B_mean_M2/M4/M8 的 ΔPSNR 约 −0.001 ~ 0.000，基本为零。
- 有效配置是 covariance shrinkage，而非普通 endpoint 平均。
- 最终候选（单 seed=2026）：

| 配置 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A0 original joint + plain | 13.931 | 0.3219 | 0.6512 |
| A1 original joint + TTC | 14.090 | 0.3154 | 0.6432 |
| B2 base_repeat + TTC (M=4, λ=6000, ω=0.3) | 14.254 | 0.3120 | 0.6362 |
| C2 safe_full + TTC (λ=12000) | 14.237 | 0.3230 | 0.6157 |

- 结论：test-time TTC 有稳定小增益，但 SSIM 随收缩强度下降；且这是单 seed、跨 session GPU RNG 漂移 0.04–0.4 dB 的开发结果。

## 3. UA-12/123/124 的 medium demo（07-01）

来源：`UNSB_C21/UA_4090_medium/README_RESULTS.md`（60 张，20/域）。

| 方案 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| unsb（plain） | 20.945 | 0.6554 | 0.3036 |
| ua12 | 20.887 | 0.6514 | 0.2709 |
| ua123 | 20.563 | 0.6304 | 0.3067 |
| ua124 | 19.933 | 0.6264 | 0.3137 |

结论：没有任何一个 UA 方案在 PSNR/SSIM 上稳定优于 plain；UA-12 只有 LPIPS 有局部改善。这是放弃 `confidence/netU/低秩训练端` 路线的重要依据。

## 4. UA123 TTO 被末步阻尼“祛魅”（07-03）

来源：`UNSB_C21/UA123_terminal_ablation/reports/TERMINAL_ABLATION_SUMMARY.md`。

| 候选 | ΔPSNR |
|---|---:|
| ua123_all_index_time | +0.1884 |
| ua123_only_last_index_time | +0.1354 |
| ua123_without_last_index_time | +0.0004 |
| mc_mean_all_steps | −0.0064 |
| constant_single_last_s080 | +0.1793 |
| constant_mean_last_s070 | **+0.2736** |

结论：去掉末步后 UA123 基本归零；普通末步常数收缩甚至反超 UA123。说明 UA123 test-time 收益主要来自 terminal damping，而非独立的 uncertainty-aware 机制。

## 5. 训练端 u_match 的转折（07-05）

来源：`UNSB_Cov4/reports/SERVER_UMATCH_NEXT_MINIMAL_FINAL_20260705.md`。

| 分支 | overall PSNR | ΔPSNR vs baseline |
|---|---:|---:|
| baseline（plain UNSB） | 18.8625 | 0 |
| u_match_all_fixed | 18.7629 | −0.10 |
| u_match_rs_fixed | 18.0785 | −0.78 |
| **u_match_rs_early_plain** | **19.1367** | **+0.27** |

结论：固定全程 u_match 退化；只有「早期短窗口 u_match 后转 plain」的 early_plain 变体胜出。这直接催生了后续的 `decay10to20` 设计。

## 6. 六域 final6 的尺度问题与 DT-CovMatch（07-08）

来源：`UNSB_Cov5/reports/UNSB_Covariance_Roadmap_CN_20260708.md`。

| 分支 | PSNR | LPIPS | final logU | ΔPSNR |
|---|---:|---:|---:|---:|
| same_warmup_plain | 18.7360 | 0.2868 | −12.5620 | 0 |
| **decay10to20** | **19.3155** | **0.2718** | −12.2018 | **+0.5794** |
| early20 | 18.1033 | 0.3568 | −9.3096 | −0.6328 |
| decay0to30 | 17.9130 | 0.2889 | −9.7618 | −0.8231 |
| split_plain | 17.4485 | 0.4241 | −8.4218 | −1.2875 |

结论：`decay10to20` 是六域 all-in-one 里最合理的 U 信号；`final logU` 有诊断价值（健康分支 −12.x，坏分支 −9.x/−8.x）。据此提出 domain-time 校准的 DT-CovMatch。

## 7. 术语边界（贯穿全程）

- 只能说：`predictive proposal covariance`、`MC endpoint-induced disagreement`、`pathwise uncertainty proxy`。
- 不能说：`true posterior covariance`、`calibrated predictive uncertainty`。
- 训练用 unpaired 视图；任务标签不进模型；GT 只用于离线评测。
