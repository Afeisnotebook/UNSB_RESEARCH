> **注**：本文是早期“材料抽取”说明；文中的 +1.0439 / +1.4729 是**历史 modified 基线的绝对数字**，仅作参考。当前收益基准是干净框架相对 **+0.8875 dB**，见 `refactor/BASELINE_DECISION.md`；当前目标与阶段见 `PLAN.md`、`DOCS_INDEX.md`。

# UNSB 研究工作区：原始算法 + DT-CovMatch + HJ-PatchNCE 精选材料

这个文件夹是从服务器上多个 `UNSB*` 项目里抽取、整理出来的最小可用工作区，供后续在这个目录上做内容（复盘、对比、改写、写文档、继续实验）使用。

## 目录结构

```text
unsb_tired/
├── 00_original_UNSB/   # 原始 UNSB 算法源码（clean-room 语义基线）
├── 01_DT_CovMatch/     # DT-CovMatch 最优结果相关的代码、配置、报告
└── 02_HJ_PatchNCE/     # HJ-PatchNCE 最优结果相关的代码、配置、报告
```

## 来源与性质

- `00_original_UNSB/` 来自 `UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806/baseline/`，是对上游 `cyclomon/UNSB`（冻结 commit `d1f644f7777e19d5afe5aea3e5cb4bd3afd9b88b`）做 clean-room 抽取后的语义基线，已去掉 `__pycache__` 和 `.pyc`。
- `01_DT_CovMatch/` 来自 `UNSB_Cov5/` 里的 six-domain DT-CovMatch 系列实验，选的是最高收益分支 `dtcov_grouped_ramp5hold15decay25_l001_all6`。
- `02_HJ_PatchNCE/` 来自 `UNSB_Patch/` 里的 continuous layer0-HJ（`true_constant`）路线，对应 V13 最终报告。

## 两个算法的最好结果（单 seed=2026，不可当稳定性结论）

### DT-CovMatch

- 最好分支：`dtcov_grouped_ramp5hold15decay25_l001_all6_plain_e200`
- 相对 plain：**PSNR +1.0439 dB**（19.7800 vs 18.7360），LPIPS -0.0235
- 另有 `dtcov_homog_decay10to20_cos_l001_all6`：**+0.8080 dB**，SSIM 0.6100
- 注意：DT 对 lambda / schedule / estimator / teacher 敏感，同批有负向分支。

### HJ-PatchNCE

- 最好分支：`fin6_patchnce_l0h_true_constant_e200_b16_r128_s2026`（continuous layer0-HJ）
- val-O（5 域×16，offset560）相对 plain：**PSNR +1.4729 dB**（CI95 [+0.89, +2.08]），LPIPS -0.0619
- val-N e200 延迟收益：**+1.33 dB**
- 注意：性能门 PASS，但归因门 FAIL（相对 roll 对照的 SSIM -0.0202），verdict 是 `PERFORMANCE_CONFIRMED_ATTRIBUTION_HOLD`。

## 使用提醒

- 这些都是研究/开发期产物，`DT` 和 `HJ` 都不是最终论文方法，也不构成 Schrödinger Bridge 方法贡献的确认。
- 数据、checkpoint 和完整可运行仓库仍在原项目目录里，这里只放“理解这两个算法 + 最好结果”所需的核心材料。
- 原始 UNSB 代码是后续所有改动的共同基座，DT/HJ 本质是在其 `sb_model.py` / `patchnce.py` 上做的增量改动。
