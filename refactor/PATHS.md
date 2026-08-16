# 共享路径速查（DT / HJ 两路重构共用）

> 本文件只做“在哪里”，不规定“怎么做”。怎么做以 `CONTRACT.md` 为准。
> 两个 agent 都只读原件，不要改 `01_*`、`02_*`、`UNSB_Cov5`、`UNSB_Patch`。

## 只读的“原始乱码实现”

DT：
- `/home/yc/unsb_tired/01_DT_CovMatch/code/uncertainty_rollout.py`
- `/home/yc/unsb_tired/01_DT_CovMatch/code/sb_model.py`
- best opt：`/home/yc/unsb_tired/01_DT_CovMatch/config/best_branch_train_umatch_active_opt.txt`
- 报告：`/home/yc/unsb_tired/01_DT_CovMatch/reports/`

HJ：
- `/home/yc/unsb_tired/02_HJ_PatchNCE/code/sb_model.py`
- `/home/yc/unsb_tired/02_HJ_PatchNCE/code/patchnce.py`
- `/home/yc/unsb_tired/02_HJ_PatchNCE/code/correspondence_uncertainty.py`
- `/home/yc/unsb_tired/02_HJ_PatchNCE/code/deterministic_ops.py`
- best opt：`/home/yc/unsb_tired/02_HJ_PatchNCE/config/true_constant_train_opt.txt`
- 报告：`/home/yc/unsb_tired/02_HJ_PatchNCE/V13_FINAL_REPORT_CN.md`

## 干净的基线（可读、可复用）

- `/home/yc/unsb_tired/refactor/baseline/`（已经放好的干净 UNSB 基线）
- `/home/yc/unsb_tired/00_original_UNSB/`

## 数据

- 六域原始图：`/home/yc/UNSB_C21/dataset_all/{RainCityscapes,RainDS-syn,SnowTrafficData,FoggyCityscapes,LowLightTrafficData,RSCityscapes}/input|target`
- DT 用的 final6 unpaired：`/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired/`
- HJ 的 val-N / val-O 等：`/home/yc/UNSB_Patch/datasets/`，其中 val-O = `final6train_valO5x16_offset560_unpaired`（以及 `final6train_per_domain_valO5x16_offset560`）
- HJ smoke 数据：`/home/yc/UNSB_Patch/datasets/smoke_patchnce_l0h_train3_test1/`

## 已有 checkpoint（能复用就复用，别从零重训）

DT：
- warmup e20：`/home/yc/UNSB_Cov5/runs/checkpoints_final6srv_b16/fin6srv_b16_unsb_warmup_e20_s2026/`
- plain e200：`/home/yc/UNSB_Cov5/runs/checkpoints_final6srv_b16/fin6srv_b16_samewarm_plain_e200_s2026/`
- best（+1.0439）：`/home/yc/UNSB_Cov5/runs/checkpoints_final6srv_b16/fin6srv_b16_dtcov_grouped_ramp5hold15decay25_l001_all6_plain_e200_s2026/`

HJ：
- 全部 lineage：`/home/yc/UNSB_Patch/runs/checkpoints_patchnce_layer0_handoff_medium/`
  - `fin6_patchnce_l0h_prefix_e4_b16_r128_s2026`
  - `fin6_patchnce_l0h_plain_e200_b16_r128_s2026`
  - `fin6_patchnce_l0h_true_constant_e200_b16_r128_s2026`
  - `fin6_patchnce_l0h_roll_constant_e200_b16_r128_s2026`
  - `fin6_patchnce_l0h_roll_ancestor_e100_b16_r128_s2026`
  - 已生成的 valO 结果目录：`*__valO_true`、`*__valO_roll`、`*__valO_plain`

## 环境

- python：`/home/yc/anaconda3/envs/unsb_cov/bin/python`
- GPU：单卡 4090，24GB。两个 agent 共享，任何训练/评测必须走 `/home/yc/unsb_tired/refactor/gpu_run.sh` 串行，禁止裸跑。
- 输出根：`/home/yc/unsb_tired/refactor/`

## 目标收益（不可凑）

- DT：best branch 相对 plain PSNR `+1.0439`（plain 18.7360，best 19.7800）。
- HJ：valO true vs plain PSNR `+1.4729`；注意归因门 `true vs roll` SSIM `-0.0202` 是 HOLD 状态。
