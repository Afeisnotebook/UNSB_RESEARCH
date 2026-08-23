# 早期搜寻时间线

> 日期为服务器文件/报告里的生成时间（2026 年）。这里只列有决策意义的节点，不列 smoke/脚手架节点。

## 0. 阶段总览

```text
原论文复现
  → joint vs single 严格消融（动机起源）
    → covariance proxy 定义 + test-time 校准
      → UA-12/123/124 训练端初筛
        → test-time 收益被末步阻尼证伪
          → 训练端 log-U consistency（u_match）
            → 六域尺度不匹配
              → DT-CovMatch 路线（交给算法设计模块）
```

## 1. 06-22 ~ 06-26：原论文复现与动机消融

- 项目在 `UNSB复现/` 下建立 strict ablation，核心问题是「一个共享 UNSB 联合训练雨/雪/低光，是否比各自单任务训练更差」。
- 规则明确：训练只用 unpaired 视图；GT 只做测试指标；任务标签不进模型；主 NFE=5 预先固定。
- 结果（`joint - single`）：rain −0.503、snow −0.620、lowlight −3.060 dB PSNR。

结论：**AIO 联合训练确实退化，LowLight 最严重**。这是后续一切工作的原始动机。

## 2. 06-27：状态复盘与 covariance proxy 定义

- 核心文档：`UNSB_Cov_Project/01_research_status_and_evidence_summary.md`。
- 把最初「真实 posterior covariance」的野心收敛为可验证定义：同一 generator、同一 `X_t`、同一 `t`、不同 `z_m` 下的 endpoint proposal 分歧。
- 明确「不是 posterior covariance，只是 predictive proposal uncertainty proxy」。
- 同时定下 harness 纪律（manifest、日志、metrics、diagnostics、return package、baseline mismatch 检查）。

结论：**先别声称 posterior inference，先从 proxy 开始做。**

## 3. 06-27 ~ 06-28：test-time calibration 主线的建立与收窄

- 先做 inference-only：让 `omega` 进入有效区间，路径确实会变。
- mean-only self-ensemble 基本无效；真正起作用的是 covariance shrinkage / omega 路径调制。
- naive train-rollout 在严格 baseline recheck 后被判定负向。
- 最终候选验证里，test-time TTC 最优约 `+0.32 dB PSNR`（单 seed，跨 session GPU RNG 漂移 0.04–0.4 dB）。

结论：**test-time 方向当时是可复现的小增益，但 train-rollout 第一版失败。**

## 4. 07-01 ~ 07-03：训练端 UA 三方案与 test-time 收益的“祛魅”

这一段的服务器目录主要是 `UNSB_C21/`。

- UA-12 = MC 方向方差 + 区域收缩；UA-123 = 在 UA-12 上加低秩区域协方差；UA-124 = side-car `netU` 蒸馏。
- medium demo：ua12/123/124 相对 plain 都没有清晰整体优势（LPIPS 有局部改善，但 PSNR/SSIM 更差或持平）。
- 决策验证：matched q50 下只有 `ua123 test-only` 通过 gate，训练端没有候选通过。
- 标准 TTO：UA123 rank1 test-time 整体 `+0.129 dB`，5/6 域为正。
- **末步消融是最关键的一步**：`constant_mean_last_s070`（普通末步常数收缩）达到 `+0.274 dB`，超过 `ua123_all_index_time` 的 `+0.188 dB`；去掉末步的 UA123 基本归零。

结论：

1. 训练端 `confidence / mpweight+uband / netU` 方向证伪。
2. test-time UA123 的收益很大程度可被普通末步阻尼解释，不再作为独立贡献。

## 5. 07-04 ~ 07-05：Rain/Snow 训练端 scratch + matched joint

- `UNSB_Cov3/`：rain/snow 100-epoch scratch 分支里，固定窗口 UA 候选没有稳定过 gate。
- `UNSB_Cov4/`：matched joint 里，`u_match`（log-U consistency）的 `early20→plain`（前 20 个 epoch 用 u_match，之后 plain）整体 `+0.27 dB`，是唯一同时改善 LowLight 和 Snow 的配置。

结论：**不是「训练端 covariance 一律无效」，而是「固定/全局的训练端 UA 过约束；早期短窗口 + 后续 plain」更合理。**

## 6. 07-07 ~ 07-10：六域 final6 与 DT-CovMatch 路线

- `UNSB_Cov5/` 用六域 final6 数据集（batch16、e200、同 split/warmup）做严格对照。
- 结果排序：`decay10to20` 最强（+0.579 dB / LPIPS −0.015），`early20` 反而 −0.633 dB。
- final `logU` 尺度有诊断价值：plain/decay10to20 在 −12.x，坏分支漂到 −9.x / −8.x。
- 据此提出 `DT-CovMatch`：全局绝对 log-U 匹配改为 **domain-time 归一化坐标下的相对匹配**。

结论：**U/covariance 信号是真的，但六域下绝对尺度跨域不可比；下一步是 domain-time calibration。** 这一路线随后进入算法设计模块。

## 7. 07-15 ~ 07-26：PatchNCE / harmful-joint 早期尝试

- `UNSB_PatchNCE2/` 和 `UNSB_Patch/archive/` 里有一段 HU-PatchNCE 的 harmful-joint 验证。
- 这批工作没有形成干净、稳定的增益，后被算法设计模块的 clean-room + determinism 取代。

结论：**不单独展开，只留下「没经过确定性口径的 patch 收益不可信」这条教训。**

## 8. 边界

本模块结束于算法设计模块的 clean-room 重构起点。之后的 `UNSB_Patch` V5–V9、`UNSB_Long`、`UNSB_PASSION` 已分别进入算法设计模块和动机搜寻模块。
