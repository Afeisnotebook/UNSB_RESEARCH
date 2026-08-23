# 早期搜寻模块（Early Search）

> 本模块只汇总服务器上早期探索里**真正有价值、且没有被后面两个模块覆盖或证伪**的部分。它是整条研究的“考古层”，不是成果全集。

## 一、模块定位

整个项目按时间可以切成三段：

1. **早期搜寻（本模块）**：2026-06-22 到 07-10 左右，重点是「原论文复现 → 定义 predictive covariance proxy → test-time 校准 → train-side UA/u_match → DT-CovMatch 路线」。
2. **算法设计模块**：把早期 DT-CovMatch / HJ-PatchNCE 做 clean-room 重构和确定性修复，最终只剩桥原生 `hnek_g0.25` 存活。
3. **动机搜寻模块**：用干净基线重建「多域共享训练是否改变条件恢复方向几何」的动机证据链。

本模块的定位是：**保存早期探索中仍然成立的动机、关键负结果和方法学教训，不重复搬运已经进入后两个模块的最终代码与最终结论。**

## 二、一句话结论

早期探索的最大价值不是某一组 PSNR，而是四件事：

1. 证明 **All-in-One 联合训练相对单任务会退化**，尤其 LowLight（严格消融 −3.06 dB PSNR），这是整个项目的原始动机。
2. 把「uncertainty」严格收敛为 **MC endpoint proposal disagreement proxy**，并明确**不声称 true posterior covariance**，这个口径一直沿用到最终论文。
3. 排除了多条错误路线：naive train-rollout、confidence 降权、side-car netU、低秩训练端 UA、high-U 门控、专家/prompt routing 都在早期被证明走不通。
4. 找到并留下唯一有希望的训练端方向：**decay10to20 的 log-U consistency → DT-CovMatch（domain-time calibrated covariance consistency）**，它直接成为算法设计模块的前身。

## 三、时间线速览

| 时间（2026） | 阶段 | 关键结论 | 现在怎么看 |
|---|---|---|---|
| 06-22 ~ 06-26 | 原论文复现 + joint vs single 严格消融 | AIO 联合训练退化，LowLight −3.06 dB | 原始动机，后被动机模块用 5 域干净实现重建 |
| 06-27 | 状态复盘 + harness 设计 | 定义 covariance proxy；test-time 校准为早期主线 | 口径保留；harness 纪律沿用至今 |
| 06-27 ~ 06-28 | test-time 校准与最终候选验证 | TTC 稳定小幅正向，mean-only 为零 | 后续发现收益可部分被 terminal damping 解释，降级为背景 |
| 07-01 ~ 07-03 | UA-12/123/124 medium + 快筛 + 决策 + 标准 TTO + 末步消融 | UA123 TTO 收益塌缩为常数末步阻尼；训练端 confidence/netU 失败 | 明确的证伪结果，节省后续大量成本 |
| 07-04 ~ 07-05 | Rain/Snow scratch + matched joint | 固定 u_match 不稳；early_plain（早期窗口后转 plain）最强 | 促成「短窗口 + warmup」设计 |
| 07-07 ~ 07-10 | 六域 final6 + DT-CovMatch roadmap | decay10to20 最强；全局 log-U 跨域不可比，提出 domain-time 校准 | 直接导出算法设计模块 |
| 07-15 ~ 07-26 | PatchNCE harmful-joint / early patch 系列 | 未产出干净稳定增益，被 clean determinism 取代 | 不单独汇总，仅保留方法学教训 |

详细时间线见 [docs/TIMELINE.md](./docs/TIMELINE.md)。

## 四、真正有价值的内容（保留）

### 1. 原始动机证据

`joint - single`（seed=2026，NFE=5，20 张/域）：

| 任务 | single PSNR | joint PSNR | ΔPSNR |
|---|---:|---:|---:|
| rain | 15.576 | 15.073 | −0.503 |
| snow | 23.391 | 22.770 | −0.620 |
| lowlight | 16.190 | 13.130 | −3.060 |

这组数据是「为什么不能简单地一个共享模型吃所有天气」的最早实证。它的价值是**定性动机**，不是最终定量结论（最终动机模块改成了 5 域、去掉 RainDS-syn、干净确定性实现）。

### 2. 术语与方法的边界（最有价值的“护栏”）

- 只允许说：`predictive proposal covariance` / `MC endpoint-induced disagreement` / `pathwise uncertainty proxy`。
- 不允许说：`true posterior covariance` / `calibrated predictive uncertainty`。
- 训练用 unpaired 视图；GT 只用于离线评测；任务子目录不输入模型。

这条边界是早期踩坑后定下来的，后续所有模块都在遵守。

### 3. test-time 校准的关键对照

`mean-only self-ensemble` 基本为零，真正起作用的是一步 `omega_t = omega_min + (1-omega_min)/(1+lambda_t*u_bar_t)` 的 **covariance shrinkage**。当时最优约 `+0.13 ~ +0.32 dB PSNR`，但 SSIM 随强度下降。

### 4. 训练端方向的演化（最有“接力”价值）

早期训练端 `confidence 降权 / mpweight+uband / side-car netU` 全部失败；而 **additive 的 log-U consistency**（`u_match`）在 `decay10to20`（先 warmup、短窗口引入、再退回 plain continuation）上给出六域最强信号。据此提出了 **DT-CovMatch**：把全局绝对 log-U 匹配改为 **domain-time 归一化坐标下的相对匹配**，这是算法设计模块的直接前身。

### 5. 工程与实验纪律

统一 harness、strict matched control、safety image gate、manifest provenance、return package、baseline mismatch 检查。这些不是论文贡献，但避免了「跑的是 A、以为是 B」的反复返工，被后续模块继承。

## 五、已被证伪 / 不再推进（只留结论，不搬代码）

- naive `train-rollout`（从 epoch 0 起 stop-gradient 替换训练路径）在大测试集上稳定负向。
- 固定全局 `early20` 窗口在六域上过硬，不能当最终方法。
- high-U 的 image/pixel 级门控会把故事写散，实验上也不稳。
- 训练端 `confidence 降权`、`mpweight + u_band`、side-car `netU` 蒸馏均未通过 matched gate。
- UA123 的 test-time 收益被 `constant_*_last`（普通末步常数收缩）复现甚至超过，说明它很大程度只是 **terminal damping** 的复杂包装。
- 专家模型 / prompt routing / MoE / per-domain generator 被明确排除。

完整清单见 [docs/SUPERSEDED.md](./docs/SUPERSEDED.md)。

## 六、不纳入汇总的“无意义探索”

这些内容在服务器上占很大体积，但对结论没有贡献，本模块不展开：

- 大量重复的 local smoke / preflight / 打包 harness 脚手架（Cov3/4/5 的 README 基本都是同一套协议模板）。
- 早期过拟合、口径不一的临时结论和未实际评估却被写进报告的配置（如某轮把「λ=5000」写成峰值）。
- `archive/` 里几十 GB 的 legacy runs/checkpoints 与旧 prompt 迭代。
- 单纯为了工程完整性留下的 config/status/manifest 文件。

它们唯一留下的价值是教训：**先定口径、再做实验；没跑过的配置不能写进结论。**

## 七、阅读顺序

1. 本文件 → 快速判断这个模块的边界。
2. [docs/TIMELINE.md](./docs/TIMELINE.md) → 按时间看清每一步为什么做、是否成立。
3. [docs/EVIDENCE.md](./docs/EVIDENCE.md) → 关键证据与来源路径。
4. [docs/SUPERSEDED.md](./docs/SUPERSEDED.md) → 哪些方向已经死掉、为什么死掉。

若要看最终结论：回到仓库根目录，进入 [动机搜寻模块](../动机搜寻模块/) 和 [算法设计模块](../算法设计模块/)。
