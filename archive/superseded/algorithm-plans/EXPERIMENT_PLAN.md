# 机制定位与自适应介入实验计划（相对 +0.8875 dB 为基准）

> **已停止的历史计划：** 其 +0.8875 dB 基准已被确定性 clean rerun 撤销。不要按本文重启 DT/HJ 调参；当前状态见 [CURRENT_STATE_CN.md](../../../CURRENT_STATE_CN.md)。

所有实验单 seed=2026、128×128、统一 harness；只关/改一个变量，记录 eval-off 收益。

## A. DT knock-out 消融（证明每个 trick 必要）

基线分支：`dtcov_clean_best_e200`（+0.8875 dB）。

| 编号 | 改动 | 检验的 trick |
|---|---|---|
| DT-A1 | `grouped_domain` → `equal`（batch 统一） | 是否必须按 domain 分组等权 |
| DT-A2 | frozen teacher → self（当前模型当参考） | 是否必须冻结 first-use 参考 |
| DT-A3 | `domain_time_ema` → 全局单一 EMA | 是否必须 (domain,time) 归一化 |
| DT-A4 | schedule 固定 λ=0.001 全程 | 是否必须短窗口/衰减退出 |
| DT-A5 | 去掉 signal 归一化（U_reg_norm → U_reg） | 是否必须尺度归一化 |

判据：若去掉某项后收益相对 plain 掉到 ~0 或 CI 下界 < 0，则该项是 necessary；若几乎不掉，则是可删的惯性。

## B. HJ 归因与 knock-out

基线分支：`hj_clean_true_e200`（已完成，val-O 结果见 `refactor/_runs/ABLATION_RESULTS.md` 与 `HJ_ATTRIBUTION_RESULT.md`）。

### 归因（重点）

- performance：true vs plain（val-O）。
- attribution：true vs roll（val-O），roll 把结构方向按 patch 平移。
- 附加对照：`hj_direction` 换成纯随机方向（结构方向消融），证明不是任意投影。

若 true 显著 > plain，且 true 显著 > roll（SSIM 等关键指标），则收益是结构方向特异的；否则只能叫通用稳健化。

### knock-out

| 编号 | 改动 |
|---|---|
| HJ-A1 | `central_consensus` → `onesided` |
| HJ-A2 | `boundary_scale` → 0 |
| HJ-A3 | `min_risk` → 0 |
| HJ-A4 | `strength` 0.5 → 1.0 / 0.25 |

## C. 自适应介入 schedule（目标：≥ 手调 schedule）

用可观测诊断量驱动介入强度/时机，替代 λ=0.001 + ramp5hold15decay25。

DT 候选诊断量：

1. teacher-student 分布距离：当前 `log U` 与参考 `log U` 的标准化 mismatch（即 DT loss 本身）；
2. endpoint 响应漂移率：相邻 epoch 的 `U_θ` 相对变化；
3. SB 熵项梯度范数 `‖∇_θ L_SB‖`。

设计：`λ_t = λ0 · g(diag_t)`，当 drift 进入 plateau（连续 k 个 epoch 相对变化 < ε）就把 `g` 退到 0，实现“收敛即退出”，替代固定 decay25。实验比较 adaptive 与手调 ramp5hold15decay25 的最终 eval 收益。

HJ 候选诊断量：directional conflict 占比（gate 命中率）、SB 熵项梯度范数。设计：`α_t` 随冲突占比缩放，冲突消失则 α→0。

## 顺序

1. 等 HJ true/roll 训练 + val-O 评测完成，先出 HJ 归因结论。
2. 跑 DT-A1..A5（GPU 串行）。
3. 跑 HJ-A1..A4。
4. 实现并跑 adaptive schedule，对比手调。
