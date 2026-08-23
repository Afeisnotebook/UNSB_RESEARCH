# 自适应介入 schedule（DT 已实现）

## 目标

用可观测诊断量替代手调 `λ=0.001 + ramp5 hold15 decay25`。

## 实现（`dtcov.model.SBModelDTCovMatch`）

新增 `--dtcov_lambda_schedule adaptive`：

1. 每个 epoch 累积 `loss_U_match`（本质是 teacher-student 分布距离的标准化 mismatch），在 epoch 结束时算 epoch 均值并做 EMA（momentum 0.9）。
2. 若相邻 epoch 的 EMA 相对变化 `< ε`（`--dtcov_adaptive_epsilon`，默认 0.02）连续 `patience`（默认 5）个 epoch，判定 mismatch 已进入 plateau，即“参考律已收敛”，λ 退到 0。
3. 否则 λ 走“ramp → 保持 base → plateau 退出”的反馈闭环，替代固定 decay 窗口。

## 可观测诊断量

- teacher-student 分布距离：直接由 `loss_U_match` 提供。
- endpoint 响应漂移：由相邻 epoch 的 EMA 相对变化提供。
- SB 熵项梯度范数：留作后续 HJ/扩展诊断量（当前未接入）。

## 实验对比

- 手调：`dtcov_clean_best_e200`（+0.8875 dB 基准）。
- 自适应：待消融队列空出后跑 `--dtcov_lambda_schedule adaptive --dtcov_ramp_end_epoch 5 --dtcov_adaptive_patience 5`，对比最终 eval-off 收益，目标是 adaptive ≥ 手调。

## 待办

- HJ 的自适应强度已实现：`--hj_schedule adaptive`，`strength` 权重 = conflict EMA / conflict peak（冲突占比高时投影更强，冲突消失时自动退到 0）。
