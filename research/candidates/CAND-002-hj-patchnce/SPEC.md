# HJ-PatchNCE 重构规格

## 一句话定位

HJ 不是新标量 loss，而是一个 forward 不变、backward 定向的梯度手术：用源图结构切方向审计 PatchNCE 局部更新，只在方向冲突位置把梯度里与结构方向正对齐的分量投影掉，前向特征和 PatchNCE loss 数值都不变。

## 算法本体（保留）

给定一个 PatchNCE layer 的 query/key 特征 `f_q`、`f_k`，源图 `source`，NCE 目标 `tgt_nce`：

1. 求源图结构切方向 `d = source_structure_direction(tgt_nce, source)`，默认 `joint`（edge 与 SSIM 梯度的单位 RMS 平均）。
2. 构造反事实目标 `tgt_perturb = tgt_nce - step * d_hat`，`d_hat` 为 RMS 归一化并 clamp 到 `[-5,5]` 的方向；若 `central_consensus` 则同时构造 `tgt_opposite = tgt_nce + step * d_hat`。
3. 用无梯度编码得到 `f_q_perturb`（和可选 `f_q_opposite`），计算 PatchNCE 的 delta：
   - 单侧：`delta = (loss_perturb - loss_raw)/step`；
   - central_consensus：`delta = min(one_sided, central)`，`central = (loss_perturb - loss_opposite)/(2*step)`。
4. `risk = clip(delta_plus / per_image_quantile(delta_plus, q), 0, 1)`；`boundary_scale > 0` 时再乘 correspondence boundary instability 并开根号。
5. `gate`：对 risk 取每图 top-quantile 正项，并用绝对证据阈值过滤。
6. 用 `apply_factorial_structure_control` 把 gate 作用到方向（`control=true` 为原样）。
7. 计算 `direction = (f_q_perturb - f_q)/step`（central 时为 `(f_q_perturb - f_q_opposite)/(2*step)`），`project_dose = strength`。
8. `projected_q = project_conflicting_gradient(f_q, direction, strength)`：forward 恒等，backward 去掉 `alignment.clamp_min(0)/norm_sq * direction`。
9. 返回 `projected_loss`，替代原 PatchNCE loss。

不新增参数；`t=0`（关闭干预）时 `projected_loss == raw_loss`。

## 当前训练期协议：有限期方向导航

- HJ forward 恒等、只改变 layer-0 PatchNCE backward 的核心公式不变。
- 先 pure UNSB 运行 1.6 个真实数据 epoch；随后启用 HJ 6.4 个 epoch；第 8.0 epoch 起永久关闭 HJ，后续完全 plain。
- 本地每域 25 张时窗口为 `[240,1200)` optimizer updates；每域 100 张时按数据曝光缩放为 `[960,4800)`，不随 30k/60k/120k 总预算拉长。
- handoff 后训练和推理计算均回到 plain；不根据 paired PSNR 重新开启 HJ。

该协议来自 `EXP-L1-SEARCH-002-DTHJ-20260827`：1200 步 discovery70 为 `+0.710548 dB`、6/6 域正，关闭 HJ 后到 1600 仍为 `+0.660975 dB`。它仍是单 seed development 证据。

## 历史 continuous 协议

- 连续 layer0-HJ 到 200：`nce_uncert_mode=structure_project`，`nce_structure_layers=0`，`start_epoch=5`，`schedule=constant`。该版本已关闭，仅保留为形成史。
- 确定性环境：`--deterministic --deterministic_strict --no_flip`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`PYTHONHASHSEED=0`。
- 评估：`--nce_uncert_mode none --eval --serial_batches --no_flip`，关闭一切训练干预。

## 工程边界

重构后只保留 `structure_project` 这一条路径。原 3260 行 `sb_model.py` 里大量 `bridge_*`、`structure_risk/softpos/posmargin/relational/pixel_project`、`corr_*`、`curation`、`side-car netU` 对最好收益是死代码，移出核心。

## 有意简化点

- 原实现以 `model` 为全局状态，`self._nce_*` 散落；这里把核心抽成无状态的 `structure_project_nce_step` 函数 + `StructureProjectConfig` dataclass。
- 诊断累积/落盘移出核心；上层可另行统计。
- `_structure_event_weight` 在最好配置 `z=0` 时恒为 1，直接省略事件状态。

## 参考实现对照

- `02_HJ_PatchNCE/code/patchnce.py`
- `02_HJ_PatchNCE/code/sb_model.py`
- `02_HJ_PatchNCE/code/correspondence_uncertainty.py`
