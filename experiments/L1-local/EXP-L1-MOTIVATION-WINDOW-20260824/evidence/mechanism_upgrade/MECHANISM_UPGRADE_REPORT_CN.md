# UNSB 动机图机制量升级完成说明

> 状态：两项升级机制量已补齐，均为只读分析，未修改 `/home/yc/unsb_tired`，未启动 DT/HJ，未设计算法。
> 范围：seed=2026；AIO plain 与 5 个 Single 的已保存 checkpoint。

## 1. 任务一：真实跨域参数梯度冲突

### 1.1 完成情况

- 脚本：`code/mechanism_gradient_real.py`
- 输出：`reports/mechanism_upgrade/gradient_conflict_real.json`
- 已覆盖重点 epoch：`1, 3, 4, 5, 6, 17, 20`（此前缺 `3`、`17`，本次已补齐）
- 层：5 个 netG 层
  - `model_res.2.conv_fin.2`
  - `model_res.5.conv_fin.2`
  - `model_res.8.conv_fin.2`
  - `model.4`
  - `model_upsample.5`

### 1.2 实现方式

从保存的 `netG/netF/netD/netE` checkpoint 只读重建 `SBModel` 训练图，对每个重点 epoch、每个域取 unpaired A/B 小批量前向，计算 `compute_G_loss()` 后反向，收集 netG 各层权重梯度，再计算五域两两梯度 cosine、conflict fraction 与各域梯度范数。

### 1.3 主要观察

| Epoch | 五层平均 cosine | 平均 conflict fraction |
|---:|---:|---:|
| 1 | 0.559 | 0.000 |
| 3 | 0.234 | 0.020 |
| 4 | 0.114 | 0.300 |
| 5 | 0.262 | 0.100 |
| 6 | 0.266 | 0.160 |
| 17 | 0.131 | 0.320 |
| 20 | 0.112 | 0.220 |

真实参数梯度冲突与方向 cosine 代理的结论方向一致：早期 Epoch 1 跨域梯度高度同向，Epoch 3–4 快速下降，Epoch 5–6 出现部分回升，Epoch 17–20 保持较低 cosine。它仍不能把“Epoch 4–5 过度压缩窗口”单独归因到参数梯度冲突，但补充了一个比方向 cosine 代理更接近训练机制的观测。

## 2. 任务二：方向场谱结构（M=64、多图）

### 2.1 完成情况

- 脚本：`code/mechanism_direction_rank_m64.py`
- 输出：`reports/mechanism_upgrade/direction_rank_m64.json`
- 配置：`M=64`，每域 `3` 张 discovery 图（`c_subset`），bridge time `t=1,2,3`
- 覆盖重点 epoch：`1, 3, 4, 5, 6, 17, 20`

### 2.2 统计量

每个 `(域, epoch, bridge time)` 输出：

- `effective_rank`
- `top1_energy`、`top3_energy`
- `spectral_entropy`
- `mean_energy`（平均方向能量，对应文档要求的“平均方向”）
- `cov_trace`（方向协方差迹，对应文档要求的“协方差”）

### 2.3 主要观察

AIO overall 聚合（跨域、跨 t）：

| Epoch | effective rank | mean energy | cov_trace |
|---:|---:|---:|---:|
| 1 | 1.768 | 0.617 | 0.385 |
| 3 | 1.730 | 0.680 | 0.322 |
| 4 | 1.844 | 0.759 | 0.242 |
| 5 | 1.750 | 0.747 | 0.255 |
| 6 | 1.878 | 0.441 | 0.562 |
| 17 | 1.698 | 0.584 | 0.418 |
| 20 | 1.808 | 0.564 | 0.438 |

M=64 下方向场不再是 M=32 时接近 rank-1 的状态；Epoch 4–5 并未出现 AIO 特有降秩，`cov_trace` 在 Epoch 4–5 处于低点，但 Single 侧同样在 Epoch 4–5 偏低。因此该谱结构是方向分歧的描述量，仍不能单独构成“共享表示瓶颈导致过度压缩”的机制证据。

## 3. 与既有门禁结论的关系

两项升级机制量都没有推翻 `FINAL_GATE_CN.md` 的结论：

- 方向几何差异随训练阶段变化，现象成立；
- 固定 Epoch 4–5 压缩窗口不能作为稳定算法入口；
- 真实梯度冲突与 M=64 方向谱都提示早期跨域耦合较强、中后期分化，但尚未形成一致、可操作的单一机制靶点。

## 4. 运行成本

- 真实跨域参数梯度冲突：单次约 5–20 秒（RTX 4090，2–7 个 epoch）。
- 方向场谱结构 M=64、每域 3 图、7 个重点 epoch：实测墙钟约 74 分钟，主要成本在 AIO（每 epoch 需处理 5 域）。
- 若把方向谱结构升级为每域 10 张 discovery 图，成本约为当前 3.3 倍，预计约 3.5–4 小时。
