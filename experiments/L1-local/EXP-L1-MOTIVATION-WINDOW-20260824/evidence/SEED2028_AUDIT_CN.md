# seed=2028 方向反转异常审计

> 审计范围：只读检查现有训练日志、checkpoint、raw 测量结果。未修改 `/home/yc/unsb_tired`，未启动新训练。

## 1. 结论摘要

未发现 seed=2028 存在训练日志错误、checkpoint 损坏、测量读取错位或中断证据。重新计算后，seed=2028 的 Epoch 3–6 正向反转仍然成立。

因此当前判断为：**seed=2028 的反转更可能是真实 seed 差异，而不是技术异常。**

## 2. 训练日志检查

检查文件：

- `reports/window_seed/single_FoggyCityscapes_s2028.train.log`
- `reports/window_seed/single_LowLightTrafficData_s2028.train.log`
- `reports/window_seed/single_RainCityscapes_s2028.train.log`
- `reports/window_seed/single_RSCityscapes_s2028.train.log`
- `reports/window_seed/single_SnowTrafficData_s2028.train.log`
- `reports/window_seed/aio_plain_s2028.train.log`

结果：

- 所有 5 个 Single 和 1 个 AIO 都跑满 Epoch 20；
- 每个 epoch 结束都有 `saving the model at the end of epoch`；
- 未发现 NaN、Inf、CUDA OOM、Traceback、RuntimeError；
- loss 数值正常，无爆炸或突然归零。

AI/O 部分训练 loss 均值：

| Epoch | G | NCE | SB |
|---:|---:|---:|---:|
| 1 | 3.740 | 3.407 | 0.025 |
| 4 | 2.929 | 2.710 | 0.024 |
| 5 | 2.551 | 2.023 | 0.029 |
| 6 | 2.380 | 2.165 | −0.014 |
| 17 | 1.714 | 1.458 | −0.172 |
| 20 | 2.163 | 2.098 | 0.042 |

这些数值与 seed=2027 的轨迹属于同一量级，没有异常尖峰。

## 3. Checkpoint 检查

- seed=2028 下存在 6 个训练目录：5 个 Single + 1 个 AIO。
- 每个目录都有 Epoch 1–20 的 `*_net_G.pth` 以及 `latest_net_G.pth`，共 21 个 netG 文件。
- Epoch 1 文件约 58,819,078 字节，Epoch 2–20 文件约 58,819,776 字节，与 seed=2026/2027 的 checkpoint 大小一致。
- Epoch 1、3、4、5、6、17、20 的窗口测量所需 checkpoint 全部存在。

未发现 checkpoint 损坏或缺失。

## 4. 测量结果检查

- `raw/seed2028/` 包含 48 个 JSONL，覆盖 5 个 Single × 8 个 epoch + AIO × 8 个 epoch。
- 文件名格式与 seed=2027 一致。
- AIO 方法名为 `aio_plain_s2028`，Single 方法名为 `single_<domain>_s2028`，无读取错位。
- 每行 `epoch`、`bridge_time_index`、`method`、`domain`、`stem` 字段正确。

## 5. 窗口分数重算

使用与 seed=2026/2027 相同的 paired 逻辑重算 seed=2028 的 `AIO − Single U`：

| 窗口 | 域同号数 | 多数符号 | pooled U diff | 95% CI |
|---|---:|---:|---:|---:|
| 1–1 | 5/5 | + | +2.50e-08 | [+2.15e-08, +2.85e-08] |
| 3–6 | 4/5 | + | +2.16e-05 | [+1.58e-05, +2.76e-05] |
| 4–5 | 4/5 | + | +7.33e-05 | [+6.20e-05, +8.56e-05] |
| 5–6 | 5/5 | + | +5.10e-05 | [+4.36e-05, +5.90e-05] |
| 20–20 | 3/5 | − | −8.60e-05 | [−1.21e-04, −5.55e-05] |

该结果与已有 `reports/seed2028_window_audit.json` 一致：Epoch 3–6 为正向，而不是 seed=2026/2027 的负向压缩。

## 6. 与 seed=2026/2027 的对比

| 窗口 | seed=2026 主符号 | seed=2027 主符号 | seed=2028 主符号 |
|---|---:|---:|---:|
| Epoch 1 | + | + | + |
| Epoch 3–6 | − | − | + |
| Epoch 20 | + | 混合 | − |

seed=2028 与 seed=2026/2027 的差异集中在 Epoch 3–6，且 Epoch 20 也与前两个 seed 不同。这不是单个 domain 或单个 bridge time 的翻转，而是整体窗口方向的反转。

## 7. 技术异常判断

没有发现以下异常：

- 训练日志无错误；
- checkpoint 无损坏；
- 测量命名和解析无错位；
- 重算窗口符号与现有报告一致。

因此进入第二阶段的情况 B：

> 确认 seed=2028 是真实 seed 差异，而不是需要重训练的技术异常。

下一步不重训 seed=2028，也不继续固定窗口的机制量补强。
