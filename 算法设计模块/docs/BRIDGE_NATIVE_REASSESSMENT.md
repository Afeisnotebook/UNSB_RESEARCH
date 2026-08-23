# 桥原生问题重评估（2026-08-17，只读，不讨论 DT/HJ）

> **状态说明：** C003/time-dead 的定位仍是当前方法背景；文中尚未执行/待选路径的部分已被后续 HNEK 搜索和 e200 延伸覆盖。最新结果见 [FINAL_STATUS.md](./FINAL_STATUS.md)。

本文只回答：桥原生问题是什么、怎么独立测量、怎么证伪、最小实验是什么。

## A. 已逐行确认的桥原生缺陷

### A1. 物理时刻 vs 索引时域不一致（C003，CONFIRMED）

位置：`refactor/baseline/models/sb_model.py`。

- `forward()` 构造真实物理 schedule：
  `incs=[0,1,1/2,1/3,1/4] → cumsum → normalize → 0.5+0.5·times → prepend 0`
  得到 `t = [0, 0.5, 0.74, 0.86, 0.94, 1.0]`。
- `compute_G_loss()` 的熵权重：
  `loss_SB = -(T-time_idx)/T · τ · ET_XY`，即索引时域
  `w_index = [1, 0.8, 0.6, 0.4, 0.2]`。
- 真实剩余时域 `h = 1 - t = [1, 0.5, 0.26, 0.14, 0.06]`。

结论：**熵项用的是 `(T-i)/T` 索引权重，不是真实 `1-t`**。
最大偏差 `max|w_index - h_physical| = 0.34`（index 2 处 0.6 vs 0.26；index 4 处 0.2 vs 0.06）。

**独立测量**：纯 numpy 重建上面 schedule 即可复现，不需要 paired target、不需要天气任务
（已实际运行确认，输出见下）。

```
official t            : [0.0, 0.5, 0.74, 0.86, 0.94, 1.0]
physical h = 1-t      : [1.0, 0.5, 0.26, 0.14, 0.06]
index weight (T-i)/T  : [1.0, 0.8, 0.6, 0.4, 0.2]
max |h - index|       : 0.34
```

**证伪方式**：把熵权重改成真实 `h=1-t`，不一致消失（这正是 HNEK 的坐标修正）。
该不一致只在“非均匀物理 rollout schedule”与“均匀索引熵权重”耦合时存在，离开 SB 的
rollout/熵结构就不再存在——因此它是桥原生，不是一般多任务问题。

### A2. 生成器时间注入被覆盖（time-dead，CONFIRMED）

位置：`refactor/baseline/models/ncsn_networks.py` 的 `ResnetBlock_cond.forward`。

```python
time_input = self.Dense_time(time_cond)
for n, layer in enumerate(self.conv_block):
    out = layer(x)          # 注意：每次用 x，而不是 out
    if n == 0:
        out += time_input[:, :, None, None]
```

- 循环每次都重新用 `x` 计算，`conv_block` 的中间结果被覆盖，最终 `out` 只保留最后一个
  `norm_layer(x)`；
- `time_input` 只在 n==0 加到 `pad(x)` 上，随后 n==1 的 `out = layer(x)` 把它覆盖。

结论：**官方生成器的显式时间 conditioning 不进入最终输出，功能上 time-dead**（同时
`conv_block` 第一段 Conv 也失效，只有 `norm_layer(x)` 进入后续）。

**独立测量**：确定性设置下，对同一 `x`、不同 `time_cond` 调用 `netG.forward`，若输出
逐位相同，则 time-dead。

**证伪方式**：若时间注入真正生效，不同 `time_cond` 必须产生不同输出。

## B. HNEK 现状盘点

`/home/yc/UNSB_Long` 里 HNEK 线仍完整：

- `HNEK_FROZEN_SPEC.json`：存在，自洽。
  - backbone=`OFFICIAL_TIME_DEAD`；`residual r = G_plain(x_t,z) - x_t`；
  - `endpoint y = x_t + sqrt(1-t)·r`，exponent 固定 0.5（非超参）；
  - 熵坐标 `(x_t, r)`，熵权重用真实 `h=1-t`；
  - 0 个新可学习参数；paired target 不进训练。
- `code/scripts/hnek/`：存在，自洽。
  - `hnek_kernel.py`：`bridge_schedule` / `physical_time_from_condition` /
    `horizon_from_condition` / `endpoint_from_residual` / `normalized_residual` /
    `transformed_restricted_objective`；
  - `hnek_adapter.py`：只重绑 `netG.forward` / `compute_E_loss` / `compute_G_loss`，
    带 state-dict key / 参数量不变性检查，`hnek_compute_G_loss` 用真实 h 权重；
  - `run_hnek_decisive.py` / `run_hnek_real_model_smoke.py` / `run_hnek_invariants.py` /
    `hnek_adjudicator.py` / `package_hnek_return.py`。
- `code/tests/hnek/`：`test_hnek_adapter / kernel / losses / adjudicator` 4 个测试。
- 已有运行目录 `runs/hnek-unsb-seed2026-20260811/`（含 `full_state_e50.pt.json`），
  说明此前已跑到 e50 这一关。

自洽性结论：spec 里的严格不变量（t=0/t=1 恒等、参数/state-key 不变、PatchNCE 不变、
不启用 Dense_time）与 adapter 实现一致，kernel 数学与 C003/A2 的定位一致。

## C. 距最小可证伪桥原生实验还差什么

1. **代码库对齐**：HNEK scripts 依赖 `scripts.final1`（`UNSB_Long` 树），而当前 clean
   baseline 是 `refactor/baseline`。需要确认 `hnek_kernel` / `hnek_adapter` 能直接作用在
   `refactor/baseline` 的 `SBModel` 上；`bridge_schedule` 与 baseline 的 `times` 构造已核对一致。
2. **最小实验**（已由 spec 冻结，不需要重新设计）：
   - 同 seed、同 RNG bundle 的 coupled **PLAIN vs HNEK**；
   - development 集（不碰 official test，不碰 paired target 训练）；
   - 顺序：effect-free invariants → zero-training retrofit → e50 →（adjudicator 门控）→
     e100/e200。
3. **判据（已冻结，可证伪）**：retrofit ≤ −0.5 dB 即停；e50 若 ≤ −0.15 dB 且 CI 上界 <0
   即明确失败；e200 需 ≥ +0.15 dB、CI 下界 >0 且至少 3/5 域为正。

## D. 结论

- 桥原生问题 = **官方 UNSB 在“非均匀物理 rollout”与“均匀索引熵权重”之间的时域不一致
  （C003）**，以及 **生成器显式时间注入被覆盖（time-dead，A2）**。
- 二者都可只用 numpy/确定性前向独立测量，不依赖天气任务、不依赖 paired target。
- 最小可证伪实验 = 在官方 time-dead backbone 上装 HNEK 坐标修正，跑 frozen PLAIN-vs-HNEK
  单 seed 门控；正向则证明“物理 h 坐标修正”是有效桥原生贡献，负向则证伪该假设。
