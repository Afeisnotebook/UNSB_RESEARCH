# UNSB 算法设计阶段工作总结

日期：2026-08-24
范围：2026-06 至 2026-08-24 的历史方法探索、clean-room 重构、确定性修复与 HNEK 桥原生搜索。

> 本文是事实总结。当前权威数字见 [decisions/CURRENT.md](../../decisions/CURRENT.md) 和实验注册表。

## 1. 做了什么

1. 将历史 DT-CovMatch 与 HJ-PatchNCE 从混杂实现重构为独立模块，建立数据身份、配置冻结、RNG 隔离、checkpoint 审计和配对 bootstrap harness。
2. 修复 baseline 选项与 HJ 集成问题，并将训练/测试管线收敛到同一 clean 口径。
3. 定位并修复 `reflection_pad2d` backward、`torch.bmm`、`adaptive_avg_pool2d` 相关的确定性问题，实现同 seed 模型权重逐位复现。
4. 在确定性 clean 口径下重跑 DT/HJ，发现早期正收益基本消失。
5. 逐行检查 UNSB 的时域与条件注入，定位索引时域 vs 物理剩余时域不一致，以及显式时间注入 time-dead 现象。
6. 围绕剩余时域坐标构造 HNEK，做 9 变体 e50 搜索，并将两个候选延长到 e200。

## 2. 关键阶段结果

### 确定性修复前（历史）

- DT 与 HJ 曾在旧轨迹上呈现正数字。
- 这些实验推动了方法设计，但不能继续作为当前性能主张，因为后续 clean deterministic rerun 改变了结论。

### 确定性 clean core（当前）

| 对比 | Δ PSNR |
|---|---:|
| DT best − plain | −0.2677 dB |
| HJ true − plain | +0.0381 dB |
| HJ roll − plain | −0.7521 dB |
| HJ true − roll | +0.7901 dB |

结论：DT/HJ 作为当时的应用层方法，没有在当前 clean deterministic 口径下展示稳定正收益。

### HNEK e50 搜索

| 变体 | e50 ΔdB | 正域 | 当时裁决 |
|---|---:|---:|---|
| `hnek_g0.5_ref` | −1.2328 | 0/5 | STOP |
| `hnek_g0.25` | +2.6173 | 4/5 | 转 e200 |
| `hnek_g0.75` | −1.3139 | 0/5 | STOP |
| `hnek_g1.0` | +0.7860 | 4/5 | 保留历史屏幕结果 |
| `hnek_coord_y` | +3.1481 | 4/5 | 转 e200 |
| `hnek_horizon_index` | −2.5825 | 1/5 | STOP |
| `hnek_horizon_mix` | +0.7861 | 4/5 | 保留历史屏幕结果 |
| `hnek_entropy_only` | +0.9671 | 5/5 | 保留历史屏幕结果 |
| `hnek_endpoint_only` | +1.3484 | 4/5 | 保留历史屏幕结果 |

### HNEK e200 开发延伸

| 变体 | e200 ΔdB | 正域 | 裁决 |
|---|---:|---:|---|
| `hnek_coord_y` | −1.2164 | 2/5 | DEVELOPMENT_FAIL_SINGLE_SEED |
| `hnek_g0.25` | **+0.7884** | 4/5 | DEVELOPMENT_PASS_SINGLE_SEED |

e50 最强的 `coord_y` 在 e200 翻负，说明短程搜索排名不能直接当作终局结论。`g0.25` 也从 +2.6173 回落到 +0.7884，仍需跨 seed 和未触碰数据确认。

## 3. 现在真正成立的结论

1. 确定性是方法比较的先决条件；修复后旧 DT/HJ 主张不再成立。
2. 桞原生剩余时域坐标值得继续证伪，但 HNEK 不同参数/坐标的结果差异很大，不能用一个正分支泛化整个思想。
3. `hnek_g0.25` 只是开发候选；当前没有已确认的最终算法。
4. 图像级 paired bootstrap CI 不等于训练 seed 级稳定性。

## 4. 最后一轮

1. 在全新环境以同 seed=2026 复现 plain 与冻结的 `hnek_g0.25`。
2. 保持所有超参与数据协议不变，补 2–4 个独立 seed，用 seed 级统计裁决。
3. 使用从未参与搜索的确认集一次性评估。
4. 任一阶段失败则按负结果收口，不再增加变体或回填动机。
