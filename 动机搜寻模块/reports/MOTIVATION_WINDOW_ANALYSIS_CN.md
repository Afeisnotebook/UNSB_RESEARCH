# UNSB 动机图阶段窗口证据分析

> 状态：分析草稿，供作者人工复核。
> 范围：只使用现有 checkpoints、`raw/*.jsonl`、`reports/MOTIVATION_SUMMARY.json`；未重新训练，也未修改 `/home/yc/unsb_tired`。
> 对象：Single-task（5 个独立 plain UNSB）与 Plain All-in-One（1 个共享 plain UNSB）。DT/HJ 不在本次分析内。

## 1. 动机与主对照

核心问题是：同一个 plain UNSB 从“单任务”扩展为“五域共享 All-in-One”后，条件恢复方向的几何是否发生阶段依赖变化。

主对照只有一组：

```text
Single-task（5 个独立 plain UNSB）
    vs
Plain All-in-One（1 个共享 plain UNSB）
```

所有测量都在同一个 epoch 对齐口径下进行：Single 每域每 epoch 曝光 100 图，AIO 每 epoch 曝光 500 图，因此按 epoch 比较是“每域等曝光”比较，而不是 total optimizer step 比较。

## 2. 为什么不能只用全程弱证据

如果把 Epoch 1–20 直接汇总，`AIO − Single` 的方向差异会出现：

- 有时 AIO 更分散；
- 有时 AIO 更集中；
- 部分域符号不同；
- 均值被正负阶段抵消。

这种全程汇总很容易得到“方向不稳定”或“没有一致差异”的弱结论，掩盖真正的窗口结构。我们关心的不是“全程谁更大”，而是：

> 共享训练是否在某个阶段系统性地把五个域的条件方向压得过紧，从而成为后续算法应该干预的时间窗口。

因此本报告把证据升级为阶段窗口证据。

## 3. 阶段窗口证据设计与统计口径

### 3.1 三个窗口

| 窗口 | Epoch | 角色 |
|---|---|---|
| 初始发散窗口 | 1 | AIO 相对 Single 方向更分散的初始状态 |
| 过度压缩窗口 | 4–5 | 五域共享后被过度压紧的主要动机窗口 |
| 重新分化对照点 | 20 | 训练后期多数域重新反转为 AIO 更分散 |

### 3.2 指标

- `U`：图像级 `trace(Cov(unit directions)) / ||mean(unit directions)||^2`，表示给同一输入和 bridge time 采样不同 latent 时，恢复方向的分散程度。
- `log U`：用于轨迹和跨域 consensus，避免 U 的量级差异过大。
- `u_map`：32px 区域上的 `U_reg = Var_k(d_k) / (mean(d_k)^2 + eps)`，用于空间阶段对比。

### 3.3 共识分数

`window_consensus_score.png` 中每个 epoch 先计算 5 域 × 3 bridge time = 15 个 cell 的 paired `AIO − Single log U` 差值，再汇总三个量：

1. **signed cross-domain consensus**：先取五个域跨 bridge time 的中位符号作为 majority sign，再乘“五域中与多数符号一致的比例”。Epoch 1 应为 +1，Epoch 4–5 应为 −1。
2. **bootstrap CI 与零分离的 cell 数**：15 个 cell 中有多少个 paired bootstrap 95% CI 不跨 0。
3. **pooled effect size**：15 个 cell 均值的平均，表示该 epoch 的整体 AIO−Single log U 效应。

所有 bootstrap 都使用 paired resampling，seed=2026。

## 4. 四张窗口图的具体结果

### 4.1 `window_panel_c_phases.png`

该图复现 AIO 与五个 Single 的 `median log U` 轨迹，并在 Epoch 1、4–5、20 叠加背景。

关键读数：

- Epoch 1：所有域的 `AIO − Single log U` 都为正，方向一致。
- Epoch 4–5：五个域的差异一致转为负，形成非常整齐的负窗口。
- Epoch 20：Foggy、LowLight、RSCityscapes、Snow 的差异重新为正，Rain 仍接近零或负，不再是全同号。

这说明窗口结构是真实的：不是“AIO 全程更分散”，而是“早期发散 → 中期过度压缩 → 后期再次分化”。

### 4.2 `window_panel_e_stages.png`

该图按 Epoch 1、4–5、20 和 t=1,2,3 展示图像级 paired `AIO − Single U` 差分布。

pooled 结果：

| 窗口 | t | pooled mean | 95% CI |
|---|---:|---:|---:|
| Epoch 1 | 1 | +1.24e-05 | [+9.57e-06, +1.53e-05] |
| Epoch 1 | 2 | +2.69e-05 | [+2.23e-05, +3.17e-05] |
| Epoch 1 | 3 | +5.17e-05 | [+4.51e-05, +5.82e-05] |
| Epoch 4–5 | 1 | −2.46e-05 | [−3.82e-05, −1.42e-05] |
| Epoch 4–5 | 2 | −6.36e-05 | [−9.29e-05, −4.00e-05] |
| Epoch 4–5 | 3 | −1.47e-04 | [−2.03e-04, −9.96e-05] |
| Epoch 20 | 1 | +3.69e-04 | [+1.69e-04, +5.91e-04] |
| Epoch 20 | 2 | +5.14e-04 | [+1.90e-04, +8.23e-04] |
| Epoch 20 | 3 | +6.74e-04 | [+2.33e-04, +1.06e-03] |

Epoch 4–5 的 pooled CI 在三个 bridge time 全部严格小于 0，说明这个负窗口不是由少量 outlier 拉出来的。Epoch 20 的 pooled CI 全部大于 0，但分域后 Rain 为负：

| 窗口 | Rain per-domain mean |
|---|---:|
| Epoch 4–5 / t3 | −5.19e-06 |
| Epoch 20 / t1 | −1.36e-04 |
| Epoch 20 / t2 | −6.19e-04 |
| Epoch 20 / t3 | −1.14e-03 |

因此 Epoch 20 只能写“多数域重新反转为 AIO 更分散，Rain 方向不一致”，不能写成全称。

### 4.3 `window_panel_d_stages.png`

该图分 Epoch 4–5 与 Epoch 20，按 AIO 与五个 Single 展示 32px U_reg 热图。

主要观察：

- Epoch 4–5：多数 Single 的区域 U_reg 明显高于 AIO；RSCityscapes single 尤其高，t3 达到约 0.0349，AIO 仅约 0.00031。这说明共享模型在这个阶段确实把区域方向压得更一致。
- Epoch 20：AIO 的 U_reg 上升，Foggy/LowLight/RSCityscapes/Snow single 反而较低；Rain single 则极高，t3 约 0.2627，而 AIO 约 0.0799。

Rain 和 RSCityscapes 的“反转/特殊行为”很清楚：RSCityscapes 在压缩窗口单任务分散度极高，说明它最容易被共享训练压平；Rain 在 Epoch 20 仍保持单任务高分散度，因此它的方向几何没有简单跟随多数域的反转。

### 4.4 `window_consensus_score.png`

consensus 结果：

| Epoch | signed consensus | CI 分离 cell 数 / 15 | pooled log U effect |
|---:|---:|---:|---:|
| 1 | +1.00 | 15 | +5.62 |
| 4 | −1.00 | 15 | −2.59 |
| 5 | −1.00 | 15 | −2.73 |
| 20 | +0.80 | 12 | +4.01 |

Epoch 1 形成 +1 正峰，Epoch 4–5 形成 −1 负峰，Epoch 20 为正但只有 4/5 域同号。这个图把“阶段窗口”表达成了可读分数，而不是只靠轨迹肉眼判断。

## 5. 对 Epoch 4–5“过度压缩窗口”的解释

当前数据下，Epoch 4–5 是最强的动机窗口，理由有三条：

1. **方向一致性最高**：五个域的中位符号全部为负，signed consensus = −1。
2. **统计分离最稳定**：15 个 domain × bridge-time cell 的 paired bootstrap CI 全部与零分离。
3. **图像级和空间级证据一致**：panel_e 的 pooled 差值在 t=1,2,3 全为负；panel_d 显示 AIO 的区域 U_reg 明显低于多数 Single。

我的解释是：共享生成器在 Epoch 1 尚保留来自初始化的多方向分支；训练进入 Epoch 4–5 时，跨域共享约束开始主导，五个域被拉向一个共同的、更紧的方向流形。对多数域来说，这是“过度压缩”的窗口——它不一定是生成质量最差的阶段，但方向几何上最缺乏多样性。

需要克制地写为：

> 在 Epoch 4–5，AIO 相对 Single 出现一致的、跨域和跨 bridge time 的方向过度压缩。

不要写成“AIO 全程更集中”，也不要把 U 解释为校准不确定性。

## 6. Epoch 1 与 Epoch 20 的辅助解释

### 6.1 Epoch 1：初始发散窗口

Epoch 1 的 signed consensus = +1，15 个 cell 全部 CI 分离，说明共享训练刚开始时，AIO 的恢复方向比 Single 更分散。这个窗口可作为“算法如果要在早期保留方向多样性，应在共享化之前/初期介入”的辅助依据。

### 6.2 Epoch 20：重新分化对照点

Epoch 20 的 pooled effect 为正，说明多数域最终又反转为 AIO 更分散；但 Rain 不一致。它告诉我们：共享训练的长期影响不是单方向的，不能只保护某个早期窗口就假设后续稳定。

## 7. 该窗口如何作为后续算法设计切入点

如果后续算法目标是“在方向过度压缩窗口注入方向多样性/路径约束”，当前证据可以支撑一个受限的设计假设：

> 仅在 Epoch 4–5 附近对共享 AIO 的方向流形施加多样化或路径约束，可能比全程约束更精准。

但只能作为下一轮算法实验的候选假设，不能声称它已经被本旁路证明有效。

如果还不能直接进入算法设计，缺少的是：

1. **多 seed 复现**：当前只有 seed=2026，无法判断窗口是稳定的还是单次训练路径。
2. **窗口边界敏感性**：需要比较 Epoch 3–6、4–6、3–5 等相邻窗口，确认“4–5”不是人为切分造成。
3. **干预性证据**：需要在后续实验中真正只在压缩窗口注入约束，再观察下游方向几何是否改善。
4. **更贴近机制的解释变量**：当前 U 是方向几何描述量，不足以证明“压缩”来自共享表示冲突；还需补充梯度/特征/路径层面的证据。

## 8. 单 seed 与统计限制

- 所有结果都是 seed=2026 的单次训练路径。
- 图像级配对 bootstrap 的样本为每域 10 张 discovery 图，样本量小。
- Epoch 窗口按整 epoch 聚合，未对窗口内部迭代顺序做更细的时间分辨率。
- `U` 的方向几何定义本身不是校准不确定性，也不是模型输出质量的直接度量。

因此本报告只使用 `SUPPORTED_SCREEN` 级别措辞：窗口结构方向明确，但仍需人工确认和多 seed 复现。

## 9. 作者睡醒后需要人工确认的清单

1. 确认 `window_consensus_score.png` 的 Epoch 1、4–5、20 峰谷是否符合预期。
2. 检查 `window_panel_c_phases.png` 中 Foggy/LowLight/Rain/RSCityscapes/Snow 的具体轨迹，避免只信 consensus 分数。
3. 检查 `window_panel_e_stages.png` 中 Epoch 4–5 的负 pooled CI 是否稳定，以及 Rain 在 Epoch 20 的反向是否与语义一致。
4. 检查 `window_panel_d_stages.png` 中 Rain 与 RSCityscapes 的热图是否真的形成反转/特殊结构，而非显示尺度伪影。
5. 确认“Epoch 4–5 过度压缩”的措辞没有扩展成“AIO 全程更集中”或“U 是校准不确定性”。
6. 决定下一步是否需要多 seed 复现；若只保留当前单 seed，报告结论只能停留在方向性筛查。

## 附：窗口图文件

- `figures/window/window_panel_c_phases.png`
- `figures/window/window_panel_e_stages.png`
- `figures/window/window_panel_d_stages.png`
- `figures/window/window_consensus_score.png`
