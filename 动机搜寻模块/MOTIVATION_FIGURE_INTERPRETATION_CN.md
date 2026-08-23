# UNSB 动机图可视化结果解释草稿

> 说明：本文档是基于本目录现有 `raw/`、`figures/`、`reports/MOTIVATION_SUMMARY.json` 与 `CLAIM_LEDGER.json` 的分析性解释，不是最终论文结论。实验为单 seed=2026，只允许写方向性、支持性筛查级别的解释。

## 1. 动机

我们要回答的核心问题是：

> 同一个 plain UNSB 从“单任务”扩展为“五域共享 All-in-One”后，条件恢复方向的几何是否发生阶段依赖的变化？

这里的“方向几何”不是生成图像质量，而是给同一输入、同一 bridge time 采样不同 latent 时，恢复方向是否一致。更一致的模型，其条件恢复方向集中；更分散的模型，其方向可能摇摆。

我们设置的唯一主对照是：

- Single-task：5 个独立 plain UNSB；
- Plain All-in-One：1 个共享 plain UNSB。

DT 只作为后置的路径尺度干预 sanity check，不进入主对照；HJ 不进入动机图。

## 2. 可视化方法设计

所有图都围绕路径尺度方向几何量设计：

- `d_k = (y_k - X_t) / (1 - t)`：从 bridge state 指向终点的条件恢复方向；
- `d_k_norm = d_k / ||d_k||`：panel_b 使用的单位方向；
- `U = trace(Cov(d_k_norm)) / ||mean(d_k_norm)||^2`：panel_c / panel_e 使用的图像级分散度；
- `U_reg = Var_k(d_k) / (mean(d_k)^2 + eps)`：panel_d 使用的 32px 区域分散度。

四张图的职责分别是：

| Panel | 设计目的 | 读图方式 |
|---|---|---|
| panel_b | 在 joint PCA 中比较各 arm 最终方向的几何结构 | 点/椭圆是否形成臂间分离，主导轴是否明显 |
| panel_c | 看 log U 随 epoch 的轨迹和 bootstrap CI | 看 AIO 与 Single 的相对方向是否跨阶段反转 |
| panel_d | 看 U_reg 在 32px 空间区域上的分布 | 看哪些区域方向更不稳定，以及 Rain 是否特殊 |
| panel_e | 看图像级 AIO−Single U 差值的配对分布 | 看最终 epoch 的平均差异方向与不确定性 |

重要口径是：Single 每域每 epoch 曝光 100 图，AIO 每 epoch 曝光 500 图，所以主读数按 epoch 对齐，而不是 total optimizer step 对齐。

## 3. 具体结果观察

### 3.1 panel_b：最终方向几何不是各向同性扩散，而是有主导轴和臂间聚类

joint PCA 前两个奇异值约为：

- PC1: 53.46；
- PC2: 23.60。

也就是说第一主成分明显占优。方向点更多沿着一个主轴拉开，而不是一个圆形各向同性云团。

按 arm 聚类后，centroid 有较清晰的差异：

| Arm | 投影 centroid |
|---|---|
| LowLight single | 明显偏 PC1 正侧 |
| Snow single | 偏 PC1 正侧、PC2 正侧 |
| Foggy / Rain / RSCityscapes single | 偏 PC1 负侧 |
| AIO plain | 靠近原点偏负侧 |
| AIO DT | 更偏负侧 |

这提示：把五域共享后，AIO 的方向几何并不是把五个 single 简单平均成一个“中间的圆球”。它更接近某些域的负侧结构，同时第一主轴仍然支配整体差异。

### 3.2 panel_c：AIO 相对 Single 的差异随 epoch 反号，说明是阶段依赖

`CLAIM_LEDGER.json` 中的自动检查显示，几乎所有域、几乎所有 bridge time 都出现 `sign_changes=True`，且多个 epoch 的 bootstrap CI 与 Single 分离。

也就是说，AIO 与 Single 的 log U 差值并不是稳定地“AIO 更大”或“AIO 更小”。例如 Foggy 在 t1/t2/t3 的中位差异为负，而 LowLight 在 t1 为正、t3 转为负；Snow 则大体为正但幅度随 t 下降。这说明共享训练确实改变了方向几何，但变化方向取决于域、阶段和 bridge time。

因此，支持性筛查结论应写为：

> Single 与 AIO 的条件方向几何不同，且差异具有阶段依赖。

而不是“AIO 在所有阶段都更分散”。

### 3.3 panel_d：空间分散度通常随 bridge time 增大，Rain single 最突出

从 summary 看，多数 arm 的 panel_d 标量都随 t=1→2→3 增大。这与指标设计一致：越靠近终点，`1-t` 越小，方向归一化/缩放会放大局部扰动，因此 U_reg 的数值天然更高。

值得注意的对比是：

- `single_RainCityscapes` 的 U_reg 最高，t3 约 0.037；
- `aio_plain` 在 t3 只有约 0.006；
- `aio_dt` 在 t3 反而上升到约 0.032。

一个较合理的解释是：Rain 是五域中恢复方向最不稳定/最复杂的域，single 模型需要在这些区域维持较强的方向分叉；共享 AIO 通过跨域约束，把 Rain 的区域方向波动压得更低；而 DT 激活后，又显著改变了空间方向结构，但没有使其在 U_reg 口径下变得更集中。

### 3.4 panel_e：最终 epoch 的配对差值大体为正，但 Rain 域方向相反

pooled 配对 bootstrap 显示，在最终 epoch 上 AIO−Single 的 U 差值为正：

| bridge time | mean diff | 95% CI |
|---|---:|---:|
| t1 | 0.000369 | [0.000163, 0.000586] |
| t2 | 0.000514 | [0.000188, 0.000824] |
| t3 | 0.000674 | [0.000229, 0.001049] |

这表明“最终 epoch 上，AIO 的图像级方向分散度略高于 Single”。但分域后不是同号的：Foggy、LowLight、RSCityscapes、Snow 大多为正，Rain 在 t1/t2/t3 的 mean diff 为负或 CI 覆盖零。

这个结果与 panel_c 的阶段反转放在一起看，正好说明不能只看 e20 得出结论。AIO 并不是全程更分散；它只是最终阶段整体略高，且 Rain 的行为与多数域相反。

## 4. 我认为更合理的解释

### 4.1 共享训练先压缩、后分化，导致阶段依赖

最符合当前证据的解释是：AIO 在早期会把五域的恢复方向拉进一个共享的、相对更集中的方向流形，从而部分域可能出现 AIO 比 Single 更集中的阶段；随着训练进入后期，各域需要分别解决 Fog/Rain/LowLight/Snow 等不同退化，共享生成器又不得不在同一权重下保留多种方向分支，因此 AIO 的分散度又会在部分域重新上升。

这种“先共享压缩、后域内分化”的动力学会让 AIO−Single 的符号随 epoch 反转，而不是单调正或单调负。

### 4.2 Rain 是方向几何最敏感的域

Rain 在 panel_d 中单任务 U_reg 最高，在 panel_e 中 AIO−Single 差异方向又与其他域相反。这说明 Rain 的方向结构可能最不适合被共享表示直接平均。AIO 一方面降低了 Rain 的区域分散度，另一方面没有像其他域一样在最终 epoch 给出正的 AIO−Single 差值，提示 Rain 的共享收益/代价与 Fog、LowLight、Snow 不同。

这个解释是方向性的，不是结论：只能说 Rain 在路径几何上最特殊，需要单独看，不能写成“Rain 的 AIO 表现更差或更好”。

### 4.3 DT 的 negative sanity check 更可能说明 U 不是 DT 的直接控制量

DT 作为 sanity check，原本期望“路径尺度干预降低 U”。实际结果是 `fraction_dt_lower=0.0`，即 5 个共同 epoch、三个 bridge time 中，DT 相对 AIO plain 的 U 都更高。

较合理的解释是：DT 优化的是自己的 covariance/路径目标，而 `U` 是单位方向上的分散度。DT 的 5-epoch ramp 窗口又很短，warmup 后立即加入约束，可能先扰动方向场，而不是立刻把它收紧。因此不能把 `DT 未降低 U` 直接解读为“DT 机制失效”，更不能作为主对照结论。

## 5. 仍需要人工确认的地方

1. 检查 panel_c 中具体哪些 epoch 发生符号反转，避免只看中位数。
2. 检查 panel_e 的正差异是否主要由 Foggy/RSCityscapes/Snow 驱动，Rain 是否单独呈现相反方向。
3. 检查 panel_d 的 32px 图是否和语义区域有关；当前只汇报了标量，空间热图仍需肉眼核对。
4. 检查 panel_b 的主导轴是否受 medoid 选择和 M=64 方向采样影响。
5. DT 的 U 升高需要人工核对 ramp 窗口、warmup 起点和 bridge-time 对齐后，再决定是否重跑或调整 sanity check。

## 6. 结论边界

可以写：

- Single 与 AIO 的条件方向几何不同，且差异具有阶段依赖；
- Rain/多域下路径几何改变，但不是全程同号；
- DT 作为路径尺度干预没有在当前设置下降低 U，属于 sanity check 的负结果。

不能写：

- AIO 在所有域所有阶段都更分散；
- U 是校准后的 posterior/epistemic uncertainty；
- AIO 已被证明产生更强的局部结构冲突；
- patch 数等于独立样本数；
- HJ 修复了局部冲突。
