# UNSB 动机图机制方向筛选报告

> 状态：机制量筛选草稿，供作者人工复核。
> 范围：只使用现有 seed=2026 checkpoints、raw、测量 manifest；未修改 `/home/yc/unsb_tired`；未启动 DT/HJ，也未设计具体算法。

## 1. 每类机制量的定义与实现

### 机制量 1：跨域梯度冲突 / 相似度

- 理想量：共享 netG 参数在各域训练损失上的梯度 cosine、冲突比例、幅值分布。
- 当前实现：由于本旁路只保存 netG checkpoint，未保存完整优化器/判别器训练图，因此使用**方向场 cosine 代理**。
- 代理定义：对 AIO 每个重点 epoch，采样 `M=32` 个条件方向，按域取平均单位方向，计算五域两两 cosine；`conflict_proxy = 1 - mean_pairwise_cosine`。

### 机制量 2：条件方向场秩 / 有效维度 / 协方差结构

- 输入：每域 1 张 medoid，bridge time `t=1,2,3`，`M=32`。
- 统计：单位方向矩阵中心化后 SVD，计算 effective rank、top-1/top-3 energy、spectral entropy、mean energy。

### 机制量 3：域级特征表示对齐度

- 输入：AIO 每域 1 张 medoid，8 个 latent 样本。
- 实现：hook 三个中间层，计算域间 pairwise linear CKA。
- 当前局限：样本少、层选择有限。

### 机制量 4：压缩窗口内时间变化关系

- 输入：Epoch `1,3,4,5,6,17,20` 的 U/U_reg、方向秩、gradient proxy、feature CKA。
- 只做方向一致性和时间轨迹比较，不做因果推断。

### 机制量 5：多统计量交叉定义“过度压缩”

- 分项：`aio_logU`、`aio_ureg`、方向 effective rank、mean energy、gradient proxy cosine、feature CKA。
- 合成：每个分项在 7 个重点 epoch 上 z-score，并统一符号使正值表示“更压缩”，取可用分项均值作为 `compression_score`。
- 原始值全部保留在 `reports/mechanism/compression_window.json`。

## 2. seed=2026 观察结果

### 2.1 gradient proxy

跨域平均方向 cosine：

| Epoch | t1 | t2 | t3 | 平均 |
|---:|---:|---:|---:|---:|
| 1 | 0.198 | 0.227 | 0.243 | 0.223 |
| 3 | 0.211 | 0.219 | 0.240 | 0.224 |
| 4 | 0.213 | 0.231 | 0.269 | 0.238 |
| 5 | 0.190 | 0.212 | 0.244 | 0.215 |
| 6 | 0.109 | 0.108 | 0.111 | 0.109 |
| 17 | 0.070 | 0.109 | 0.151 | 0.110 |
| 20 | 0.102 | 0.141 | 0.177 | 0.140 |

说明：Epoch 4 略高，但 Epoch 1–5 基本都在 0.21–0.24，没有明显压缩峰；Epoch 6 后 cosine 下降，代表方向冲突代理上升。因此该代理**不支持**“Epoch 4–5 发生跨域方向高度耦合”的强机制。

### 2.2 direction rank

AIO overall effective rank：

| Epoch | AIO effective rank | Single median effective rank |
|---:|---:|---:|
| 1 | 1.002 | 1.207 |
| 3 | 1.010 | 1.002 |
| 4 | 1.048 | 1.002 |
| 5 | 1.227 | 1.003 |
| 6 | 1.049 | 1.004 |
| 17 | 7.024 | 1.011 |
| 20 | 1.058 | 1.017 |

说明：Epoch 4–5 没有出现方向场降秩；Epoch 5 反而有 spectral entropy 上升、top-1 energy 下降。Epoch 17 出现明显高秩尖峰。因此“Epoch 4–5 过度压缩 = 方向场降秩”这个假设在 seed=2026 中**不被支持**。

### 2.3 feature alignment

前两个中间层的域间 CKA 在所有重点 epoch 都接近 1.0，第三个层为 0.0，导致平均值几乎恒定，无法区分窗口。这说明当前 hook 层/单 medoid 样本设计不足以形成有效的域特征对齐信号。

当前只能判定为**描述量/未可靠实现**，不能作为筛选依据。

### 2.4 时间变化关系

U/U_reg 的前序窗口审计显示 Epoch 4–5 的 AIO−Single 为负，即 AIO 相对 Single 更压缩。但 direction rank 显示 AIO 自身方向谱没有同步降秩，gradient proxy 也没有同步出现高 cosine 峰。因此：

- 方向几何量的窗口现象存在；
- 但“共享表示冲突/瓶颈导致压缩”的机制链条没有形成一致时间关系。

### 2.5 compression score

| Epoch | compression_score |
|---:|---:|
| 1 | +0.318 |
| 3 | +0.472 |
| 4 | +0.587 |
| 5 | +0.493 |
| 6 | +0.134 |
| 17 | −0.505 |
| 20 | −1.500 |

Epoch 4 得到最高正分，Epoch 3–5 也为正，Epoch 20 明显为负。这个综合分主要被 `aio_logU` 和 `effective_rank` 驱动；feature CKA 和 mean energy 几乎没有贡献。

因此 compression score 可以作为窗口筛选的粗筛工具，但不能单独证明机制。

## 3. 各机制量是否与 Epoch 4–5 压缩窗口一致

| 机制量 | 是否支持 Epoch 4–5 | 判读 |
|---|---|---|
| U / U_reg（前序窗口审计） | 支持 | 描述性方向几何量，AIOSingle 负 |
| direction effective rank | 不支持 | Epoch 5 反而升秩 |
| gradient proxy cosine | 弱/不支持 | Epoch 1–5 接近，Epoch 6 后下降 |
| feature CKA | 无有效信号 | 当前层/样本设计不敏感 |
| compression score | 部分支持 | Epoch 4 最高，但依赖有限分项 |

结论：没有至少 2–3 类独立机制量同时支持 Epoch 4–5 的机制解释。当前只有几何描述量支持窗口，机制量没有形成一致方向。

## 4. 哪些机制量是强候选靶点

1. **方向场秩 / 谱结构**：虽然当前不支持简单降秩，但它能区分不同 epoch，值得用更多图和 `M=64` 重新做。
2. **跨域梯度冲突**：当前只是方向 cosine 代理；如果能补到真实参数梯度冲突，价值最高。

## 5. 哪些机制量只是描述量

- `U` 与 `U_reg`：只能描述方向分散/空间分散，不能说明“共享表示冲突或瓶颈”。
- feature CKA：当前版本更多是采样/层选择不敏感，暂时只能作为描述量。
- compression score：是合成描述量，不能替代机制证据。

## 6. 哪些机制量互相矛盾

- `U/U_reg` 在 Epoch 4–5 显示 AIO 更压缩；
- direction effective rank 显示 AIO 自身谱在 Epoch 5 不降反升；
- gradient proxy 没有在 Epoch 4–5 出现跨域方向高度一致；
- 因此“早期过度压缩 = 共享瓶颈把方向场压窄”这一机制解释尚未闭合。

## 7. 是否需要补 seed，以及补 seed 后的结果

本次 seed=2026 机制方向混乱，按决策规则应**先不补 seed**。同时，前序任务已经补过 seed=2027、2028 的窗口审计，结果也显示窗口不稳定。因此继续盲目补 seed 不是当前最高优先级。

当前不启动新的 seed 机制测量。

## 8. 最终判断

### 8.1 当前能否筛定“共享表示冲突/瓶颈导致早期方向过度压缩”？

**不能筛定。**

seed=2026 中，方向几何量支持早期压缩窗口，但机制量未形成一致证据，无法把窗口归因到共享表示冲突或瓶颈。

### 8.2 是否具备进入算法设计的条件？

**不具备。**

算法设计需要至少一个可操作的机制靶点；目前只有描述性窗口，没有可靠的机制靶点。

### 8.3 下一步最该补什么？

优先补两项机制量：

1. 真实跨域参数梯度冲突，而不是方向 cosine 代理；
2. 更可靠的方向场谱分析，使用每域多图、`M=64`，并同时检查 direction covariance 与 mean direction。

在补完这两项之前，不建议把“Epoch 4–5 注入方向多样性”作为算法设计前提。

## 9. 图与脚本

图：

- `figures/mechanism/gradient_conflict.png`
- `figures/mechanism/direction_rank_trajectory.png`
- `figures/mechanism/feature_alignment.png`
- `figures/mechanism/compression_window_relation.png`
- `figures/mechanism/compression_score_multimetric.png`

脚本：

- `code/mechanism_common.py`
- `code/mechanism_gradient.py`
- `code/mechanism_direction_rank.py`
- `code/mechanism_feature_alignment.py`
- `code/mechanism_compression_window.py`
- `code/run_mechanism_screen.py`

设计文档：

- `reports/MECHANISM_DESIGN_PLAN_CN.md`
