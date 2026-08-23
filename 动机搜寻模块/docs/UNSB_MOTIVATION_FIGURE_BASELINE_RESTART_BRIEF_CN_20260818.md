# UNSB 动机图纯基线重启说明：Single-task 与 All-in-One 的观测证据

> 冻结日期：2026-08-18  
> 文档性质：跨环境、跨目录的独立任务说明与历史结论总账  
> 唯一研究对象：**原始 plain UNSB 在 Single-task 与 All-in-One 两种无配对训练制度下的差异**  
> 明确排除：任何后续改性分支、联合方案、性能优化方案及其训练结果  
> 使用方式：把本文件连同原始 UNSB 源码和六域数据交给一个不了解历史的新 Codex，它应能据此重新建立任务、冻结协议、训练基线、提取证据并如实裁决。

---

## 0. 一句话说明我们从哪里出发

我们不是从某个改进方案出发去寻找能够解释收益的图，而是从下面这个尚待检验的科学问题出发：

> 当同一个原始 UNSB 从“每个天气域独立训练”扩展为“六个天气域共享一个模型进行 All-in-One 无配对训练”时，数据混合与参数共享是否会系统性改变其条件恢复路径的方向几何、训练动力学或局部更新结构？

因此，动机图的第一性对照必须只有：

```text
原始 UNSB + Single-task 无配对训练
                    对比
原始 UNSB + All-in-One 无配对训练
```

在这一步中，不应加入任何后续方法，也不应以“某种方法有没有改善”倒推动机。只有先观察到 Single 与 AIO 之间稳定、可复算、经公平控制仍存在的差异，才有资格说 All-in-One 引入了一个值得处理的新现象。

---

## 1. 动机图究竟要证明什么

### 1.1 不是证明 AIO 的终点评分更差

PSNR、SSIM 或视觉质量只能说明两个训练制度最后表现不同，不能告诉我们差异是怎样产生的。动机图需要观察的是**训练过程和条件转移过程本身**：

- 同一个条件状态是否对应更分散的随机恢复方向；
- 这种分散随训练如何出现、消失或反转；
- 是否存在局部更新与源结构保持方向相冲突的区域；
- 上述现象是否在控制本域曝光、随机数、输入样本和桥时间后仍然成立。

### 1.2 真正需要建立的因果顺序

```text
同一份原始 UNSB 实现
        ↓
Single-task 与 All-in-One 两种训练制度
        ↓
严格控制每域数据、曝光量、训练超参数和观测样本
        ↓
提取路径方向与局部更新统计量
        ↓
判断 AIO 是否产生 Single 中不存在或更弱的新现象
        ↓
只有现象成立，才构成后续研究的必要性证据
```

### 1.3 当前最稳妥的中心问题

截至已有实验，应该把中心问题写成：

> 多域共享训练是否改变了 UNSB 的条件方向几何，并且这种改变是否具有训练阶段、桥时间和天气域依赖性？

不要提前把问题写成“AIO 必然更不确定”或“AIO 必然产生更强局部冲突”。已有数据不支持这种全称判断。

---

## 2. Single-task 和 All-in-One 分别是什么

### 2.1 六个天气域

已有本地研究使用六个域：

```text
FoggyCityscapes
LowLightTrafficData
RainCityscapes
RainDS-syn
RSCityscapes
SnowTrafficData
```

正式本地视图曾按每域冻结：

| 用途 | 每域数量 | 六域合计 | 使用方式 |
|---|---:|---:|---|
| trainA | 100 | 600 | 恶劣天气输入域 |
| trainB | 100 | 600 | 清晰图像域，独立无配对采样 |
| discovery | 80 | 480 | 训练后诊断候选池 |
| sealed confirmation | 20 | 120 | 在协议冻结前不得打开 |

`trainA` 与 `trainB` 必须通过 unaligned loader 独立采样。即便文件名能够配对，也禁止训练时按 stem 配对。paired target 只可用于独立的终点评估，不能进入训练或主要动机统计量。

### 2.2 Single-task UNSB 如何得到

Single-task 不是让一个已经接受过六域训练的模型只对某个域推理，也不是在 AIO checkpoint 上继续微调。正确做法是从相同初始化规则出发，分别训练六个相互独立的原始 UNSB：

```text
single_FoggyCityscapes
single_LowLightTrafficData
single_RainCityscapes
single_RainDS-syn
single_RSCityscapes
single_SnowTrafficData
```

每个模型只读取本域的 `100 trainA + 100 trainB`，具有自己独立的参数状态。它们共同构成“任务专用训练制度”的参照。

本地短程协议中，每个 Single 模型的训练量为：

```text
100 images/epoch × 6 epochs = 600 optimizer steps/domain
```

局部诊断的延长协议中为：

```text
100 images/epoch × 50 epochs = 5000 optimizer steps/domain
```

### 2.3 Plain All-in-One UNSB 如何得到

AIO 使用同一份原始 UNSB 实现，但只训练一个共享模型。它的 A/B 训练目录分别是六个域的并集：

```text
trainA_all = union(trainA_domain_1, ..., trainA_domain_6)
trainB_all = union(trainB_domain_1, ..., trainB_domain_6)
```

模型的 G/F/D/E 等组件全部在六域之间共享，数据加载仍是无配对的。每个 AIO epoch 有 600 个更新：

```text
6 epochs  = 3600 total steps
50 epochs = 30000 total steps
```

若六域采样均衡，则每个域的期望曝光分别约为 600 次和 5000 次，与 Single 每域的曝光相同。

### 2.4 公平比较不能只说“相同 epoch”或“相同步数”

Single 和 AIO 存在两个无法同时完全相等的训练计量：

1. **总 optimizer step 相等**：比较相同全局计算步数，但 Single 在这些步里全部看到本域，AIO 只有约六分之一来自该域；
2. **本域期望曝光相等**：比较每个域被训练看到的次数，但 AIO 为此需要约六倍总步数。

两种口径回答不同的问题，不能混为一谈，也不能只报告有利于结论的口径。

推荐主比较采用“本域曝光相等”，并把“总步数相等”作为独立敏感性分析。作图横轴优先使用 `per-domain exposure`，不要只写 epoch。

---

## 3. 原始动机图的 b/c/d/e 分别表示什么

### 3.1 panel b：固定条件状态下的随机恢复方向云

给定同一输入、同一桥状态 `X_t` 和同一桥时间 `t`，用不同随机采样得到多个端点预测 `x_hat_0^(k)`，定义：

```text
d_k      = (x_hat_0^(k) - X_t) / (1 - t)
d_k_norm = d_k / (||d_k||_2 + eps)
```

把 Single 和 AIO 的所有方向放入**同一个 PCA 坐标系**：

- 每个点代表一次随机恢复方向；
- 点云中心代表平均方向；
- 协方差椭圆代表方向族的形状和离散；
- 主轴旋转代表主要变化方向改变；
- 中心偏移代表平均恢复方向改变。

这个 panel 测的是 `stochastic direction geometry` 或 `direction dispersion`，不是经过校准的 posterior uncertainty，也不能直接称为 epistemic uncertainty。

### 3.2 panel c：方向离散随训练推进的轨迹

对一张图像的方向集合计算：

```text
C = Cov({d_k_norm})
U = trace(C) / (||mean(d_k_norm)||_2^2 + eps)
```

也曾使用过多尺度单位方向版本：

```text
z_k  = unit(multiscale_feature(d_k))
U_ms = 1 - ||mean_k z_k||_2^2
```

panel c 是 panel b 的训练时间版本。它应回答：

- 差异从哪个训练阶段开始；
- 是持续存在、逐步收敛，还是发生符号反转；
- 不同桥时间的轨迹是否一致；
- 结论是否依赖 total-step 或 exposure-matched 横轴。

`U`、`U_ms` 和 `median_log_U` 的特征表示、池化及对数口径不完全相同，只能在各自冻结协议内比较，禁止跨报告比较绝对数值。

### 3.3 panel d：局部结构更新冲突的空间图

原始设想还希望检查 baseline 内部的局部对比更新是否反对源图像结构。对局部特征位置 `i`，定义下降更新方向：

```text
u_i = -grad_(f_i) L_local
```

再定义由源图像边缘与结构相似性诱导的结构保持/改善方向 `s_i`：

```text
cos_i = <u_i, s_i> / (||u_i|| ||s_i|| + eps)
h_i   = max(0, -cos_i) × ||u_i|| × gate_i
```

将 `h_i` 恢复到真实 patch 空间位置，得到热图。红色表示局部更新与源结构方向相反。它不是 attention map，也不是普通显著性图。

### 3.4 panel e：局部余弦的总体分布

将多图、多层、多 patch 的 `cos_i` 汇总，至少报告：

```text
negative-tail mass = mean(cos_i < 0)
mean conflict      = mean(max(0, -cos_i) × ||u_i||)
cosine q05
cosine median
```

patch 只是在图像内的重复观测，不能把数万个 patch 当成数万个独立训练复本。正式置信区间应以图像为基本单位，并在域内嵌套或按域等权。

### 3.5 b/c 与 d/e 并不是同一个科学对象

- b/c 观察随机条件转移方向的分布几何；
- d/e 观察某个局部训练目标的梯度与源结构方向的关系。

最初把它们并排放在一张动机图中，是希望说明 AIO 同时存在“路径尺度变化”和“局部结构冲突”。后续数据只支持前者可继续研究，并没有建立后者。因此在新环境中：

1. b/c 应作为主要、预注册的动机证据；
2. d/e 可以作为独立的可证伪诊断，但不得预设 AIO 一定更差；
3. 如果论文必须保持四个 panel 且要求同一逻辑对象，建议让 d/e 改为 `U/U_ms` 的空间分解和图像级差值分布，而不是把两个不同机制强行拼接。

---

## 4. 新环境中应当怎样重新建立目录和身份

新环境的目录很可能与旧机器完全不同。任务说明中禁止依赖旧绝对路径，先建立机器可读路径映射：

```json
{
  "project_root": "<absolute path discovered in the new environment>",
  "source_root": "<plain UNSB source root>",
  "dataset_root": "<six-domain dataset root>",
  "run_root": "<new output root>",
  "state_root": "<machine-readable state root>",
  "report_root": "<figures and markdown root>"
}
```

保存为 `PATH_MAP.json`，之后所有 spec 只引用该映射或相对路径。

### 4.1 源码发现

新 Codex 应先只读搜索并确认：

- 原始 UNSB 的模型入口；
- 共享生成器和桥状态 rollout；
- unaligned dataset loader；
- 训练循环、checkpoint 保存/恢复逻辑；
- 原始损失配置；
- 是否存在历史修改，以及如何确保本轮只启用 plain baseline。

不要因为旧项目中某个目录名像“最新版”就直接使用。必须生成 `CODE_IDENTITY.json`，记录实际执行文件、SHA-256、类/函数入口和运行参数。

### 4.2 数据发现

对六个域分别生成：

- trainA/trainB 文件身份清单；
- stem、扩展名、尺寸与字节数；
- 重复 stem、大小写冲突和跨 split 重叠；
- discovery 与 sealed confirmation 的独立性；
- target 是否被读取及读取目的。

生成 `DATA_MANIFEST.csv/json`。训练身份检查只需要文件元信息，不应在冻结前随意查看 paired target 内容。

### 4.3 保护边界

- 不覆盖旧 checkpoint；
- 不复用已看过六域的模型作为 Single 初始化；
- 不在看完 effect 后换样本、checkpoint、桥时间或统计量；
- 不把 discovery 样本移入训练；
- 不打开 sealed confirmation，除非新的确认协议已冻结；
- 每次训练保存完整 optimizer、scheduler、RNG 和采样位置状态。

---

## 5. 推荐的重启训练协议

### 5.1 共同配置

已有本地严格协议使用：

```text
seed            = 2026
resolution      = 128
batch size      = 1
learning rate   = 1e-4
lambda_GAN      = 1
lambda_SB       = 1
lambda_NCE      = 1
tau             = 0.01
dataset_mode    = unaligned
paired training = false
```

在新环境中可以根据显存改变 batch size，但一旦改变，Single 与 AIO 必须共同改变，并重新冻结协议；不能把新结果与旧 batch-1 数值直接拼接。

### 5.2 第一阶段：低成本语义与短程轨迹

目的不是立即证明论文结论，而是确认两种训练制度、曝光口径和诊断实现无误。

```text
Single: 六个独立模型，各训练 6 epochs
AIO:    一个共享模型，训练 6 epochs
save:   每个 epoch 保存完整状态
```

验收重点：

- 每个 Single 只见本域；
- AIO 每个 epoch 六域计数均衡；
- 每域期望曝光与 Single 对齐；
- 相同输入与共同随机数下可以比较方向样本；
- 恢复训练不改变轨迹语义；
- 无 paired target 进入训练。

### 5.3 第二阶段：路径动机正式诊断

在看任何方法效应之前冻结：

```text
checkpoints          = epoch 1..6，必要时扩展更长轨迹
held-out images      = 每域至少 10 张；正式版尽可能使用完整 discovery
bridge times         = 至少 3 个非平凡时间点
proposals/image/time = 64（低成本 smoke 可先用 16）
representation       = full direction 或唯一冻结的 multiscale feature
aggregation          = 先图像、再域等权
randomness           = Single/AIO 共同随机数
```

必须同时输出：

- per-image raw statistic；
- per-domain summary；
- domain-balanced overall summary；
- image/domain bootstrap interval；
- total-step 与 exposure-matched 两套横轴；
- 所有 checkpoint、输入和随机种子的身份哈希。

### 5.4 第三阶段：局部诊断只作为独立检验

若继续原 d/e，不能只在训练最初几轮观察。已有最终复核使用：

```text
checkpoint      = epoch 50
bridge index    = 2
layers          = 0,4
patches/layer   = 256
images/domain   = 10
domains         = 6
rows/regime     = 6 × 10 × 2 × 256 = 30,720
```

提取阶段不执行 optimizer step，不读取 target。统计裁决以 60 张图或六个域为层级，不以 30,720 个 patch 为独立样本。

如果 epoch 50 仍未支持 AIO 额外冲突，应按失败结论保留，而不是继续换层、换样本或换 checkpoint 直到出现预期图形。

### 5.5 正式论文级复核

旧证据主要来自单 seed。若要形成跨训练复本的正式主张，至少需要：

- 多个独立训练 seed；
- effect-blind 的 held-out manifest；
- 预注册主统计量和主时间窗；
- discovery 与 confirmation 分离；
- 结果盲选代表图；
- 失败和符号反转均进入报告。

多 seed 不是低成本筛查的前置条件，但它是把“本次训练现象”升级为“训练制度性质”的必要条件。

---

## 6. 具体分析与作图流程

### 6.1 effect-blind 冻结

训练和分析前建立 `MOTIVATION_FROZEN_SPEC.json`，至少冻结：

- Single/AIO 的数据清单与初始化规则；
- checkpoint 和 exposure 对齐表；
- 桥时间、proposal 数和特征表示；
- b/c/d/e 的统计公式；
- 代表图的选择规则；
- 颜色、坐标、图例和显示放大规则；
- 主统计单位及 bootstrap 层级；
- 允许和禁止的论文措辞。

### 6.2 panel b 的计算

1. 从 held-out 池按 effect-blind 规则选图，例如每域离预冻结特征中心最近的 medoid；
2. 对同一输入、桥状态和桥时间，给 Single/AIO 使用完全相同的随机数；
3. 各生成 `K=64` 个方向；
4. 将两种制度的方向联合拟合 PCA；
5. 保存原始二维点、中心、协方差、特征值和主轴；
6. 只在显示层绘制椭圆，统计仍使用原始高维方向。

真实方向曾表现为近似秩一，椭圆会像细线。这不是绘图错误。为了可读性可以统一放大中心残差，并设置显示短轴下限，但必须：

- 对 Single/AIO 使用同一规则；
- 在图注中明确“display-only”；
- 原始特征值另存 CSV/NPZ；
- 禁止把放大后的面积当 effect size。

颜色固定建议：Single 使用青绿，AIO 使用红/橙；同一文档不再交换颜色。

### 6.3 panel c 的计算

1. 每张 held-out 图像、每个 checkpoint、每个桥时间计算 `U`；
2. proposal 只用于估计该图的 U，不作为独立样本；
3. 先在图像层聚合，再做域等权；
4. 横轴同时提供 total step 和 per-domain exposure；
5. 曲线显示中位数或预注册中心量，阴影显示图像/域级 bootstrap 95% 区间；
6. 明确标记符号反转，而不是只画最有利的时间窗。

### 6.4 panel d/e 的计算

1. 固定同一桥状态；
2. 在预冻结的层和空间网格上计算局部下降更新；
3. 与源结构方向计算 cosine 与 conflict；
4. 保存真实 `row/col/output_cell`，禁止用 patch 序号冒充空间位置；
5. d 图每张图可以用本图固定分位数归一化显示，但 raw 值必须保留；
6. e 图报告负尾比例、q05、中位数、平均负余弦幅度和门控加权 conflict；
7. 用 paired image/domain bootstrap 比较 Single 与 AIO。

### 6.5 结果盲选与自审

交付前必须回答：

- 图的主体是否仍只有 Single 与 AIO？
- Single 是否真的是独立训练，而非 AIO 的单域推理？
- AIO 是否是六域共享训练，而非六个模型结果的拼接？
- 横轴是否明确写出 total step 或 per-domain exposure？
- b 的 PCA 是否共同拟合？
- c 的 proposal 是否被误当独立样本？
- d 的 patch 是否有真实空间身份？
- 结论是否完整报告阶段反转和负结果？
- 是否把 direction dispersion 错写成校准不确定性？
- 是否为了“更像示意图”而改变了原始统计量？

任意一项答案不清楚，就不应交付最终图。

---

## 7. 已有数据告诉我们的事实

这些数字是新环境重跑时的历史参照，不是要求新结果必须复制的答案。若新实现得出不同结果，应先查身份和协议，再如实报告。

### 7.1 RainCityscapes 等总步筛查：差异具有阶段性

低成本筛查使用：Single 为 Rain 单域 `50 A + 50 B`，AIO 为六域各 `50 A + 50 B`，held-out Rain 20 张，seed 2026。

| global step | Single epoch | AIO epoch | AIO−Single median log10(U) |
|---:|---:|---:|---:|
| 300 | 6 | 1 | +4.732 |
| 600 | 12 | 2 | +3.828 |
| 900 | 18 | 3 | +1.693 |
| 1200 | 24 | 4 | −0.777 |
| 1500 | 30 | 5 | −0.392 |
| 1800 | 36 | 6 | +1.159 |

解释：AIO 早期和最终更高，但中间反转。因此不能写“AIO 全程更分散”。

### 7.2 RainCityscapes 暴露匹配：路径差异得到强化

重新将 Single 的本域曝光 `50/100/150/200/250/300` 与 AIO 的全局步数 `300/600/900/1200/1500/1800` 对齐，并使用：

```text
20 held-out images
K = 64
3 bridge times
full 128×128 direction
```

结果：

- `6 个曝光节点 × 3 个桥时间 = 18` 个单元全部是 AIO 的方向离散高于 Single；
- 以一个冻结桥时间为例，AIO−Single median `log10(U)` 为 `+3.032` 至 `+6.402`；
- 说明早期 equal-step 比较确实混入了本域曝光不等的影响。

这是目前最强的路径动机证据，但仍然只来自 Rain 单域、单 seed，状态应是 `SUPPORTED_SCREEN`，不是六域普遍定律。

### 7.3 同一暴露匹配实验没有支持局部冲突主张

暴露匹配后：

- 18/18 个“平均负余弦幅度”单元都是 AIO 低于 Single；
- 门控加权 conflict 的 18 个单元中，6 个 AIO 更低、3 个更高、9 个无法区分。

冻结主单元的结果为：

| 指标 | Single | AIO | AIO−Single 95% CI |
|---|---:|---:|---|
| 负向 patch 比例 | 32.09% | 35.08% | `+2.99 pp [0.51,5.37]` |
| mean negative cosine | 0.2240 | 0.1584 | `−0.0656 [-0.0778,-0.0540]` |
| cosine q05 | −0.9760 | −0.8020 | `+0.1795 [0.1567,0.2048]` |
| weighted conflict gap | — | — | `−7.41e-6`，CI 跨 0 |

解释：AIO 的负 patch 数量略多，但强度和极端左尾反而更轻，实际门控冲突没有稳定增加。局部指标给出的不是一个统一“AIO 更有害”的故事。

### 7.4 六域短程轨迹再次显示阶段反转

六域每域 100 图、每域期望曝光匹配、epoch 1–6 的严格 `median_log_U`：

| epoch | Single | AIO | AIO−Single 的方向 |
|---:|---:|---:|---|
| 1 | −20.073 | −11.454 | AIO 更高 |
| 2 | −16.671 | −11.490 | AIO 更高 |
| 3 | −13.795 | −13.848 | 近似相等 |
| 4 | −12.774 | −16.407 | AIO 更低 |
| 5 | −12.817 | −18.972 | AIO 更低 |
| 6 | −13.013 | −18.972 | AIO 更低 |

解释：多域共享训练确实改变路径几何，但差异不是单调的，也没有固定符号。一个更可信的论文问题应讨论**阶段依赖的几何重排**，而不是只讨论“离散度升高”。

### 7.5 epoch 50 局部复核仍不支持 AIO 额外冲突

在 60 张图、两层、每层 256 patch 的复核中：

| 制度 | 负向 patch 比例 | mean conflict | cosine q05 | cosine median |
|---|---:|---:|---:|---:|
| Single | 30.42% | 0.000558 | −0.628 | 0.266 |
| AIO | 26.52% | 0.000474 | −0.504 | 0.299 |

AIO 的负向比例、冲突强度和极端左尾都比 Single 更轻。该结果与暴露匹配局部复核一致，因此“只是 epoch 太早所以看不出来”的解释没有得到支持。

### 7.6 终点质量只能作为背景

在一个 Rain 对照中曾观察到：

| 训练制度 | PSNR | SSIM |
|---|---:|---:|
| Single 标准终点 | 19.439 | 0.6672 |
| Single 暴露匹配终点 | 17.882 | 0.5527 |
| Plain AIO | 14.181 | 0.4850 |

它说明 AIO 在该设置下存在明显终点性能缺口，但不能单独证明缺口来自方向离散、局部冲突或任何特定机制。动机图仍必须由训练过程证据承担。

---

## 8. 为什么过去会反复返工

### 8.1 把“必要性”做成了“有效性”

最早的图围绕 plain baseline 与后续方案的差异展开。即使图或评分改善，也只能说明后续方案改变了 baseline，不能回答“为什么 All-in-One 比 Single 多了一个需要研究的问题”。

修正：本任务永久删去所有后续方案，只保留原始 UNSB 的 Single/AIO 对照。

### 8.2 Single 和 AIO 的定义曾被视觉结果遮蔽

中途图像让 Single 看起来很差，容易误以为 Single baseline 本身不成立。实际问题是把中期诊断热图、路径离散和终点恢复质量混在了一起；Single 的终点质量在对应实验中反而最好。

修正：每张图只表达一个科学对象，训练中诊断与终点评估分开。

### 8.3 椭圆像三条线

原因不是样本一定太少，而是高维方向在 PCA 后接近秩一，第二特征值很小。强行画成圆会制造并不存在的二维方差。

修正：原始统计不变，只允许透明、统一的 display-only 短轴下限和中心残差放大。

### 8.4 equal-step 与 exposure-matched 被混淆

相同步数时，Single 看到本域的次数约为 AIO 的六倍，容易把数据曝光差异误认为训练制度差异。

修正：主分析按每域曝光匹配；相同步数作为单独敏感性分析。

### 8.5 以为“样本更大、训练更久”一定会让预期现象更明显

增加 K、桥时间和训练长度后，路径证据更稳定，但局部冲突主张反而被否定。更多成本的作用是提高裁决可信度，而不是保证得到更好看的正结果。

修正：正式协议必须允许 FAIL，不能以“还不明显”为理由无限改指标。

### 8.6 图形语言偏离了原始动机图

曾出现过自定义相图、新坐标、新 panel 或颜色来回变化。即便数学上可能有信息，也会让作者无法判断是否仍在回答原问题。

修正：先严格复现 b 的共同 PCA、c 的训练轨迹、d 的空间热图、e 的分布；任何新图必须标记为补充探索，不能替代冻结主图。

### 8.7 训练阶段选错

局部 cosine 在很早 checkpoint 上观察，会把未成熟训练状态当作长期机制。

修正：短程只作 smoke；局部正式裁决至少扩展到 epoch 50，并报告结果为负。

### 8.8 执行环境和任务范围发生漂移

本来是本地低成本动机验证，却一度被扩展成远端、大规模、长程训练计划，使成本和任务目标脱节。

修正：先在当前机器完成可判定的纯基线筛查；只有冻结的正式复核确实需要更大算力时才迁移，而且迁移不改变科学问题。

### 8.9 交付前没有按作者原问题自审

多次技术结果本身可运行，但主视觉、颜色、比较主体或解释顺序不符合原请求，导致作者承担了发现偏移的工作。

修正：交付前用第 6.5 节清单逐项自审，并在报告首页写出“本图是否证明 Single→AIO 的必要性现象”。

---

## 9. 当前允许和禁止的表述

### 9.1 可以写

较稳妥的中文表述：

> 当同一个 plain UNSB 从单任务无配对训练扩展为六域共享的 All-in-One 训练时，其条件恢复方向几何和训练动力学会发生显著变化。RainCityscapes 的本域曝光匹配实验在多个训练节点和桥时间上观察到更高的 AIO 方向离散，而六域短程轨迹显示该差异具有明显阶段性并可发生反转。因此，All-in-One 不能仅被视为训练数据的简单合并；共享条件转移过程本身需要被直接观测和建模。

更保守的版本：

> Our observations indicate a stage-dependent change in conditional-direction geometry when plain UNSB is extended from single-task to multi-domain unpaired training. The effect is strong under a RainCityscapes exposure-matched protocol but does not retain a universal sign throughout the six-domain short-horizon trajectory.

### 9.2 不能写

- “AIO 在所有域、所有阶段都比 Single 更不确定。”
- “方向离散 U 是经过校准的 posterior uncertainty。”
- “AIO 已被证明产生更强的局部结构冲突。”
- “数万个 patch 等于数万个独立样本。”
- “AIO 的终点 PSNR 更低，所以路径机制已经得到证明。”
- “椭圆显示得更圆，所以真实方向方差更大。”
- “单 seed 的结果已经证明这是多域训练的一般规律。”
- “为了让图明显，可以在看过结果后更换 checkpoint、域或统计量。”

### 9.3 当前证据账本

| 主张 | 当前状态 | 依据 |
|---|---|---|
| Single 与 AIO 的条件方向几何不同 | SUPPORTED | Rain 与六域多版 b/c |
| Rain 暴露匹配下 AIO 方向离散更高 | SUPPORTED_SCREEN | 18/18 网格、20 held-out、K=64、单 seed |
| AIO 在六域训练中始终更分散 | NOT SUPPORTED | e1–2 更高、e3 接近、e4–6 更低 |
| AIO 具有统一的额外局部结构冲突 | FAILED / NOT SUPPORTED | 暴露匹配局部网格与 epoch 50 |
| AIO 终点质量在 Rain 设置中低于 Single | OBSERVED_CONTEXT | PSNR/SSIM 对照；不能代替机制证据 |
| 当前结果可作跨 seed 普遍结论 | NOT YET | 正式多 seed 复核尚缺 |

---

## 10. 新环境应交付什么

建议输出结构：

```text
motivation_baseline_restart/
├─ README_CN.md
├─ PATH_MAP.json
├─ CODE_IDENTITY.json
├─ DATA_MANIFEST.csv
├─ DATA_MANIFEST.json
├─ MOTIVATION_FROZEN_SPEC.json
├─ CHECKPOINT_INDEX.json
├─ TRAINING_STATE.json
├─ CLAIM_LEDGER.json
├─ raw/
│  ├─ direction_samples_or_shard_index.*
│  ├─ image_level_path_statistics.csv
│  ├─ domain_level_path_statistics.csv
│  ├─ local_patch_statistics_or_shard_index.*
│  └─ bootstrap_draws.*
├─ figures/
│  ├─ panel_b_joint_pca.*
│  ├─ panel_c_training_trajectory.*
│  ├─ panel_d_spatial_diagnostic.*
│  └─ panel_e_distribution.*
├─ reports/
│  ├─ ENGINEERING_AUDIT.md
│  ├─ SCIENTIFIC_ADJUDICATION.md
│  └─ MOTIVATION_FIGURE_REPORT_CN.md
└─ MANIFEST.sha256
```

验收必须包含：

- 所有实际执行源码和 checkpoint 的 SHA-256；
- Single/AIO 数据身份与曝光计数；
- 训练过程中 target 未参与的运行时证据；
- raw → summary → figure 的纯函数复算；
- UTF-8、相对链接和全新目录重放；
- 正结果、反转结果与失败结果一并打包；
- 报告中不出现任何后续改性分支。

---

## 11. 可以直接交给新 Codex 的启动描述

```text
你的任务不是设计或验证任何新方法，而是重建一条纯基线的论文动机证据链。

唯一对照是同一份原始 plain UNSB 的两种无配对训练制度：
1. Single-task：六个天气域分别从头训练六个独立模型；
2. All-in-One：六域数据并集训练一个共享模型。

核心问题是：从 Single 扩展到 AIO 后，条件恢复方向几何、训练阶段轨迹和局部更新结构是否发生稳定变化。终点 PSNR/SSIM 只能作为背景，不能代替过程证据。

先只读发现当前环境的源码、数据和输出根目录，生成 PATH_MAP.json、CODE_IDENTITY.json 和 DATA_MANIFEST；禁止沿用旧绝对路径。随后 effect-blind 冻结 Single/AIO 构造、每域曝光对齐、held-out split、checkpoint、桥时间、随机 proposal、b/c/d/e 公式、统计单位、图形颜色和允许措辞。

训练时使用 unaligned A/B，禁止 paired target 进入训练。Single 不能从 AIO checkpoint 恢复；AIO 必须是一个真实共享模型，而不是六个 Single 结果的拼接。主分析按每域期望曝光匹配，同时将总步数匹配作为独立敏感性分析。

panel b 使用 Single/AIO 共同 PCA 的随机恢复方向云；panel c 展示方向离散随训练和每域曝光的轨迹；panel d/e 若继续原局部冲突定义，必须作为独立可证伪诊断，不能预设 AIO 更差。proposal 和 patch 不是独立样本，统计必须先到图像层并按域等权。

已知历史参照是：Rain 暴露匹配下，18/18 个路径网格单元为 AIO 方向离散更高；六域短程轨迹却在后期反转；暴露匹配与 epoch50 的局部诊断均未支持 AIO 额外局部冲突。你必须允许新结果推翻这些历史参照，但不得通过看结果后换指标或 checkpoint 得到想要的图。

最终交付 raw evidence、机器可读 spec/state/claim ledger、b/c/d/e 图、中文裁决报告和 SHA-256 manifest。裁决必须区分 SUPPORTED、SUPPORTED_SCREEN、UNRESOLVED、NOT_SUPPORTED 和 FAILED。
```

---

## 12. 最终状态

这项工作的起点和当前状态可以压缩为一句话：

> 我们要在不引入任何后续方案的前提下，用同一个原始 UNSB 的 Single-task 与 All-in-One 无配对训练作唯一对照，观察多域共享是否改变条件转移的路径几何；现有证据支持“路径几何发生阶段性改变”，但不支持“AIO 始终更分散”或“AIO 具有统一额外局部冲突”。

因此，新环境重启时不需要继承旧目录，也不需要继承任何方法假设；只需要忠实继承这个对照、这些统计边界、已有的正负证据和允许被证伪的研究问题。
