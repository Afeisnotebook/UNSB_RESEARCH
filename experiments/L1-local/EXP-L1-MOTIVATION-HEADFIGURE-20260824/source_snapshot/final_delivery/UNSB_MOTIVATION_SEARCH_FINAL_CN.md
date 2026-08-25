# UNSB 动机搜索最终报告：共享桥域相位失同步

> 研究范围：只比较同一个 plain UNSB 的 task-specific 与 All-in-One 无配对训练；没有接入任何候选算法，也没有用 paired target 或输出质量挑选现象。

## 1. 最终裁决

**`SUPPORTED_DOMAIN_PHASE_MAPPING_REPLICATION`**。

在 Fog / Low-light / Rain / Rain-streak / Snow 五域上，同一个 AIO epoch-1 checkpoint 在 bridge time 0.50、0.74、0.86 分别对应固定的 task-specific 条件核年龄映射：

- `t=0.50: e4 / e3 / e2 / e4 / e5`
- `t=0.74: e4 / e3 / e2 / e4 / e5`
- `t=0.86: e4 / e3 / e2 / e2 / e5`

该映射先出现在 20 图/域发现 split，在额外 24 图/域 split 上 15/15 复现；修复失效 null 后，又在第三个 16 图/域、与前两组及历史测量均零重叠的 split 上 **15/15 复现**。最终 mapping permutation `p=0.0002`，14/15 单元 bootstrap modal share ≥80%，M16/M32 为 15/15。

独立性更正：最初 discovery selector 存在 stem namespace 缺陷（历史账本为 suffix stem、冻结视图为 domain-prefixed stem），因此 100 张 discovery 图中有 14 张与历史测量身份重合。该组只保留为发现证据，不算独立确认；后续 24 图/域和 16 图/域两个 split 使用 canonical stem 排除，历史 overlap 均为 0，并完整复现同一 15-cell map。

论文动机应写成：**一个共享 AIO 训练时钟并没有把所有天气域同步到同一个 task-specific Schrödinger-bridge transition phase。**

## 2. 搜索过程：哪些漂亮结论被否决了

### 2.1 初始方向离散增大：真实，但只能标记 EXPOSURE_ONLY

历史六 seed 的 epoch-1 结果为 30/30 seed×domain 正号、18/18 seed×bridge-time pooled CI 位于零上方；fresh seed=2041 也复现 exposure-matched 差异。但 optimizer-clock 和 Single e1–e5 年龄包络控制不支持更强结论：

| t | exposure gap | 95% CI | clock gap | 95% CI | age-envelope excess | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | +3.562 | [+1.147, +5.798] | +0.122 | [-0.766, +1.215] | -0.545 | [-1.720, +0.772] |
| 0.74 | +3.674 | [+1.150, +6.021] | +0.071 | [-0.809, +1.100] | -0.594 | [-1.785, +0.757] |
| 0.86 | +3.774 | [+1.193, +6.212] | +0.057 | [-0.816, +1.171] | -0.545 | [-1.740, +0.868] |

所以“ AIO 初始方向更散”只保留为进入问题的观测，不能成为必要性结论，也不能被称为校准不确定性。

### 2.2 AIO 是否越出整个 Single 早期轨迹：否

互易共同状态探针裁决为 `NOT_DISTINCT_FROM_SINGLE_AGE_TRAJECTORY`：三处 bridge time 只有 33%、33%、36% 图像高于 `max(MC floor, Single age span)`，0/3 过门。AIO 大体可以由某个 Single 年龄近似，因此“全新条件核”叙事被删除。

### 2.3 第一次 phase confirmation 的置换 null 为什么失效

第一确认集本身给出与发现集完全相同的 15-cell map、M16/M32 15/15，但原协议把整个五年龄 KDD profile 跨域置换，产生 `NOT_REPRODUCED_ON_CONFIRMATION_SPLIT`。该 profile 是相对不同域的不同 Single checkpoint 生成的，跨域不可交换；混合它们测试的是混合参考系统的随机谷底，不是固定域→年龄映射是否复现。因此该 null 永久标为 `INVALID_NULL_EXCHANGEABILITY`，原高 p 不作为科学 FAIL，15/15 也不直接“翻案”。

修复协议预先固定前两 split 共同得到的 map，第三 split 只在每个 bridge time 内打乱五个预定年龄对五个固定域身份的指派，任何 KDD profile 都不离开自己的模型系统。

## 3. 最终 SB-native 观测量

对 AIO e1 和每个 Single age `e∈{1,…,5}`，分别在 AIO rollout 状态 `X_A` 与 Single rollout 状态 `X_S,e` 上计算随机 endpoint direction 的单位平均方向：

`mu_r(X,t)=unit(mean_m(unit((G_r(X,t,z_m)-X)/(1-t))))`

互易条件核方向距离为：

`KDD_e = 1/2[(1-cos(mu_A(X_A),mu_S,e(X_A))) + (1-cos(mu_A(X_S,e),mu_S,e(X_S,e)))]`。

域–时有效年龄定义为：

`e*(d,t)=argmin_e mean_image KDD(d,t,e)`。

它不是把 epoch 标签硬套给 AIO，而是在相同 bridge time、互易模型诱导状态上询问：AIO 当前条件转移方向最接近这个域的哪个 task-specific checkpoint。

## 4. 第三 split 冻结裁决

- 固定 map 命中：**15/15**（门：≥14/15）。
- within-time mapping permutation：**p=0.0002**（门：≤0.001）。
- bootstrap 稳定且等于固定预测：**14/15**（门：≥13/15）。
- M16/M32 有效年龄一致：**15/15**（门：≥14/15）。
- 每个 bridge time 都保留至少三种年龄且 range≥2：**[True, True, True]**。

| t | domain | fixed age | M32 age | M16 age | bootstrap mode | modal share | KDD margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.50 | FoggyCityscapes | 4 | 4 | 4 | 4 | 100.0% | 0.0455 |
| 0.50 | LowLightTrafficData | 3 | 3 | 3 | 3 | 100.0% | 0.0318 |
| 0.50 | RainCityscapes | 2 | 2 | 2 | 2 | 100.0% | 0.1180 |
| 0.50 | RSCityscapes | 4 | 4 | 4 | 4 | 100.0% | 0.0272 |
| 0.50 | SnowTrafficData | 5 | 5 | 5 | 5 | 99.8% | 0.0322 |
| 0.74 | FoggyCityscapes | 4 | 4 | 4 | 4 | 100.0% | 0.0427 |
| 0.74 | LowLightTrafficData | 3 | 3 | 3 | 3 | 100.0% | 0.1318 |
| 0.74 | RainCityscapes | 2 | 2 | 2 | 2 | 100.0% | 0.1847 |
| 0.74 | RSCityscapes | 4 | 4 | 4 | 4 | 100.0% | 0.0395 |
| 0.74 | SnowTrafficData | 5 | 5 | 5 | 5 | 100.0% | 0.0476 |
| 0.86 | FoggyCityscapes | 4 | 4 | 4 | 4 | 100.0% | 0.0151 |
| 0.86 | LowLightTrafficData | 3 | 3 | 3 | 3 | 100.0% | 0.0558 |
| 0.86 | RainCityscapes | 2 | 2 | 2 | 2 | 100.0% | 0.1385 |
| 0.86 | RSCityscapes | 2 | 2 | 2 | 2 | 76.5% | 0.0080 |
| 0.86 | SnowTrafficData | 5 | 5 | 5 | 5 | 100.0% | 0.0364 |

## 5. 它如何支撑论文动机

1. **先有框架差异，再谈改法。** 对象始终是 plain UNSB：五个 task-specific 训练与一个 AIO 共享训练。主证据不是任何新算法相对 baseline 的得分。
2. **落在 Schrödinger Bridge 本身。** 有效 phase 由模型诱导 bridge states 上的条件 endpoint transition direction 定义，不是通用梯度冲突、loss 大小或输出 PSNR。
3. **指出可操作的结构错配。** 单一 global checkpoint 同时对应 Rain 的 e2、Low-light 的 e3、Fog 的 e4、Snow 的 e5；因此对所有域使用同一训练阶段假设并不成立。后续方法的直接靶点应是缩小或协调这种 domain×bridge-time phase spread。
4. **不越过证据。** 本轮没有证明这种失同步必然伤害恢复质量；方法阶段必须分别验证 phase spread 被改善和最终质量得到提升。

## 6. 头图读法

- A：从框架层面提出五个独立 bridge clocks 与一个共享 clock 的差异。
- B：六 seed 初始离散复现是稳定入口现象。
- C：clock/age control 否决把入口现象直接写成必要性的做法。
- D：第三 split 在 `t=0.74` 的五条 KDD–age profile；谷底分别位于 e4/e3/e2/e4/e5。
- E：发现 20 图/域、第一 held-out 24 图/域、第三 held-out 16 图/域的 15-cell map 完全一致；held-out 格同时显示 bootstrap modal share。
- F：修复后的 frozen mapping null；只打乱预定年龄指派，不移动域专用模型产生的 profile。

## 7. 写作边界

允许：在本地五域训练制度、seed=2041 下，一个 AIO e1 checkpoint 对应跨域不同的 task-specific conditional-kernel phases；固定映射在三个零重叠图像 split 上复现。

禁止：多训练 seed 确认、外部/sealed confirmation、RainDS-syn 覆盖、校准 posterior uncertainty、因果恢复伤害，或已经证明某种 phase correction 会提升最终指标。

## 8. 工程身份和成本

- 第三 split：`TERTIARY_INTERNAL_HELDOUT_FROM_SAME_SOURCE_POOL_NOT_SEALED`；80 张；历史/发现/第一确认 overlap 均为 0；manifest `4bfcbf56f10f6cb4cd31d1795b6a2b0201b8f23946a62e257463b90d75e61a01`。
- 模型：fresh seed=2041；五个 Single e1–e5 与一个 AIO e1；本阶段无新训练。
- 第三测量：1200 age rows、240 primary rows、M=32、GPU 31.3 分钟、target_content_read=false。
- 统计：9999 mapping permutations；每 cell 5000 次 image bootstrap；M16 prefix 是冻结门。

## 9. 文件

- PNG：`E:\UNSB_Expl\UNSB_HEADFIGURE_PHASE_TERTIARY_20260824\figures\UNSB_MOTIVATION_HEADFIGURE_FINAL.png`
- PDF：`E:\UNSB_Expl\UNSB_HEADFIGURE_PHASE_TERTIARY_20260824\figures\UNSB_MOTIVATION_HEADFIGURE_FINAL.pdf`
- SVG：`E:\UNSB_Expl\UNSB_HEADFIGURE_PHASE_TERTIARY_20260824\figures\UNSB_MOTIVATION_HEADFIGURE_FINAL.svg`
- 最终裁决：`reports/TERTIARY_PHASE_ADJUDICATION.json`
- 单元摘要：`reports/TERTIARY_PHASE_CELL_SUMMARY.csv`
- 原始 evidence：`raw/RECIPROCAL_KERNEL_BY_AGE.csv`
