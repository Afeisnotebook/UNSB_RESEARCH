# UNSB 动机图窗口结论敲定报告

> 状态：最终决策草稿。
> 范围：只使用现有 checkpoints、`raw/*.jsonl`、`reports/MOTIVATION_SUMMARY.json`；未修改 `/home/yc/unsb_tired`。
> 本次已新增 seed=2027、seed=2028 的 Single-task 与 Plain All-in-One 训练及窗口测量，未训练或测量 DT/HJ。

## 1. 窗口审计量的定义

对每个 epoch 窗口、每个域、每个 bridge time，从 `raw` 中取同一 `(domain, stem)` 的 AIO 与 Single 图像，先在窗口内按 epoch 取均值，再计算 paired 差值：

```text
d_i = U_AIO(i) - U_Single(i)
```

随后统计：

| 量 | 定义 |
|---|---|
| 域符号一致率 | 五域中，域内跨 t 的 mean diff 符号与多数域符号一致的比例 |
| signed consensus | 多数域符号 × 域符号一致率 |
| pooled paired mean | 50 个 `(domain, stem)` 单元的平均差值，域等权 |
| pooled 95% CI | paired bootstrap，seed=2026 |
| 三个 bridge time 的 pooled CI | 每个 t 单独对五域 pooled 差做 bootstrap |
| U_reg 阶段化汇总 | 每个域、每个 t 的 `u_map` 总和，AIO−Single 后按窗口聚合 |

候选窗口为：

`[3,4]`、`[3,5]`、`[3,6]`、`[4,5]`、`[4,6]`、`[5,6]`、`[4,7]`、`[1,1]`、`[20,20]`。

## 2. seed=2026 各候选窗口对比表

符号统一为 `AIO − Single`。负号表示 AIO 方向更压缩/更集中。

| 窗口 | 域同号数 | signed consensus | t1/t2/t3 pooled 符号 | U_reg 同号数 | pooled U diff |
|---|---:|---:|---|---:|---:|
| 1–1 | 5/5 | +1.00 | + / + / + | 5/5 | +3.03e-05 |
| 3–4 | 4/5 | −0.80 | − / − / − | 4/5 | −6.20e-05 |
| 3–5 | 4/5 | −0.80 | − / − / − | 4/5 | −6.47e-05 |
| 3–6 | 4/5 | −0.80 | − / − / − | 4/5 | −5.15e-05 |
| 4–5 | 5/5 | −1.00 | − / − / − | 5/5 | −7.85e-05 |
| 4–6 | 5/5 | −1.00 | − / − / − | 5/5 | −5.64e-05 |
| 5–6 | 5/5 | −1.00 | − / − / − | 5/5 | −4.11e-05 |
| 4–7 | 3/5 | −0.60 | − / − / − | 3/5 | −3.28e-05 |
| 20–20 | 4/5 | +0.80 | + / + / + | 4/5 | +5.19e-04 |

### 窗口边界敏感性结论

在 seed=2026 中，负向“过度压缩”信号不是只发生在 Epoch 4–5，而是主要覆盖 Epoch 3–6。其中：

- Epoch 4–5 是域一致性和 effect size 最强的窗口；
- Epoch 4–6、5–6 也满足 5/5 域同号和三个 bridge time 同侧；
- Epoch 3–4、3–5、3–6 降为 4/5 域同号；
- Epoch 4–7 开始破裂，只有 3/5 域同号。

因此，单看 seed=2026，更合适的措辞是“early-to-mid window Epoch 3–6 存在方向过度压缩，Epoch 4–5 最干净”，而不是把它锁定成一个非常窄的唯一点窗口。

## 3. 多 seed 复现结果

为验证窗口是否稳定，新增 seed=2027 和 seed=2028，训练协议与 seed=2026 一致；测量覆盖 Epoch 1–6、17、20。

| Seed | Epoch 1 | Epoch 3–6 主符号 | Epoch 4–5 域同号 | Epoch 4–5 三个 t pooled | Epoch 20 主符号 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 正 | 负 | 5/5 | 全部负 | 正（Rain 例外） |
| 2027 | 正 | 负 | 4/5 | 全部负 | 不明确 / CI 跨零 |
| 2028 | 正 | 正 | 4/5 正 | 全部正 | 负 / 混合 |

关键点是：**seed=2028 没有复现负向压缩窗口，反而在 Epoch 3–6 给出正向 AIO−Single 方向差。**

这直接推翻了“Epoch 4–5 过度压缩窗口稳定存在”的单 seed 判断。三个 seed 中只有两个支持负向压缩，第三个不仅不支持，还出现相反方向。因此窗口结论不能敲定。

## 4. 当前窗口结论：可敲定还是不可敲定

**不可敲定。**

证据状态：

- seed=2026：支持，Epoch 4–5 为最强候选；
- seed=2027：基本支持负向 early-to-mid window；
- seed=2028：不支持，方向相反。

当前最多只能说：

> 单 seed=2026 中存在一个明显的 early-to-mid 负向窗口；但在三 seed 复现中不稳定，因此不能作为算法设计前的固定阶段窗口依据。

## 5. 如果不能敲定，缺什么证据

1. **更多 seed**：当前 3 个 seed 已经出现 1/3 方向反转，继续盲目补 seed 可能只是增加噪声样本，优先级低于机制量。
2. **窗口内的机制一致性**：当前只有 U/U_reg 两个方向几何描述量，无法区分“压缩”来自共享表示冲突、优化瞬态、域间竞争还是某个域的支配效应。
3. **更细时间分辨率**：目前按整 epoch 窗口，未看 epoch 内部迭代尺度；窗口边界可能是检查点粒度造成的。
4. **跨协议稳定性**：需要知道更换 batch 构型、学习率或域组合后，Epoch 3–6 的负向窗口是否仍然存在。

## 6. 现在是否具备进入算法设计的条件

**不具备。**

理由：

- 核心动机窗口尚未复现稳定；
- 方向差异在 seed 间出现符号反转；
- 当前只有路径几何描述量，没有把窗口解释为“过度压缩”的直接机制证据。

因此不能把“仅在 Epoch 4–5 注入方向多样性”写成有数据支持的设计前提。

## 7. 下一步是否应补机制量，以及补哪些

是，下一步应优先补机制量，而不是继续盲目补 seed。

建议的机制量方向包括：

1. 共享生成器不同域之间的梯度方向冲突或相似度；
2. 条件方向场在训练过程中的秩 / 有效维度 / 协方差结构变化；
3. 域级特征表示的对齐度或任务冲突；
4. 压缩窗口内方向场的时间变化是否与梯度一致性或共享瓶颈相关；
5. 用多个“方向几何统计量”交叉定义“过度压缩”，避免只依赖 U/U_reg。

这些只是下一步应补的机制量清单，不在本任务内实施。

## 8. 剩余不确定性清单

- seed=2028 的方向反转没有机制解释；
- 三 seed 样本过少，不能排除 seed=2028 是离群点；
- Epoch 4–5 和 Epoch 3–6 的边界仍是粗粒度；
- U 和 U_reg 均为几何描述量，不构成因果证据；
- 图像级样本每域仅 10 张 discovery 图，paired bootstrap 的统计精度有限。

## 9. 作者睡醒后需要人工确认的清单

1. 核对 `figures/window_audit/seed2028/seed2028_window_score.png`，确认 seed=2028 在 Epoch 3–6 是否确实为正。
2. 检查三个 seed 的训练日志，确认 seed=2028 没有因 GPU 竞争或 checkpoint 问题导致异常。
3. 检查 seed=2027/2028 的测量是否只覆盖要求的 8 个 epoch，没有误用 seed=2026 raw。
4. 决定是否接受“窗口不可敲定”的结论，还是需要先排除 seed=2028 的异常。
5. 决定下一步是补机制量，还是先做一次更小规模的 seed 敏感性分析。

## 附：窗口审计图

- `figures/window_audit/seed2026_window_score.png`
- `figures/window_audit/seed2026_panel_e_stages.png`
- `figures/window_audit/seed2026_panel_d_stages.png`
- `figures/window_audit/seed2027/seed2027_window_score.png`
- `figures/window_audit/seed2027/seed2027_panel_e_stages.png`
- `figures/window_audit/seed2027/seed2027_panel_d_stages.png`
- `figures/window_audit/seed2028/seed2028_window_score.png`
- `figures/window_audit/seed2028/seed2028_panel_e_stages.png`
- `figures/window_audit/seed2028/seed2028_panel_d_stages.png`
