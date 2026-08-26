# SEARCH-002 DT/HJ 方向重推导本地报告

## 结论

这轮找到的不是 DT/HJ 旧配置的超参复现，而是一个明确的训练协议变化：**HJ 只负责早期改变优化方向，达到固定数据曝光量后永久关闭，后续完全交还 plain UNSB**。

当前候选是 `finite-horizon HJ → plain handoff`，本地分类为 `positive_but_fragile`，不是已确认算法。

## 为什么回到 HJ

SEARCH-001 的小面板结果曾因三点均值排序而淘汰 HJ，但它的冻结 1200 步点实际为 `+0.804544 dB`、6/6 域正、SSIM `+0.017074`。SEARCH-002 先尝试把 DT/HJ 重写为输出空间 LTTR；三条派生均在 800 步反转：

| 派生 | 400 步 ΔPSNR | 800 步 ΔPSNR | 裁决 |
|---|---:|---:|---|
| LTTR tangent | +0.339362 | -1.097963 | 关闭 |
| LTTR one-epoch pulse | +0.263723 | -1.757175 | 关闭 |
| LTTR direction barrier | +0.109617 | -0.807077 | 关闭 |

共同失败说明：冻结输出/切向参考会把后续轨迹推入坏盆地；历史 HJ 真正有价值的对象仍是 **PatchNCE backward 的局部方向控制**，不是一个新的输出标量正则。

## 独立扩展验证

没有重训或重选 checkpoint，直接加载 SEARCH-001 的 matched plain/HJ 1200 步 checkpoint，在 screen 未使用的 discovery70（每域 70 张，共 420 张）上评估：

| checkpoint | ΔPSNR | 正向域 | 最差域 | ΔSSIM | ΔLPIPS |
|---|---:|---:|---:|---:|---:|
| HJ step1200 | +0.710548 | 6/6 | +0.174754 | +0.020316 | -0.034900 |

六域 PSNR delta：Fog `+0.206683`、LowLight `+0.446709`、RS `+0.174754`、RainCity `+0.193321`、RainDS `+2.143560`、Snow `+1.098258`。该面板通过全部护栏，confirmation20 未打开。

## 有限期 handoff

本地每个数据 epoch 为 150 updates。冻结窗口为：

```text
plain warmup:       [0, 240)      = 1.6 epochs
HJ direction steer: [240, 1200)   = 6.4 epochs
plain handoff:      [1200, ...)
```

从 matched step1200 checkpoint 继续，两条 lane 都只运行 plain 更新：

| step | plain PSNR | HJ-handoff PSNR | ΔPSNR | 正向域 | 最差域 | ΔSSIM | ΔLPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1600 | 13.497782 | 14.158757 | +0.660975 | 5/6 | -1.080550 | +0.040196 | -0.037947 |
| 2000 | 11.686738 | 15.478568 | +3.791830 | 5/6 | -1.748773 | +0.104666 | -0.038704 |

2000 步的大 delta 主要来自 plain 在 Fog/RS/RainCity 的后期坍塌，不能解释成普适 `+3.8 dB`。它支持的更窄结论是：**HJ 导航后的状态没有进入 matched plain 的同一个坏盆地**。LowLight 在 handoff 后转负，因此候选仍是 fragile；推荐以 1200 的全域正结果和 1600 的保留结果为主要证据，2000 只作抗坍塌诊断。

## 4090 冻结协议

窗口按真实数据曝光缩放，不按总训练预算百分比缩放。每域 100 张时一个 epoch 为 600 updates，因此 HJ 固定为 `[960,4800)`，之后到 30k/60k/120k 全部 plain。不得根据中间 paired PSNR 改窗口、方向、layer、strength 或 checkpoint。

风险边界：只有 seed=2026；训练视图仍是本地 25 张/域；discovery 已参与选择；LowLight handoff 后退化；confirmation20 仍封存。下一步必须是 full-view matched 4090，而不是继续本地调参。
