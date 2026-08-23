# UNSB_RESEARCH

无配对（unpaired）All-in-One 多天气图像恢复的 Schrödinger Bridge（SB）研究仓库，按「模块」组织：

| 模块 | 状态 | 说明 |
|---|---|---|
| [算法设计模块](./算法设计模块/) | ✅ 已上传 | clean-room 重构、确定性修复、DT/HJ 应用层方法、HNEK 桥原生搜索与 e200 确认 |
| 动机模块 | ⏳ 待上传 | Single vs All-in-One 路径几何 / 局部结构冲突的动机证据（后续补充） |

## 快速入口

- 想了解「我们做过什么、得到什么结论」：先读 [算法设计模块/README.md](./算法设计模块/README.md)。
- 想复现或看实现：看 `算法设计模块/code/`（baseline / DT-CovMatch / HJ-PatchNCE / harness）。
- 想查关键数字：看 `算法设计模块/evidence/`。

## 一句话结论

在确定性干净口径下，早期的 DT/HJ 收益基本消失；桥原生 HNEK 坐标修正中，唯一在 e200 后仍存活的是 `hnek_g0.25`（γ=0.25、residual 坐标、真实物理剩余时域），e200 macro PSNR delta +0.7884 dB（4/5 域为正，单 seed、非 confirmatory）。
