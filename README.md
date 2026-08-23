# UNSB_RESEARCH

UNSB 研究的项目仓库，按模块组织。研究无配对（unpaired）All-in-One 多天气图像恢复的 Schrödinger Bridge（SB）框架。

## 模块

- [早期搜寻模块](./早期搜寻模块/)：早期探索的“考古层”，只保留真正有价值、未被后续证伪的结论与方法学教训。
- [动机搜寻模块](./动机搜寻模块/)：重建 UNSB 论文的纯基线动机证据链，研究 Single-task 与 All-in-One 的条件恢复方向几何差异。
- [算法设计模块](./算法设计模块/)：clean-room 重构、确定性修复、DT/HJ 应用层方法、HNEK 桥原生搜索与 e200 确认。

## 快速入口

- 想了解「研究是怎么一步步走到这里的，哪些早期路线被放弃」：读 [早期搜寻模块/README.md](./早期搜寻模块/README.md)。
- 想了解「动机证据与裁决」：读 [动机搜寻模块/README.md](./动机搜寻模块/README.md)。
- 想了解「算法探索与最终结论」：读 [算法设计模块/README.md](./算法设计模块/README.md)。

## 一句话结论

- 早期搜寻：联合训练退化是最初动机；covariance proxy 口径沿用至今；早期 test-time/confidence/netU 路线被证伪，唯一活下来的训练端方向是 domain-time 校准的 DT-CovMatch。
- 动机：多轮实验支持「训练制度改变条件路径几何」，但固定窗口的普遍性结论未立住。
- 算法：确定性干净口径下 DT/HJ 收益基本消失；唯一存活候选是桥原生 `hnek_g0.25`（e200 macro PSNR delta +0.7884 dB，4/5 域为正，单 seed、非 confirmatory）。
