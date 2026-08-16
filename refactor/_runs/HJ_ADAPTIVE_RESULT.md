# HJ adaptive vs 手调（单 seed=2026，val-O）

| 分支 | PSNR | delta vs plain |
|---|---:|---:|
| plain | 16.6755 | 0 |
| 手调 strength 0.5（true） | 19.4287 | +2.7533 |
| adaptive（strength ∝ conflict EMA/peak） | 18.7945 | +2.1190 |

结论：HJ adaptive **低于** 手调（-0.6343 dB）。conflict 占比驱动的全局 strength 缩放会稀释 per-location 投影，反而不如固定 strength 0.5。

## 补充：per-location risk amplitude 也无效

| 分支 | PSNR | delta vs plain |
|---|---:|---:|
| 手调 constant amplitude（strength 0.5） | 19.4287 | +2.7533 |
| risk amplitude（每位置按 risk 缩放 dose） | 17.6911 | +1.0156 |

risk amplitude 比手调 **低 1.7377 dB**。即无论全局还是 per-location 的自适应剂量都劣于固定强度，说明 HJ 的局部性完全由 gate 保证，dose 应保持常数 0.5。

## 诚实结论

- DT：adaptive plateau 退出 ≥ 手调（+0.0458 dB）。
- HJ：gate 已经负责“在哪些位置投影”，全局 strength 缩放与 per-location risk 缩放都有害；应保持固定 strength 0.5、amplitude=constant。
- 这本身是一个可写进论文的分析点：介入时机/强度自适应对“响应尺度正则（DT）”有益，但对“响应方向修正（HJ）”无效，因为后者的局部性由 gate 保证，全局强度应固定。
