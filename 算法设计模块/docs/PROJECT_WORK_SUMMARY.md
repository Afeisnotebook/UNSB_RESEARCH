# UNSB 项目阶段性工作总结（事实记录）

日期：2026-08-20
覆盖范围：约 2026-06 至 2026-08-20 的 clean-room 重构、验证、确定性修复与桥原生探索阶段。

> 本文只做事实记录，不修饰。所有数字均来自项目内的 metrics / eval / state 文件。单 seed 数字均不构成稳定性结论。

## 一、主线做了哪几件事

1. 把历史 DT-CovMatch 与 HJ-PatchNCE 从数千行混杂实现重构为干净包（`dtcov` / `hj`），并建立统一 harness（数据身份、配置冻结、确定性 RNG、checkpoint、配对 bootstrap）。
2. 修复 baseline 的历史 argparse bug（`--mode`/`--model` 冲突、netG/netD/netE choices、缺 `--seed`）和 HJ 集成 bug（netG encode_only 单 layer 返回结构错误）。
3. 发现并修复训练非确定性：`reflection_pad2d` backward / `torch.bmm` / `adaptive_avg_pool2d`，最终做到同 seed 逐位复现。
4. 从“AIO 不确定性”（DT/HJ）转向“桥原生”（HNEK），做了 HNEK frozen 生死测试和变体搜索。

## 二、关键数字（按阶段）

### 1. 非确定性阶段（修复前）

- DT vs plain：单 seed +0.8875 dB；3-seed 均值 +0.6996 dB（CI [0.1712, 1.2280]，不含 0）。
- HJ vs plain：单 seed +2.7533 dB；3-seed 均值 +2.07 dB（CI 含 0）；HJ true−roll 的 3-seed CI 含 0（归因未跨 seed 复现）。

### 2. 确定性修复

- 同 seed 两次 3-epoch GPU smoke 的 `3_net_G.pth` SHA256 一致，逐位复现达成。

### 3. 确定性单 seed 干净重跑（seed 2026）

| 对比 | delta |
|---|---:|
| DT best − plain | −0.2677 dB |
| HJ true − plain | +0.0381 dB |
| HJ true − roll | +0.7901 dB |

结论：DT/HJ 相对 plain 的收益基本消失；早期 +0.89 / +2.75 主要是特定非确定性训练轨迹的假象（确定性下 plain 基线抬升 1~2 dB）。

### 4. HNEK 桥原生 frozen 测试（base γ=0.5）

- e50：macro PSNR delta −0.7438 dB（CI [−1.0567, −0.4356]），1/5 域正。
- verdict：`STOP_E50_CLEAR_FAIL`。

### 5. HNEK 变体搜索（阶段3，确定性、单 seed、e50）

| 变体 | delta (dB) | 正域 | verdict |
|---|---:|---:|---|
| γ=0.5 参考（base HNEK） | −1.2328 | 0/5 | 失败 |
| γ=0.25 | +2.6173 | 4/5 | 转 e200 |
| γ=0.75 | −1.3139 | 0/5 | 失败 |
| γ=1.0 | +0.7860 | 4/5 | 转 e200 |
| 熵坐标 (X_t,Y) | +3.1481 | 4/5 | 转 e200 |
| horizon = index | −2.5825 | 1/5 | 失败 |
| horizon = mix | +0.7861 | 4/5 | 转 e200 |

注：`entropy_only` / `endpoint_only` 两个 partial 变体在整理本文件时尚未跑完。

## 三、核心结论

1. DT/HJ 作为“应用层”方法，在确定性干净口径下没有稳定正收益；早期收益主要是轨迹假象。
2. HNEK 的桥原生思想并非无效：base γ=0.5 失败，但 γ=0.25、γ=1.0、熵坐标 (X_t,Y)、horizon mix 等变体在单 seed e50 上出现明显正信号（+0.79 ~ +3.15 dB）。
3. 这些正信号仍是单 seed、e50、development，不是 confirmatory；需要 e200 确认 + 更好服务器的多 seed 才能下结论。

## 四、当前状态与未完成

- 阶段 3 搜索尚未完全结束（还剩 2 个 partial 变体）。
- 正信号变体未做 e200 确认，也未做多 seed。
- 最终论文定位未定：若 HNEK 正变体在多 seed 下仍成立，桥原生方法有戏；否则退为“干净框架 + 诚实的轨迹假象 / 负结果”。
