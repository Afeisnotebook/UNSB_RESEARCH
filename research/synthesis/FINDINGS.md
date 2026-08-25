# 重构发现：哪些是必要机制，哪些是工程惯性（截至 CPU 阶段）

> **阶段性历史文档：** 本文的“能拿收益”表述发生在最终确定性 GPU clean rerun 之前。最终 DT/HJ 相对 plain 的收益已基本消失；见 [当前裁决](../../decisions/CURRENT.md)。

这两个算法“能拿收益但机制糊”，经过 clean-room 重构，结论已经清晰很多。

## DT-CovMatch

真正产生最好收益（+1.0439 dB）的机制只有一条：

- 按 domain 分组的 MC endpoint 分歧统计；
- 冻结 first-use teacher；
- `(domain, time)` 上的 teacher log-U EMA z-score；
- 当前模型与 teacher 的 smooth-L1 匹配；
- `ramp_hold_cosine_decay` 的短时附加正则（λ=0.001）；
- 最终 eval-off。

被认定为惯性/假象的：

- `ua_scheme=12` + `ua_train_rollout=True` 在训练桥构造里实际仍走 plain netG，是“开关开着但没作用”；
- side-car `netU`、scheme12/123 low-rank covariance、risk weighting、bridge gate、allowlist、诊断落盘都是死代码。

关键点：`grouped_domain` 不能简化成 batch 内统一 EMA，它是和同批 `homog/eqdom` 结果差异的核心。

## HJ-PatchNCE

真正产生最好收益（val-O +1.4729 dB，但归因 FAIL）的机制只有一条：

- 源图结构切方向（edge/SSIM joint）；
- ±step 反事实探针（`central_consensus`）；
- delta/risk + top-quantile gate + 绝对证据门槛；
- `project_conflicting_gradient`：forward 恒等，backward 投影掉与结构方向正对齐的冲突分量。

被认定为惯性/假象的：

- `bridge_*`、`structure_risk/softpos/posmargin/relational/pixel_project`、`corr_*`、`curation`、side-car `netU` 都是死代码；
- `nce_structure_event_z=0` 的事件门控恒为 1，直接省略；
- 大量 `loss_NCE_*` 诊断字段只是落盘噪声。

## 还没解决的关键问题

- HJ 的收益是“性能成立但归因不成立”：涨了，但未必是结构方向特异，可能是通用投影/稳健化。这是 ICLR 级工作必须正面回答的点。
- 以下项目是否真的必要，仍需 GPU knock-out 消融：DT 的 `grouped_domain` / 冻结 teacher；HJ 的 `boundary_scale` / `min_risk` / `central_consensus`。

## 对“工程复杂度”的回答

原先那堆复杂度里，大部分是历史叠块。每个算法真正有效的核心，都能用一个函数 + 一个 dataclass 表达清楚，参数数从几十个降到个位数。
