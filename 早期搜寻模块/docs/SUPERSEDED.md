# 已证伪 / 不再推进的方向

> 这一页是「为什么后来不再做这些」的集中说明。保留它，是为了避免未来的自己或协作者重复走同一条弯路。

## 一、训练侧：naive train-rollout

- 做法：从 epoch 0 开始，用 stop-gradient 的 covariance calibrated endpoint 替换训练中间状态，不修改 GAN/PatchNCE/SB loss。
- 结果：严格 baseline recheck 后输给 plain joint；大测试集（300 张）上仍负向。
- 为什么死：全量替换 path 会形成 off-policy training states，并压低 z_sensitivity，反而削弱 proposal diversity。
- 现在的位置：不是「训练阶段利用不确定性」整体失败，而是「这个粗糙的第一版 rollout」失败。后续训练端研究改成 additive 的 `u_match` / `DT-CovMatch`。

## 二、训练侧：confidence 降权 / mpweight+uband / side-car netU

- 这些是 `UNSB_C21` 里 UA-123/124 训练端的几种形式。
- 结果：matched q50 下训练端候选没有一个稳定通过 gate；`ua123_conf_train` 甚至 −7 dB。
- 为什么死：confidence 降权破坏 UNSB 的 MSE anchor；side-car `netU` 的 teacher-student 校准不稳定；低秩区域协方差在 M=4 时样本太少。
- 现在的位置：训练端最终转向 **log-U consistency**，而不是直接改训练目标的权重或加 side-car。

## 三、test-time：UA123 作为独立 uncertainty-aware 机制

- 早期 UA123 rank1 test-time 有 `+0.13 ~ +0.19 dB` 的小增益，一度被视为主线。
- 末步消融后被发现：去掉末步基本归零，且 `constant_*_last`（普通末步常数收缩）可以复现甚至超过它。
- 结论：它很大程度只是 terminal damping 的复杂包装，不能作为论文主贡献。

## 四、训练侧：固定全局 early20 窗口

- 三域时 `early20` 有明显正信号，一度想当最终方法。
- 六域 final6 下 `early20` 反而 −0.633 dB，final logU 漂到 −9.x。
- 结论：固定全局窗口过硬，不能跨域泛化。被 `decay10to20`（先 warmup、短窗口、再退回 plain）替代。

## 五、high-U 图像 / 像素级门控

- 想按 U 高低对图像或像素做差异化调制。
- 结果：高 U 在不同域的含义不一致（RainCityscapes 高 U 更受益，Foggy/LowLight 高 U 更容易受损）。
- 结论：会把叙事写散，实验也容易翻车，不采用。

## 六、架构方向：专家 / routing / MoE / per-domain generator

- 早期讨论过用退化专家、prompt routing、门控模型等解决多域问题。
- 结论：被明确排除。研究主线始终是「共享 generator + 桥路径的 covariance/uncertainty 正则」，而不是退化成多分支或专家系统。

## 七、PatchNCE harmful-joint 早期尝试

- `UNSB_PatchNCE2` 和 `UNSB_Patch/archive` 里的 HU-PatchNCE harmful-joint 验证。
- 结果：没有形成干净、稳定、可跨 seed 复现的增益。
- 结论：没经过确定性口径的 patch 收益不可信。后续由算法设计模块的 clean-room + determinism 收口。

## 八、方法论教训（适用于之后所有工作）

1. 先定 baseline 口径，再开始扫超参；baseline mismatch 会制造假阳性。
2. 没有实际评估过的配置不能写进结论（早期出现过把未评估的 λ 写成峰值的情况）。
3. 所有训练/推理/评估必须同一 manifest、同一 split、同一 seed、同一指标脚本。
4. 收益要过 matched control 和 safety gate，再谈是否进入更大规模。
5. 单 seed 开发结果只能说“开发信号”，不能写成稳定结论。
