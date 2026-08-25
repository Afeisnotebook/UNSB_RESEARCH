# DT-CovMatch 重构规格

## 一句话定位

DT-CovMatch 不是“后验协方差估计”，而是一个训练期 additive functional
regularizer：在 domain × bridge-time 坐标里，把当前生成器随机 endpoint
proposal 的对数分歧 ``log U`` 约束到冻结 teacher 的分布尺度上。最终评测不启用它。

## 算法本体（保留）

给定一个 batch 的带噪桥状态 ``X_t``、时间索引 ``time_id``、时间归一化值
``t_norm`` 以及每个样本的 domain key：

1. 把 batch 按 domain 分组，每组独立执行下面的 MC 估计。
2. 用当前 generator 采样 ``m`` 个 endpoint proposal：
   ``Y_k = netG(X_t, time_idx, z_k)``，``k=1..m``。
3. 用冻结 first-use teacher 做同样采样，``teacher`` 是当前 ``netG`` 的
   deep copy，参数冻结，永不更新。
4. 对每个 MC 集合计算桥方向统计：
   - ``D_k = (Y_k - X_t) / (1 - t_norm)``
   - ``v_bar = mean_k D_k``
   - ``U_pix = channel-mean of Var_k(D_k)``
   - ``signal = channel-mean of v_bar^2``
   - 区域池化后 ``U_reg_norm = pool(U_pix) / (pool(signal) + eps)``。
5. 当前端 ``U_reg_norm`` 允许反传（``detach_uncertainty=False``）；
   teacher 端一律 detach + no_grad。
6. 取对数并做数值下限：
   ``log_current = log(clamp(U_current, floor))``，
   ``log_teacher = log(clamp(U_teacher, floor))``。
7. 在 ``(domain, time_id)`` 上维护 teacher ``log U`` 的 EMA mean/var。
   用“更新前”的统计得到 ``mu``、``sigma``：
   ``z = (log_U - mu) / sigma``。
8. 对 ``z_current``、``z_teacher`` 做 clip，再用 smooth-L1 匹配，
   先按 image 聚合，再按 domain group 平均。
9. 每个 group 贡献相同权重，得到标量 ``loss``；再用 EMA 更新 teacher 统计。

最终损失是 ``base_UNSB_loss + lambda * loss``，其中 ``lambda`` 由 epoch
schedule 给出。算法不新增可学习参数，不替换 endpoint，也不乘在 bridge MSE
上；它只是一个附加正则项。

## 训练期协议（要保住收益的协议）

- warmup：纯 UNSB 训练到 e20，作为 teacher 的 first-use 快照来源。
- DT 激活段：约 25 epoch，``lambda`` 使用
  ``ramp_hold_cosine_decay``：epoch 1→5 ramp 到 ``base=0.001``，
  epoch 5→15 hold，epoch 15→25 cosine 衰减到 0。
- 后续 plain：DT 关闭，继续 plain 到 e200。
- 评测：``dtcov_lambda=0`` 或直接使用 plain 模型，关闭一切 DT 干预。

## 工程边界

重构后只保留上述算法。原来散落的 scheme12/123、low-rank covariance、side-car
uncertainty head、bridge gating、risk weighting、诊断 flush 等都不属于
DT-CovMatch 最优分支，已移出核心实现。

## 有意简化点

- 原实现以 ``model`` 为全局状态，通过 ``model._ua_*`` 字段串数据。这里改成
  ``DTCovMatch`` 拥有显式状态（teacher、DomainTimeStats、iter），数据通过参数
  传入，不再依赖隐藏全局字段。
- 诊断写入从核心 loss 中移除；诊断逻辑可以在上层另行调用。
- 原实现有 ``preserve_torch_rng`` 包裹每段 MC 采样。这里保留同样语义，但把
  teacher/current 采样分别放进独立 context，RNG 账目更清晰。

## 参考实现对照

原实现：

- ``01_DT_CovMatch/code/uncertainty_rollout.py``
- ``01_DT_CovMatch/code/sb_model.py``

本文法对应其中 ``compute_direction_statistics``、
``covariance_match_regularizer_grouped_domain``、``_dtcov_*``、
``ensure_frozen_teacher``、``sample_endpoints`` 以及 train.py 中的
``scheduled_ua_train_reg_lambda``。
