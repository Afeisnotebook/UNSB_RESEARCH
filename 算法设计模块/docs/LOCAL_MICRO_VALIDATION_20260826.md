# 本地真实数据微验证（2026-08-26）

## 结论

当前 GitHub baseline 已通过真实本地数据上的工程级确定性微验证：官方 unpaired 抽样语义被保留，确定性 reflection padding 的前向等价与严格 CUDA backward 成立，同 seed 的一步完整训练可以生成字节完全相同的 G/F/D/E checkpoint。

当前精简 baseline 对 2026-08-11 历史 checkpoint 的推理结果可以稳定重放，但不与当时完整研究代码生成的 PNG 字节一致。这说明两棵代码树不是同一条历史数值轨迹，不表示当前实现仍有不确定性。项目现决定以当前严格确定性实现作为**新的 canonical baseline**；历史 PNG 字节一致性只用于旧结果取证，不再作为后续工作的启动门槛。

机器可读结果见 `算法设计模块/evidence/LOCAL_MICRO_VALIDATION_20260826.json`。

## 1. 环境与输入

- 被测提交：`98cbf333c66857fc632e0b22a67efabdbc471314`
- Python：3.13.5
- PyTorch：2.6.0+cu124；CUDA runtime：12.4
- GPU：NVIDIA GeForce GTX 1660
- AIO 真实训练视图：5 域，每侧 500 张，128×128 训练配置
- paired 微验证域：FoggyCityscapes，3 张 held-out 图像
- 历史 `14_net_G.pth` SHA-256：`b5fbed2318f976e9f5a9c9799db5b82305d1d250daeb5b0f593382b4443888ff`

本地数据和 checkpoint 没有提交到 Git；仓库只保存验证事实和身份哈希。

## 2. Sampler oracle：固定排序配对被明确否决

在 500 A / 500 B 的真实 AIO 训练视图上：

| 路径 | 样本数 | 同域 | 同文件名 | B 唯一数 |
|---|---:|---:|---:|---:|
| A/B 各自排序后一一固定配对 | 500 | 500 | 500 | 500 |
| 当前官方 unaligned 路径，seed=2041，epoch 1 前 100 | 100 | 17 | 0 | 86 |
| 当前官方 unaligned 路径，seed=2041，epoch 2 前 100 | 100 | 24 | 0 | 93 |

同 seed 从全新进程状态重建后，两 epoch 的路径序列完全一致；epoch 1 与 epoch 2 不同；seed=2042 与 seed=2041 不同。

这证明需要控制的是随机序列及其可恢复状态，而不是把随机过程删除。`clean_reexploration` 初轮的固定排序配对会把 unpaired 训练改成 100% 同名、100% 同域配对，不能并入 canonical baseline。

## 3. 确定性 reflection padding

在一张真实 FoggyCityscapes 训练图像上，padding=3：

- 当前 `DeterministicReflectionPad2d` 与 `F.pad(..., mode="reflect")` 前向逐元素相等，最大绝对误差 0；
- 当前实现与 2026-08-10 历史 `deterministic_ops.py` 前向逐元素相等；
- 当前实现可在 `torch.use_deterministic_algorithms(True)` 下完成 CUDA backward；
- 同输入重复 backward 的梯度逐位相同；
- 原生 `reflection_pad2d_backward_cuda` 在同一严格模式下按预期报无确定性实现错误。

因此，替换掉的是 PyTorch 原生 reflection-pad CUDA backward 的实现级非确定性，不是 UNSB 的桥噪声、latent、time index、patch sampling 或 unpaired B 抽样。

## 4. 历史 checkpoint 推理

当前 baseline 对同一历史 epoch-14 checkpoint 和 3 张 held-out 图像连续运行两次，共 18 张 `real/fake_1..5` PNG，18/18 字节哈希一致。说明当前代码自身的推理重放成立。

交叉 oracle 结果：

- 用 2026-08-10 完整历史代码重跑，3/3 个 `fake_5` 与 2026-08-11 已保存输出字节一致；
- 用当前精简 baseline 重跑，3/3 个 `fake_5` 与历史已保存输出字节不同；
- 把两个代码路径在加载 checkpoint 后重置到完全相同的 CUDA RNG 状态，五个 NFE 的浮点 tensor 仍有很小的数值差异；
- 对 3 张图的 PSNR 差为 `[-0.013943, +0.006264, +0.002966] dB`，SSIM 差绝对值不超过 `0.000725`。

这不是“当前实现不确定”：当前实现重复运行完全一致。它表示完整历史研究树与当前精简树之间存在数值执行/RNG 消费契约差异，因此不能把当前结果冒充历史 PNG 的字节重放。若未来要法证式复原旧实验，仍需单独做 historical oracle；若目标是从当前干净实现开始新探索，则该差异不是 blocker。

历史 endpoint 脚本还携带当前精简 baseline 已删除的 UA/NCE CLI 参数。当前 `train.py/test.py` 在 `seed >= 0` 时直接启用严格确定性；旧脚本不能原样作为新 runner，必须由当前冻结配置重新生成命令，不能靠吞掉未知参数伪装兼容。

## 5. 一步真实训练 twin gate

使用真实 AIO 数据、seed=2041、batch=1、128×128、`lambda_GAN=lambda_SB=lambda_NCE=1`、`tau=0.01`、5 个 time steps，各运行一次完整更新。两次运行均通过严格确定性 backward，损失显示值一致：

```text
G_GAN 0.987; D_real 1.002; D_fake 0.000; G 7.535;
NCE 6.605; SB 0.011; NCE_Y 6.469
```

保存的四个 checkpoint 在两个独立运行目录中字节完全相同：

| 文件 | SHA-256 |
|---|---|
| `1_net_G.pth` | `e6aeeac70773173471222e2df7398c4a62941095b254d1a5607e0378e0dd15cc` |
| `1_net_F.pth` | `bef23719da91dc02a092cc382f0c4cab88aaa369df9cffd7ec7b0b64b58dc083` |
| `1_net_D.pth` | `41353a8759c3ff868cdffc836f12bae2d0f9bd223ee7cb867701a4f9022355da` |
| `1_net_E.pth` | `d7d00891535e1e88d24e96ac94a9da769a802fb93dfd442d685eee673a8163cb` |

## 6. 启动裁决

已通过：

1. 真实数据上的官方 unpaired sampler 语义与同 seed 重放；
2. deterministic reflection padding 的前向等价和严格 CUDA backward；
3. 当前 baseline 自身的历史 checkpoint 推理重放；
4. 一步完整训练的跨进程字节级 twin gate。

尚未覆盖、但不阻断把当前代码定为新 canonical：

1. 完整历史代码/输出的法证式 evaluator oracle；
2. 长程 canonical twin replay；
3. full-state resume 后的下一随机 bundle 与 optimizer/scheduler/sampler 一致性；
4. 方法 lane 激活前共同父状态和在线 controller/access gate。

当前代码已经可以用于后续讨论、实现和小规模验证。昂贵长跑前建议补短程 plain/method twin、激活前 anchor 和必要的 full-state resume 检查；这些检查用于保证**新实验**公平，不再承担重放旧不确定轨迹的任务。
