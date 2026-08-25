# Clean Re-exploration 包独立审计（2026-08-26）

## 结论

`clean_reexploration_work_20260826.zip` 的初轮返回包在字节和逐图统计层面自洽，但它没有完成 repair contract，不能作为最后一轮训练父节点。它的正确定位是：**失败运行取证 + 修复需求输入**。

仓库只吸收不会改变历史方法口径的基础修复和显式状态接口；没有把包内未校准的 runner、离线 controller、固定排序 sampler 或旧绝对路径直接并入可执行基座。

## 1. 输入身份与可复核结果

- 工作包 SHA-256：`45697e5b75e69042e07e4d755f5edb351198903a4eae641b15af3a2fd19a86cd`
- 初轮嵌套 return SHA-256：`d793432abf5d3179a8637eb5cf4a8c38d9ef9a6757235a6ebd205dac465bf919`
- 嵌套 manifest：45/45 文件存在且哈希匹配。
- `PER_IMAGE_METRICS.csv`：23,040 行，无非有限值、无完全重复行。

重新聚合的 e200 paired-development 结果：

| lane | macro PSNR | 相对 plain | 95% paired image-bootstrap CI | 正域 |
|---|---:|---:|---:|---:|
| canonical plain | 13.60318 | — | — | — |
| DT | 13.62240 | +0.01923 | [-0.04783, +0.08746] | 2/5 |
| HJ | 13.43954 | -0.16364 | [-0.23626, -0.09530] | 1/5 |
| HNEK FULL | 13.86945 | +0.26628 | [+0.18374, +0.34686] | 4/5 |

这些 CI 只反映固定 seed 内的图像抽样，不包含训练 seed 不确定性。

## 2. 确定性修复到底是什么

官方 UNSB 源码（审计提交 `d1f644f7777e19d5afe5aea3e5cb4bd3afd9b88b`）在 `models/networks.py` 和 `models/ncsn_networks.py` 中共有 15 处 `nn.ReflectionPad2d` 使用。其 CUDA backward 依赖非确定归约；即使随机种子相同，长训练也可能产生逐位不同的梯度轨迹。

当前仓库已将这 15 处全部替换为 `DeterministicReflectionPad2d`：

- slice/index-select/concat 实现反射边界；
- forward 与 `F.pad(..., mode="reflect")` 逐位等价；
- backward 不调用 `reflection_pad2d_backward_cuda`；
- 严格 deterministic、CuBLAS workspace、cuDNN/TF32 配置与同 seed twin smoke 共同构成确定性门。

这个修复**只消除相同随机轨迹下的实现级非确定性**。它没有删除 UNSB 本来就需要的随机变量：unpaired B 抽样、数据增强、桥噪声、latent `z`、time index 和 PatchNCE patch sampling 仍应存在，只是必须由可保存、可恢复的 RNG/sampler 状态驱动。

因此，包内把 A/B 排序后固定一一配对、每个 epoch 从相同顺序重放，不是 reflection-pad 修复的必然结果，而是另一个会改变训练分布的 sampler 改动。

## 3. 与 padding 无关的阻断项

### 3.1 canonical/evaluator/sampler 未校准

本轮 plain e200 为 `13.6032 dB`，历史同任务 authoritative plain 约为 `18.8467 dB`，相差约 `5.24 dB`。padding 的前向等价替换本身不足以解释该量级差异。

包内没有完成：

- authoritative checkpoint evaluator oracle；
- authoritative loader/config sampler replay；
- canonical checkpoint replay 交叉验证。

在三项闭合前，无法区分评估器、数据变换、配对顺序、训练配置或 checkpoint 载入漂移。

### 3.2 方法 lane 不是真正的共同 full-state 分叉

初轮四条 lane 只是同 seed 独立初始化并分别训练，没有从 canonical pre-e1/pre-e5/post-e20 继承完整状态。

- HJ 在应当尚未启用的 e1 已比 canonical 低约 `0.824 dB`；
- HNEK e0 输出也没有证明与 canonical 父状态一致；
- DT 早期数值相同是经验现象，不是父 checkpoint 契约。

参数初值相同或 seed 相同，不等于 optimizer、scheduler、RNG、sampler、global step 都来自同一父节点。

### 3.3 controller 是训练后离线审计，不是在线控制

保存的 DT/HJ/HNEK controller history 为空，训练循环没有真实执行 `observe -> record -> decide`。`CONTROLLER_SIGNALS.json` 是 checkpoint 生成后的 post-hoc 计算，不能证明控制器曾驱动训练或可从中断状态恢复。

HNEK handoff 还缺少完整 repeat floor、安全包络和校准后的判据；未触发不能解释为已经找到合理的继续训练时机。

### 3.4 paired-target 访问闸门没有接入真实读取

初轮 ledger 约 960 行全是被拒绝的自测 probe，没有 freeze 后 evaluator 的实际 allowed target read。训练/评估仍存在直接 `Image.open()` 路径，guard 也会把某些合法的 unpaired B 目录误判为 paired target。

因此 ledger 不能证明 paired target 在训练冻结前保持密封。

### 3.5 full-state resume 与长程编排未闭合

- checkpoint 保存了 controller 字典，但恢复函数没有把它载回 controller 实例；
- DT teacher 只记 hash，没有把 teacher state 作为可恢复训练状态保存；
- 生产 CLI 没有真正使用 resume path；
- determinism gate 只覆盖 plain，没有覆盖方法状态和 teacher；
- launch 脚本没有自动完成四 lane、watcher、裁决和唯一回传。

### 3.6 结果裁决和报告不完整

- mechanical summary 没有给 HNEK FULL 标签；
- final report 漏掉 HNEK FULL 和已知工程失败；
- attempt ledger 被写成空数组；
- repair return、repair frozen spec、oracle audits 和 `TRAINING_FROZEN.ok` 均不存在。

### 3.7 还有两个独立的官方 UNSB 算法属性

这两个问题不是 CUDA 不确定性，也不应在 plain 基线里被静默“修好”：

1. official rollout 使用非均匀物理时刻约 `[0, 0.5, 0.74, 0.86, 0.94, 1]`，但 SB 熵权重使用均匀索引 `(T-i)/T`；真实剩余时域与索引权重不一致。
2. `ResnetBlock_cond.forward()` 在遍历层时反复执行 `out = layer(x)`，前一层加入的显式 time embedding 被后续层覆盖，形成已记录的 `time-dead` 行为。

当前权威 plain 刻意保留这两个官方行为，以免改变基线定义。HNEK `gamma=0.25` 是针对物理剩余时域/残差坐标的独立开发候选；不能把 HNEK 信号解释为 deterministic reflection padding 的附带收益。若要修复 time conditioning，必须注册为新的模型变体并从共同 canonical 父状态单独证伪。

## 4. 本次并入仓库的安全修复

1. 补回 2026-08-25 repair contract，作为运行前硬门禁而不是执行结果。
2. `BaseModel.set_train_epoch()`：plain 为 no-op，方法显式接收 physical epoch。
3. 标准训练入口同时更新两条 dataset epoch，并在首 batch 前调用模型的 physical-epoch hook。
4. DT 支持从 canonical post-e20 `netG` state 严格注入冻结 teacher，并记录稳定 SHA-256。
5. DT/HJ 的 epoch gate 不再依赖进程局部的 scheduler 调用次数。
6. HNEK 增加无参数、幂等的 ON/OFF 状态开关，同时保留当前权威 `gamma=0.25` 默认值和参数校验。
7. 增加 HNEK 状态不变性和 DT teacher 身份单测。

没有并入包内 `clean_reexploration/train_executor.py` 等执行文件，因为其 sampler、fork、controller、guard、resume 和路径问题尚未达到基座标准。

## 5. 重新进入 GPU 训练前的顺序

1. evaluator oracle：历史 checkpoint 在新 evaluator 上复现到预注册容差。
2. sampler/config oracle：保留 UNSB 的随机训练过程，但固定并保存全部 RNG/sampler 状态；禁止用固定配对替代 unpaired 抽样。
3. canonical replay：从同一 pre-e1 full-state 重放并在多个 anchor 做逐位或预注册数值核对。
4. fork gate：HNEK/HJ/DT 在激活前与 canonical 的网络、optimizer、scheduler、sampler 和下一随机 bundle 相同。
5. controller gate：真实在线调用、history 非空、状态可恢复，HNEK repeat floor 与安全判据完成校准。
6. access gate：所有 paired target 读取走唯一入口，freeze 前零读取，freeze 后 ledger 与逐图评估一一对应。
7. 最后才冻结新 spec/code/data identity 并启动 GPU lane。

任一 oracle 失败时，先停止算法解释，不用新的方法数字覆盖基座问题。

## 6. 本地真实数据补充微验证

2026-08-26 在本地 GTX 1660 上完成了 sampler、reflection padding、历史 checkpoint 推理和一步完整训练 twin gate。当前 baseline 的同 seed 一步训练可生成字节完全相同的 G/F/D/E checkpoint，固定排序配对也被真实数据直接证明会把 unpaired 训练改成 100% 同名、100% 同域配对。

同时，当前精简 baseline 虽能稳定重放自身输出，但尚不与 2026-08-11 完整历史研究代码的 PNG 字节一致。因此工程微门通过不等于 authoritative evaluator/canonical oracle 已闭合。完整结果见 [LOCAL_MICRO_VALIDATION_20260826.md](./LOCAL_MICRO_VALIDATION_20260826.md)。
