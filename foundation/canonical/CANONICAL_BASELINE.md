# 最后一轮使用的干净确定性基座

> 冻结日期：2026-08-26
>
> deterministic 代码锚点：`b9dc39f07e90653f7b3c6eb0726f86cc2ecfcc48`；默认关闭的新搜索扩展锚点：`495a092`
>
> 适用范围：以当前仓库为新的 canonical baseline，开展后续 DT、HJ、HNEK 或新方法讨论与实验。

## 1. 裁决

当前仓库可以作为**新的干净确定性基座**。

验收标准不是让新实现逐字节复刻过去三个月中某棵旧研究树的输出，而是：

1. 保留原始 UNSB 的算法语义和随机变量；
2. 删除实现级不可控非确定性；
3. 同一份新代码、同一配置、同一 seed 能精确重放；
4. baseline 与方法在同一新 canonical 下公平比较。

历史完整代码生成的 PNG 与当前精简代码不逐字节相同，只是一条 provenance 记录，**不是新基座的阻断项**。新实现本来就不应被要求复刻旧实现级非确定轨迹。

## 2. 已经闭合的内容

- 官方 unaligned A/B 抽样语义保留；没有用固定排序的一一配对冒充“确定性”。
- `DeterministicReflectionPad2d` 与原生 reflection padding 前向逐元素相等。
- 严格 CUDA deterministic 模式下 backward 成立，重复 backward 梯度逐位一致。
- 当前代码对同一 checkpoint 的重复推理输出字节一致。
- 真实 AIO 数据上的两次独立一步训练，G/F/D/E 四组 checkpoint 均字节一致。
- DT、HJ、HNEK 的代码入口仍然存在，确定性修复没有删除三个算法的核心实现。

机器证据见 [evidence.json](../../experiments/L0-contract/EXP-L0-CANONICAL-MICRO-20260826/evidence.json)，解释见 [REPORT.md](../../experiments/L0-contract/EXP-L0-CANONICAL-MICRO-20260826/REPORT.md)。

## 3. 明确没有被改掉的随机性

以下是 UNSB 算法的一部分，必须保留：

- unpaired B 抽样；
- 数据增强随机性；
- bridge noise；
- latent `z`；
- time index；
- PatchNCE patch sampling。

确定性的含义是这些随机量由 seed 和可复核的随机流驱动，而不是把它们全部删除。固定 A/B 同名配对会改变任务定义，已被本地数据验证明确否决。

## 4. 后续实验的公平比较契约

每轮正式实验只需冻结并记录：

1. 当前 Git commit；
2. 数据 manifest/hash 与实际路径映射；
3. 完整 CLI/config、seed、环境和输出目录；
4. plain 与方法相同的初始化、数据协议和评估协议；
5. 方法激活前的 anchor 应一致。

最理想的做法是从同一个完整训练状态分叉；如果采用从头连续训练，也必须依赖当前严格确定性实现，并在方法激活前核对网络与关键训练量一致。旧历史 checkpoint、旧 UA/NCE 参数和旧 PNG 不得作为新 canonical 的隐含依赖。

短程 twin replay、full-state resume 和共同父节点检查仍然是昂贵长跑前的良好保险，但它们服务于**新实验的公平性**，不再承担“复刻旧随机轨迹”的错误任务。

## 5. 当前边界

- 本仓库是可审计代码基座，不包含全部原始数据和大型 checkpoint。
- 实际迁移到 4090/5090 前仍需核对服务器数据身份和依赖环境。
- 过去 DT/HJ/HNEK 的收益数字属于历史研究证据；后续比较必须全部在这套新 canonical 下重新解释。
- 本文只裁决工程基座是否可用，不裁决三个算法是否正确或最终有效。

## 6. 下一步

现在可以直接进入后续内容讨论、实验协议冻结和小规模方法验证。若要启动昂贵长训练，先补一个短程 plain/method twin 与激活前 anchor 检查即可；无需先证明当前代码能逐字节重放旧研究树。
