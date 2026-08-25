# 复现与启动指南（当前仓库边界）

> 更新：2026-08-24
> 本文取代确定性修复前的旧指南。旧指南中 DT `+0.8875 dB`、“`reflection_pad2d` 导致不可约 1 dB 方差”和未上传的 `refactor/_runs/*.sh` 不再是当前口径。

## 1. 先说清仓库能与不能做什么

当前仓库可以：

- 审计 plain UNSB、DT、HJ 和 HNEK 的核心实现；
- 运行数学原语、模型适配器和 harness 元数据测试；
- 核对已汇总的 JSON/TSV 裁决与文档数字；
- 作为最后一轮重建数据/训练编排的源码基座。

当前仓库不能直接：

- 从零一键复现所有服务器实验；
- 不依赖外部数据/checkpoint 复算动机模块 raw JSONL；
- 验证 `MANIFEST.sha256` 中未上传的 `/home/yc/.../raw/*` 文件；
- 使用已删除或未上传的历史 orchestration scripts。

因此，最后一轮的第一项工作是重建并冻结 orchestration，而不是假设仓库已经是完整发行包。

## 2. 克隆后的静态与 CPU 验证

在仓库根目录运行：

```bash
python -m pytest -q
python foundation/harness/self_test.py
python -m compileall -q foundation research
```

需要 Python、PyTorch、NumPy 和 pytest。PyTorch/CUDA 版本必须在真实 GPU smoke 中单独记录，不建议在跨平台 `requirements.txt` 中强行锁定一个 CUDA wheel。

## 3. 当前权威性能口径

确定性 clean core（seed=2026）：

| 对比 | Δ PSNR |
|---|---:|
| DT best − plain | −0.2677 dB |
| HJ true − plain | +0.0381 dB |
| HJ roll − plain | −0.7521 dB |
| HJ true − roll | +0.7901 dB |

来源：[clean-core evidence.json](../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/evidence.json)。

HNEK e200 开发候选：

```text
--model hnek_search
--hnek_gamma 0.25
--hnek_coord residual
--hnek_horizon_mode physical
--hnek_partial all
```

它在 seed=2026 paired-development 上的 macro PSNR delta 为 +0.7884 dB，4/5 域为正，但 LowLight 为负。这是 development result，不是 confirmatory result。

**不要**使用 `--model sb --hnek true` 当作当前候选；该开关对应 legacy 冻结 `gamma=0.5` HNEK 参照，该参照在开发实验中失败。

## 4. 最后一轮需重建的输入

在新服务器上，必须先准备并存档：

1. 五域训练 manifest 和每张图像 hash（Foggy、LowLight、RainCity、RSCity、Snow）；
2. 与开发集隔离的确认集 manifest；
3. plain/HNEK 完整有效配置、源码 commit 和环境指纹；
4. 每个 seed 独立的 name/checkpoint/result 目录；
5. 完整 model/optimizer/scheduler/RNG 恢复状态；
6. 逐图配对指标，不只是汇总均值。

历史 provenance 中的 `/home/yc/...` 路径必须映射到新环境并核对文件身份，不得在源码中盲目复制旧绝对路径。

## 5. 最小复现顺序

1. 从全新 clone 通过第 2 节测试。
2. 跑单 batch 数据身份 smoke，确认训练仅使用 unpaired views，GT 只用于离线评估。
3. 跑 plain 和 HNEK 的真实模型 forward/backward 与 checkpoint-resume smoke。
4. 用 seed=2026 进行同环境复现，不在看到结果后改参数。
5. 复现通过后，补独立 seed，用 seed-level delta 统计裁决。
6. 最后仅在冻结候选上解封未触碰确认集。

## 6. 确定性边界

当前代码已用手工确定性 reflection pad 消除旧 CUDA backward 问题，同 seed 3-epoch smoke 权重 hash 一致。但这不意味跨 PyTorch/CUDA/GPU 版本必然逐位一致；跨环境应先定义指标与权重容差，再解释算法差异。

更详细的实验门禁见 [../../CURRENT_STATE_CN.md](../../CURRENT_STATE_CN.md)。
