# UNSB 当前科学状态与最后一轮启动契约

> 更新：2026-08-28
> 用途：本文是仓库的首要状态入口。当历史文档与本文冲突时，以本文、各模块最新裁决和机器可读 evidence 为准。

## 0. SEARCH-005 最新裁决

SEARCH-005 已按修正后的路线一目标完成：它不搜索退出窗口、paired-PSNR 控制器、whole-state 分支选择或 handoff，而是从 DT/HJ/HNEK 与 plain 的反转证据出发，构造 target-blind、自消隐/不变量明确的 UNSB 数学算子。

六类 Generation-1 机制和四次因果修订均通过工程门禁并完成相应本地筛查。没有候选通过 2400-step 持续收益门禁，也没有启动 full100、第二 seed 或 confirmation20。唯一冻结项 `G1-GAME-PCOA` 仅为 `weak_fallback`：400/800/1200 分别约 `+0.044/+0.075/+0.193 dB`，但 1600/2400 反转为 `−0.570/−0.861 dB`，自身峰值到终点回撤约 `2.10 dB`。其严格保持原生 Adam 步长范数的 G2-NPOOA 在 400 为 `+0.231 dB`，800 已为 `−0.966 dB`，因此 coupled-game 机制按两代上限关闭。

当前权威结论是：**正窗口真实存在，但本轮没有找到可顺畅维持的路线一算法。** PCOA 不是论文算法，也不自动获得 4090 资格；只有在明确接受 weak-fallback 风险时，才可用已冻结 full100 命令作一次高算力证伪。下一步若转向 gap-aware handoff，必须另立路线二计划，不能改写 SEARCH-005 的结论。详见 [SEARCH-005 结果](../research/searches/SEARCH-005-long-horizon-operator-discovery/RESULTS.md) 与 [DEC-20260828-SEARCH005-ROUTE1-STOP](./records/DEC-20260828-SEARCH005-ROUTE1-STOP.md)。

## 0.1 SEARCH-002 历史裁决

2026-08-27，SEARCH-002 没有围绕旧 DT/HJ 做超参网格，而是先重推导输出空间 LTTR，再回到实际 HJ backward 方向对象。LTTR tangent、one-epoch pulse、direction barrier 均在 800 步反转，当前实现关闭。

原 HJ 的 SEARCH-001 step1200 checkpoint 在未参与 screen 的 discovery70（420 张）上相对 matched plain 为 `+0.710548 dB`，6/6 域正，SSIM `+0.020316`，LPIPS `-0.034900`，最差域 `+0.174754 dB`。将 HJ 固定为 `[1.6,8.0)` 真实数据 epoch 的有限方向导航并在 step1200 handoff 给 plain 后，step1600 仍为 `+0.660975 dB`。step2000 的 `+3.791830 dB` 主要由 matched plain 坍塌放大，只作为盆地稳定性诊断。

当时 CAND-002 `ITER-007-finite-horizon-handoff` 以 `positive_but_fragile` 冻结为第一候选，HNEK 降为递补一，并预注册了 full-view 4090 门禁。该历史排序现已由上面的 SEARCH-005 裁决覆盖，不再自动执行。详见 [SEARCH-002 报告](../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/REPORT.md) 和 [当时决策](./records/DEC-20260827-HJ-FINITE-HORIZON-LOCAL-CANDIDATE.md)。

## 0.2 SEARCH-001 历史裁决

2026-08-26，SEARCH-001 在新的 deterministic canonical 上完成八条初筛 lane、两条合成、完整视图复赛以及总冠军/plain 到 12k updates 的等量延长。`confirmation20_opened=false`。

| 对象 | 最新状态 | 核心证据 |
|---|---|---|
| HNEK | **DEVELOPMENT_FROZEN** / `positive_but_fragile` | stage2 最后三点 `+0.339392 dB`；8k/10k/12k 均值仅 `+0.006322 dB`；最终 3/6 域正 |
| DT | 递补二，当前仍不重开调参 | 2k 为 `+0.566439 dB`、5/6 域正，但 3k/4k 反转，最后三点 `-0.586284 dB` |
| HJ | 本轮淘汰 | 1200 点 `+0.804544 dB`、6/6 域正，但 800 点 `-0.724102 dB`，小视图均值仅 `+0.042388 dB` |
| LBST/PTQ/DCUM/AEB | **CLOSED_NEGATIVE**（当前实现） | 无 standalone lane 保持正的晚期轨迹；DCUM 合成也未通过 stage2 |
| HNEK+DCUM | 递补一 | stage1 第一，但 stage2 最后三点 `-0.381743 dB`，4k `-2.506480 dB` |

SEARCH-001 当时选择 HNEK 的裁决保留为历史事实，但已被上述 SEARCH-002 裁决取代。详见 [完整本地报告](../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/REPORT.md) 和 [DEC-20260826-SEARCH-001-LOCAL-WINNER](./records/DEC-20260826-SEARCH-001-LOCAL-WINNER.md)。

## 1. 当前不可混淆的结论

| 问题 | 当前结论 | 证据边界 |
|---|---|---|
| All-in-One 与 Single 是否完全相同 | 不同，条件方向几何呈阶段依赖差异 | 观察性结果，不是因果机制 |
| 固定 Epoch 4–5 “过度压缩窗口”是否稳定 | **不支持，已关闭** | seed=2031 翻正；seed=2030 仅 3/5 域同号 |
| Epoch 1 的 AIO 方向发散 | 6/6 seeds 一致为正 | 可作现象级动机，不是算法靶点 |
| 六域 shared-clock regret | seed=2051 下三个 bridge time 均为正：`0.0201 / 0.0369 / 0.0164` | held-out 图像 cross-fit + bootstrap；不包含训练 seed 不确定性，不等于因果恢复伤害 |
| `U / U_reg` 是什么 | 方向分歧/空间方向分散程度 | 不是 true posterior covariance，不是 calibrated uncertainty |
| DT-CovMatch | 早期非确定实现有正数字，但确定性 clean rerun 为 **−0.2677 dB** | 不再作为当前有效方法 |
| HJ-PatchNCE | 旧 continuous clean rerun仅 **+0.0381 dB**；finite-horizon discovery70 为 **+0.710548 dB** | 历史 `positive_but_fragile` 窗口候选；SEARCH-005 后不再是当前路线一答案 |
| HNEK `gamma=0.25` | 历史 e200 为 **+0.7884 dB**；SEARCH-001 延长的最后三点均值为 **+0.006322 dB** | 递补一，不是已确认算法 |

HNEK 表中的 95% CI `[0.5916, 0.9933]` 是固定 seed=2026 与开发集条件下的配对样本 bootstrap，**不包含训练 seed 之间的不确定性**，也不能抵消 9 个变体搜索与开发集反复使用带来的选择偏差。

2026-08-26 收到的 `clean_reexploration_work_20260826.zip` 另有一轮数值：plain `13.6032`、DT `+0.0192`、HJ `-0.1636`、HNEK FULL `+0.2663 dB`。该轮的文件哈希和逐图统计自洽，但未通过 canonical/evaluator/sampler/controller/access/resume 门禁，**只作为失败运行取证，不更新上表的科学结论，也不得作为最后一轮训练父节点**。详见 [DEC-20260826-CLEAN-REEXPLORATION-AUDIT](./records/DEC-20260826-CLEAN-REEXPLORATION-AUDIT.md)。

MOT-001 六域结果的完整解释见 [阅读指南](../research/motivations/MOT-001-aio-path-geometry/READING_GUIDE_CN.md) 和 [冻结报告](../experiments/L2-medium-4090/EXP-L2-MOTIVATION-SIXDOMAIN-20260824/source_snapshot/final_delivery/reports/UNSB_SIXDOMAIN_PHASE_FINAL_CN.md)。它使用一个新训练 seed=2051；5000 次图像 bootstrap 只刻画 seed 内 held-out 图像抽样不确定性。

## 2. 当前总裁决

1. 仓库目前**没有一个已经跨 seed、未触碰确认集验证的最终算法**。
2. DT/HJ/HNEK 的历史正窗口均保留为证据；SEARCH-005 没有给旧名字保护名额，并已把窗口/控制器与数学算子路线明确分开。
3. 当前没有通过本地持续收益门禁的第一候选。PCOA 只作为路线一 `weak_fallback` 冻结；finite-horizon HJ 与 HNEK 保留为历史递补证据，不自动进入 4090。
4. 动机模块支持“共享训练改变条件方向几何，且差异具有阶段/seed/域依赖”；六域扩大实验还支持单训练 seed 下正的 held-out shared-clock regret。它不支持固定窗口、跨 seed 稳定、因果恢复伤害或已知机制靶点。
5. 最后方法必须对 Schrödinger Bridge 本身形成可辨认贡献；不再把通用 routing、gradient surgery、confidence weighting 或额外网络包装成 SB 贡献。SEARCH-005 没有产生满足这一条件且持续为正的最终方法。

## 3. 最后一轮的最小科学门禁

### Gate 0：基座复核

- 从全新 clone 运行根目录测试、harness 自测和全量 Python 编译。
- 固定源码 commit、数据 manifest hash、训练/评估配置和每个 seed 的输出目录。
- 评估禁用 test-time UA/TTO；默认 `ua_scheme none`。
- 保留官方 unpaired sampler；确定性来自受控随机流，不能用固定 A/B 配对替代。
- HNEK/HJ/DT 应从同一新 canonical 出发。完整 full-state 分叉是首选；若采用从头连续训练，必须在方法激活前核对确定性 anchor 一致。

2026-08-26 的本地真实数据微验证已经通过 sampler 语义、deterministic reflection padding、当前代码自身推理重放和一步完整训练 twin gate；两次一步训练的 G/F/D/E checkpoint 字节完全一致。因此当前仓库已被接受为**新的干净确定性 canonical baseline**。它无需逐字节复刻确定性修复前的完整历史研究树；历史输出差异只保留为 provenance。详见 [canonical acceptance](../foundation/canonical/CANONICAL_BASELINE.md) 和 [L0 real-data micro validation](../experiments/L0-contract/EXP-L0-CANONICAL-MICRO-20260826/REPORT.md)。

### Gate 1：同 seed 跨环境复现

- 先用 seed=2026 复现 plain 与 `hnek_g0.25`。
- HNEK 必须显式冻结为：

```text
--model hnek_search
--hnek_gamma 0.25
--hnek_coord residual
--hnek_horizon_mode physical
--hnek_partial all
```

- 不得用 `--model sb --hnek true`；该开关是已失败的 legacy `gamma=0.5` 参照。
- 若 plain 或方法无法复现到预注册容差，停止算法解释，先查数据/配置/环境。

### Gate 2：训练 seed 级确认

- 在不改超参、不重新选窗口、不重新挑 checkpoint 的前提下补 2–4 个独立 seed。
- 以“每个 seed 一个配对 delta”为统计单位，报告 mean、std 和 seed-level CI；图像 bootstrap 只作 seed 内辅助。
- 必须报告逐域结果，LowLight 不能被 macro 平均掩盖。

### Gate 3：未触碰数据确认

- 搜索过程使用的 T3/saturated paired-development 不再承担最终确认。
- 在看结果前封存新的图像清单或数据集，一次性解封评估。
- 若跨 seed 或未触碰确认失败，将 HNEK 收口为诚实负结果，不再开新变体救援。

## 4. 明确禁止的表述

- `true posterior covariance`
- `calibrated predictive uncertainty`
- `fixed over-compression window`
- `HJ has stable gains across seeds / is confirmed under the clean deterministic protocol`
- `hnek_g0.25 is confirmed / generalized / robust`
- 用固定 seed 的图像级 CI 替代训练 seed 级稳定性
- 用开发集反复搜索后的结果宣称 confirmatory

## 5. 权威顺序与历史文档

当数字或状态冲突时，按以下顺序处理：

1. `evidence/**/*.json` 中的机器可读原始裁决；
2. 本文与 [DEC-20260824-ALGORITHM-STATUS](./records/DEC-20260824-ALGORITHM-STATUS.md)；
3. [WINDOW_FINAL_VOTE_CN.md](../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/WINDOW_FINAL_VOTE_CN.md)；
4. 两个模块 README；
5. 其他带日期的历史分析/计划文档。

以下文档保留了确定性修复前的历史实验或当时的计划，不是当前结论：

- `ABLATION_RESULTS.md`
- `ADAPTIVE_SCHEDULE.md`
- `BASELINE_DECISION.md`
- `DIAGNOSTIC_ANALYSIS.md`
- `EXPERIMENT_PLAN.md`
- `METHOD_GROUNDING.md` / `METHOD_NARRATIVE.md` 中的旧收益数字
- `HNEK_RUN_PLAN.md`

这些文档可用于理解思路如何被证伪，不得直接复制其结论到新论文或新实验协议。

## 6. 仓库完整性边界

- 仓库包含精简源码、裁决文档、汇总 JSON/TSV 和图。
- 原始数据、大型 checkpoint、大部分 raw JSONL 与历史 orchestration scripts 未上传；因此当前仓库是**可审计研究基座**，不是一键端到端复现包。
- `PATH_MAP.json`、`CODE_IDENTITY.json`、`DATA_MANIFEST.json` 和 `MANIFEST.sha256` 中的 `/home/yc/...` 是原环境 provenance，不是新机器的默认路径。
- 新环境必须重建路径映射并核对 manifest/hash，不得通过创建同名空目录绕过检查。
- 2026-08-26 clean re-exploration 包没有完成 repair return；包内 runner 和初轮 checkpoint 被隔离为历史取证材料，未直接并入可执行基座。

## 7. 克隆后的基座验证

```bash
python -m pytest -q
python foundation/harness/self_test.py
python -m compileall -q foundation research
```

测试通过只证明源码契约、数据身份工具和核心数学原语未破坏，不代替 GPU 真实模型 smoke、同 seed 复现和跨 seed 确认。
