# UNSB 当前科学状态与最后一轮启动契约

> 更新：2026-08-26
> 用途：本文是仓库的首要状态入口。当历史文档与本文冲突时，以本文、各模块最新裁决和机器可读 evidence 为准。

## 1. 当前不可混淆的结论

| 问题 | 当前结论 | 证据边界 |
|---|---|---|
| All-in-One 与 Single 是否完全相同 | 不同，条件方向几何呈阶段依赖差异 | 观察性结果，不是因果机制 |
| 固定 Epoch 4–5 “过度压缩窗口”是否稳定 | **不支持，已关闭** | seed=2031 翻正；seed=2030 仅 3/5 域同号 |
| Epoch 1 的 AIO 方向发散 | 6/6 seeds 一致为正 | 可作现象级动机，不是算法靶点 |
| `U / U_reg` 是什么 | 方向分歧/空间方向分散程度 | 不是 true posterior covariance，不是 calibrated uncertainty |
| DT-CovMatch | 早期非确定实现有正数字，但确定性 clean rerun 为 **−0.2677 dB** | 不再作为当前有效方法 |
| HJ-PatchNCE | clean rerun 相对 plain 为 **+0.0381 dB** | 视为基本消失；`true−roll` 不等于方法相对 plain 有效 |
| HNEK `gamma=0.25` | e200 单 seed paired-development 为 **+0.7884 dB**，4/5 域正，LowLight −0.1813 dB | 唯一存活的**开发候选**，不是已确认算法 |

HNEK 表中的 95% CI `[0.5916, 0.9933]` 是固定 seed=2026 与开发集条件下的配对样本 bootstrap，**不包含训练 seed 之间的不确定性**，也不能抵消 9 个变体搜索与开发集反复使用带来的选择偏差。

2026-08-26 收到的 `clean_reexploration_work_20260826.zip` 另有一轮数值：plain `13.6032`、DT `+0.0192`、HJ `-0.1636`、HNEK FULL `+0.2663 dB`。该轮的文件哈希和逐图统计自洽，但未通过 canonical/evaluator/sampler/controller/access/resume 门禁，**只作为失败运行取证，不更新上表的科学结论，也不得作为最后一轮训练父节点**。详见 [算法设计模块/docs/CLEAN_REEXPLORATION_AUDIT_20260826.md](./算法设计模块/docs/CLEAN_REEXPLORATION_AUDIT_20260826.md)。

## 2. 当前总裁决

1. 仓库目前**没有一个已经跨 seed、未触碰确认集验证的最终算法**。
2. DT/HJ 的历史正收益已被确定性 clean rerun 降级为历史轨迹现象，不能作为最后一轮的默认起点。
3. `hnek_g0.25` 是当前唯一值得带入最后一轮的方法候选，但首先要尝试证伪，不是继续给它补动机。
4. 动机模块只支持“共享训练改变条件方向几何，且差异具有阶段/seed/域依赖”；它不支持固定窗口或已知机制靶点。
5. 最后方法必须对 Schrödinger Bridge 本身形成可辨认贡献；不再把通用 routing、gradient surgery、confidence weighting 或额外网络包装成 SB 贡献。

## 3. 最后一轮的最小科学门禁

### Gate 0：基座复核

- 从全新 clone 运行根目录测试、harness 自测和全量 Python 编译。
- 固定源码 commit、数据 manifest hash、训练/评估配置和每个 seed 的输出目录。
- 评估禁用 test-time UA/TTO；默认 `ua_scheme none`。
- 保留官方 unpaired sampler；确定性来自受控随机流，不能用固定 A/B 配对替代。
- HNEK/HJ/DT 应从同一新 canonical 出发。完整 full-state 分叉是首选；若采用从头连续训练，必须在方法激活前核对确定性 anchor 一致。

2026-08-26 的本地真实数据微验证已经通过 sampler 语义、deterministic reflection padding、当前代码自身推理重放和一步完整训练 twin gate；两次一步训练的 G/F/D/E checkpoint 字节完全一致。因此当前仓库已被接受为**新的干净确定性 canonical baseline**。它无需逐字节复刻确定性修复前的完整历史研究树；历史输出差异只保留为 provenance。详见 [算法设计模块/docs/CLEAN_DETERMINISTIC_BASELINE_20260826.md](./算法设计模块/docs/CLEAN_DETERMINISTIC_BASELINE_20260826.md) 和 [算法设计模块/docs/LOCAL_MICRO_VALIDATION_20260826.md](./算法设计模块/docs/LOCAL_MICRO_VALIDATION_20260826.md)。

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
- `DT/HJ has stable gains under the clean deterministic protocol`
- `hnek_g0.25 is confirmed / generalized / robust`
- 用固定 seed 的图像级 CI 替代训练 seed 级稳定性
- 用开发集反复搜索后的结果宣称 confirmatory

## 5. 权威顺序与历史文档

当数字或状态冲突时，按以下顺序处理：

1. `evidence/**/*.json` 中的机器可读原始裁决；
2. 本文与 `算法设计模块/docs/FINAL_STATUS.md`；
3. `动机搜寻模块/reports/WINDOW_FINAL_VOTE_CN.md`；
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
python 算法设计模块/code/harness/self_test.py
python -m compileall -q 动机搜寻模块/code 算法设计模块/code
```

测试通过只证明源码契约、数据身份工具和核心数学原语未破坏，不代替 GPU 真实模型 smoke、同 seed 复现和跨 seed 确认。
