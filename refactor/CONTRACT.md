> **历史/部分废弃**：本文是早期 clean-room 重构契约。trick 分类法（necessary_algorithmic / robustness / inertia / hack / unknown）仍有效；但“成本门”“L0/L1 才准 L2”等旧门控已被新目标取代。当前目标见 `PLAN.md`。

# 重构总规则（DT-CovMatch 与 HJ-PatchNCE 共用）

## 目标

把两个“能拿收益、但机制不清晰、工程堆叠、超参靠经验”的算法，重构为最小、优雅、可解释的实现，并在关闭干预后保住或超过当前已知最好收益。

## 每个算法必须交付

1. `SPEC.md`：算法本体（数学/机制）与工程包装的精确边界。
2. `TRICK_LEDGER.md`：当前实现里每一项超参与工程块的分类账。
3. `code/`：最小、优雅的重构实现。
4. `tests/`：纯 CPU 验证。
5. `REPORT.md`：保住了什么、删了什么、为什么、还缺什么证据。

## trick 分类法

- `necessary_algorithmic`：算法本体，删掉就不再是这个方法。
- `necessary_robustness`：数值稳定、收敛或确定性所必需。
- `inertia_legacy`：历史惯性保留，可删。
- `hack_artifact`：明显是绕 bug 或历史事故的 hack，可删或应改写。
- `unknown`：暂时无法判定，必须靠后续消融决定。

每一项都要给出：代码位置、它做了什么、当前分类、判断理由、保留/删除/改写建议。

## 最小实现要求

- 关闭干预时（eval-off）必须回到 plain 行为，理想是逐位或容差内等价。
- 不新增可学习参数，除非有充分理由并在 SPEC 中说明。
- 每个保留超参都要写“为什么存在、影响面、默认值依据”。
- 确定性要求：RNG 账目清楚，不做隐藏的全局 RNG 消耗。
- 核心 forward 尽量不动，干预尽量做成明确、可审计的局部改动。

## 验证层级（先便宜后贵）

- `L0` 静态/单元测试：CPU、零 GPU。
- `L1` 等价性测试：CPU，小张量 forward/backward 与原实现对齐，或明确声明“有意简化点”并说明差异。
- `L2` 收益复现：GPU，最少次数，见成本门。

## 成本门（重要）

- 只有 `L0`、`L1` 全部通过后，才允许进入 `L2`。
- 每个算法 GPU 新训练分支上限 3 次：1 次干净实现复现 + 最多 2 个 knock-out 消融。
- 固定 `seed=2026`、`128×128`，尽量复用已有 warmup / prefix / plain 锚点，不从零重训一切。
- GPU 总预算先按约 6 小时封顶；达到上限仍未收敛就停并回报，不继续烧。
- 不跑官方 test，不多 seed，不写稳定性结论。

## 禁止

- 改 seed、split、评测门槛或方法语义来“凑收益”。
- 复活 DTHS 或引入新方法。
- 覆盖或修改 `01_DT_CovMatch/`、`02_HJ_PatchNCE/`、`UNSB_Cov5/`、`UNSB_Patch/` 的原件（只读）。
- 把 `L1` 等价性当成 `L2` 收益证据，或编造任何收益。

## done 定义

`L0`、`L1` 全过，且 `SPEC.md` / `TRICK_LEDGER.md` / `code/` / `tests/` / `REPORT.md` 齐全；`L2` 能跑则跑，不能跑则明确写 `blocked` 与下一步。
