# SEARCH-001: clean directional search

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | plain/DT/HJ/HNEK anchors 与 DCUM/LBST/PTQ/AEB 的冻结分级搜索、合成和排序控制器。 |
| 当前结论 | `LOCAL_COMPLETE_CANDIDATE_FROZEN`：HNEK 为 `positive_but_fragile` 唯一候选；下一步 4090 matched 验证。 |
| 时间线位置 | T8 通过工程门，T9 完成本地八 lane、合成、复赛和 12k 延长。 |
| 先看哪里 | [完整本地报告](../../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/REPORT.md) → [唯一候选](../../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/CANDIDATE.json) → `src/protocol.py`。 |

这是最后一轮 clean UNSB 的冻结分级搜索控制器。L0 工程门禁已在
[`EXP-L0-SEARCH-001-GATE-20260826`](../../../experiments/L0-contract/EXP-L0-SEARCH-001-GATE-20260826/README.md)
通过。本地全部阶段随后在 [EXP-L1-SEARCH-001-DIRECTIONAL-20260826](../../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/README.md) 完成；confirmation20 全程未打开。

## 本地裁决

- stage1 前三：HNEK+DCUM、HNEK、DT；HJ 排名第四，虽在 1200 点 6/6 域正，但轨迹不稳。
- stage2：只有 HNEK 的最后三点均值为正（`+0.339392 dB`）。
- stage3：HNEK 在 8k/10k/12k 的 matched delta 均值为 `+0.006322 dB`，因此分类为 `positive_but_fragile`。
- 下一步：固定候选和 seed=2026，只运行 HNEK/plain 的 30k/60k/120k 4090 验证。

## 冻结内容

- anchors：plain、DT-CovMatch、HJ-PatchNCE、HNEK；
- 新机制：LBST、PTQ、DCUM、AEB；
- stage 1：小视图方向筛选，并构造 new+new 与 legacy+new 两条 synthesis；
- stage 2：前三名进入完整 development view；
- stage 3：首名延长，输出一个带风险与复现说明的 `CANDIDATE.json`；
- verify4090：仅消费冻结候选，在 30k/60k/120k matched checkpoints 比较。

排序规则、lane 定义和阈值位于 `src/protocol.py`。训练/恢复状态位于 `src/runtime.py`。评估器只允许 `discovery` split，所有输出显式记录 `confirmation20_opened: false`。

## 运行顺序

先显式提供本机数据身份参数并运行工程门禁：

```bash
python research/searches/SEARCH-001-clean-directional/run_search.py \
  --stage gate \
  --manifest <DATA_MANIFEST.csv> \
  --train-view <materialized-unpaired-train-view> \
  --data-root <paired-evaluation-root> \
  --output runs/SEARCH-001/gate \
  --gpu 0
```

门禁检查 plain twin 精确一致、full-state resume、重复评估和 PTQ block mass。通过后，新建 `EXP-L0-...` 记录并锁定 commit/manifest hash，才可运行 `stage1`；不得直接跳到 `verify4090`。

默认 CLI 中的本地绝对路径只是 2026-08-26 工作机便利值，不属于协议身份；正式运行以显式参数和输出的 `PROTOCOL_LOCK.json` 为准。
