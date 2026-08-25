# SEARCH-001: clean directional search

这是最后一轮 clean UNSB 的冻结分级搜索控制器。当前状态为 `IMPLEMENTED_UNRUN`：实现已经进入仓库，但尚无登记在 `experiments/` 的运行证据，因此不能据此改变任何候选的科学状态。

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
