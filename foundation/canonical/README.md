# Canonical deterministic UNSB

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | plain UNSB 和候选薄接入共用的可执行源码，是所有新实验的代码父节点。 |
| 当前结论 | `READY`：保留官方 unpaired sampler 和受控随机流，以 deterministic reflection padding 去掉实现级不确定 CUDA backward；真实数据 L0 已通过。 |
| 时间线位置 | T5 确定性 clean core → T7 新 canonical 接受。 |
| 先看哪里 | [验收说明](./CANONICAL_BASELINE.md) → [复现入口](./REPRODUCE.md) → [确定性修复](./contracts/determinism/DETERMINISM_FIX.md) → `src/`。 |

当前 canonical 保留 official unpaired sampler、bridge noise、latent、time 与 PatchNCE sampling，并用确定性 reflection padding 消除实现级不可控 CUDA backward。

- 验收契约：[CANONICAL_BASELINE.md](./CANONICAL_BASELINE.md)
- 复现入口：[REPRODUCE.md](./REPRODUCE.md)
- 确定性修复：[contracts/determinism/DETERMINISM_FIX.md](./contracts/determinism/DETERMINISM_FIX.md)
- L0 真实数据微验证：[EXP-L0-CANONICAL-MICRO-20260826](../../experiments/L0-contract/EXP-L0-CANONICAL-MICRO-20260826/README.md)

`src/models/dtcov_model.py`、`hj_model.py` 和 HNEK 文件目前是候选接入的薄层/历史兼容位置；候选科学身份以 `research/candidates/` 为准。

边界：这里定义可执行基座，不定义 DT/HJ/HNEK 是否有效。候选状态只由 `experiments/` 与 `decisions/` 更新。
