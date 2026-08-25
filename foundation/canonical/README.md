# Canonical deterministic UNSB

当前 canonical 保留 official unpaired sampler、bridge noise、latent、time 与 PatchNCE sampling，并用确定性 reflection padding 消除实现级不可控 CUDA backward。

- 验收契约：[CANONICAL_BASELINE.md](./CANONICAL_BASELINE.md)
- 复现入口：[REPRODUCE.md](./REPRODUCE.md)
- 确定性修复：[contracts/determinism/DETERMINISM_FIX.md](./contracts/determinism/DETERMINISM_FIX.md)
- L0 真实数据微验证：[EXP-L0-CANONICAL-MICRO-20260826](../../experiments/L0-contract/EXP-L0-CANONICAL-MICRO-20260826/README.md)

`src/models/dtcov_model.py`、`hj_model.py` 和 HNEK 文件目前是候选接入的薄层/历史兼容位置；候选科学身份以 `research/candidates/` 为准。
