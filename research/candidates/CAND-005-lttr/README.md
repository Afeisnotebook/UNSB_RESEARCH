# CAND-005: Latent-Tangent Trust Region

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 从 DT 的 endpoint dispersion 与 HJ 的 harmful direction 重新推导的确定性训练期信任域。 |
| 当前结论 | `IMPLEMENTED`：CPU 数学测试及方法级 GPU twin/resume preflight 通过，尚无效果裁决。 |
| 与旧算法关系 | 不复用 DT/HJ 的旧 loss 或超参数网格；保留它们指出的可计算对象。 |
| 先看哪里 | [规格](./SPEC.md) → `lttr/core.py` → [SEARCH-002](../../searches/SEARCH-002-dthj-rederivation/README.md)。 |

LTTR 以固定 `z/-z` endpoint 对构造每张图的 latent tangent chart，避免 batch=1 下 domain-time EMA z-score 的稀疏/饱和问题。冻结 first-use generator 只提供局部参考，不进入推理。`safe` 版本增加一侧方向 barrier，只阻止高 latent-risk、高结构权重区域的 endpoint mean direction 反转。
