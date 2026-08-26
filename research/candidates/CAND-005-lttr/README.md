# CAND-005: Latent-Tangent Trust Region

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 从 DT 的 endpoint dispersion 与 HJ 的 harmful direction 重新推导的确定性训练期信任域。 |
| 当前结论 | `CLOSED_NEGATIVE`：确定性门禁通过，但 tangent/pulse/direction 三条 lane 均在 800 步反转。 |
| 与旧算法关系 | 不复用 DT/HJ 的旧 loss 或超参数网格；保留它们指出的可计算对象。 |
| 先看哪里 | [规格](./SPEC.md) → `lttr/core.py` → [SEARCH-002](../../searches/SEARCH-002-dthj-rederivation/README.md)。 |

LTTR 以固定 `z/-z` endpoint 对构造每张图的 latent tangent chart，避免 batch=1 下 domain-time EMA z-score 的稀疏/饱和问题。冻结 first-use generator 只提供局部参考，不进入推理。确定性工程判断成立，但效果判断为负：tangent 在 400 步 `+0.339362 dB` 后于 800 步变为 `-1.097963 dB`；one-epoch pulse 和 direction barrier 在 800 步分别为 `-1.757175 / -0.807077 dB`。源码保留用于解释为什么输出状态约束不等价于 HJ 的 PatchNCE backward 方向控制。
