# SEARCH-002: DT/HJ mechanism re-derivation

本搜索不复活 DT-CovMatch 或 HJ-PatchNCE 的旧超参数配置。它保留两条历史收益线的可计算对象：DT 的 latent endpoint response dispersion 与 HJ 的 harmful direction constraint，并重新推导为 LTTR。

LTTR 使用固定 `z/-z` 对构造每张图的 endpoint latent tangent，以冻结 first-use generator 为局部参考。`tangent` 分支只约束相对 tangent energy；`safe` 分支进一步只惩罚高风险结构区域里的 endpoint mean-direction reversal。两者都只在训练中启用，推理路径与 plain 完全一致。

运行顺序：

```bash
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage gate
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage screen
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage full
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage extend
```

confirmation20 在整个本地搜索中封存。
