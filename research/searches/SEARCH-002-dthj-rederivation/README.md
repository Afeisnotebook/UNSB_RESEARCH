# SEARCH-002: DT/HJ mechanism re-derivation

本搜索不复活 DT-CovMatch 或 HJ-PatchNCE 的旧超参数配置。它保留两条历史收益线的可计算对象：DT 的 latent endpoint response dispersion 与 HJ 的 harmful direction constraint，并重新推导为 LTTR。

LTTR 使用固定 `z/-z` 对构造每张图的 endpoint latent tangent，以冻结 first-use generator 为局部参考。初始 tangent、one-epoch pulse 和 direction barrier 都在 400 步短程为正、800 步反转，因此全部关闭。

随后回查 SEARCH-001 的真实 HJ 轨迹：step1200 在 discovery10 为 `+0.804544 dB`、6/6 域正。直接加载 checkpoint 扩展到未参与 screen 的 discovery70（420 张）后仍为 `+0.710548 dB`、6/6 域正、SSIM `+0.020316`、LPIPS `-0.034900`。据此把 HJ 重写为有限期方向导航：按数据曝光在 1.6 epochs 后启用，6.4 epochs 后永久 handoff 给 plain。handoff 后 400 次 plain 更新仍为 `+0.660975 dB`。

当前唯一候选是 `hj_finite`，分类 `positive_but_fragile`。完整证据见 [EXP-L1-SEARCH-002-DTHJ-20260827](../../../experiments/L1-local/EXP-L1-SEARCH-002-DTHJ-20260827/README.md)。

运行顺序：

```bash
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage gate
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage screen
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage full
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage extend
python research/searches/SEARCH-002-dthj-rederivation/run_search.py --stage verify4090 --verify-milestones 30000 60000 120000
```

confirmation20 在整个本地搜索中封存。
