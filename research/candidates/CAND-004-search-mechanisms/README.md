# CAND-004: Search mechanisms incubation

这是对 commit `495a092` 中、生命周期重构开始前已存在代码的登记，不是算法有效性裁决。

当前包含：DCUM 同域不同 stem unpaired marginal、LBST rollout EMA generator、PTQ physical-time quadrature、AEB antithetic endpoint averaging，以及 DT/HJ/HNEK 的额外训练状态恢复接口。

当前状态：`IMPLEMENTED`。四条新 lane 及 plain、DT、HJ、HNEK anchors 已在 [SEARCH-001](../../searches/SEARCH-001-clean-directional/README.md) 中冻结；在 L0 工程门禁、正式实验记录和决策完成前，不进入 HNEK/DT/HJ 的现有裁决，也不写入论文输出。
