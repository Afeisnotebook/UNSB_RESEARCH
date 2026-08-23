# 本地论文头图验证：初始共享桥扇出

本目录冻结一项纯基线、无方法分支的本地验证。研究对象只有：

```text
五个独立 Single-task plain UNSB
                vs
一个五域共享 All-in-One plain UNSB
```

核心问题不是 AIO 最终分数是否更低，而是 AIO 在第一个 epoch 后的随机条件恢复方向，是否同时超出：

1. Single epoch 1 的本域曝光匹配参照；
2. Single epoch 5 的总 optimizer-step 匹配参照；
3. Single epoch 1–5 的整个早期方向离散包络。

主统计量为有界、无分母放大的球面两两方向离散度 `D_sph`。旧 `U` 只作为单调相关的 secondary statistic。

新训练 seed 固定为 2041。held-out 图像从旧 discovery 池中排除历史上已经用于方向测量的 stem 后，按哈希盲选每域 20 张。它们是“相对旧 10 图子集的 held-out”，不是从未使用过的正式确认集。

执行代码位于：

```text
../code/headfigure_local/
```

大 checkpoint 与运行 raw 输出写到仓库外：

```text
E:/UNSB_Expl/UNSB_HEADFIGURE_LOCAL_20260824/
```

科学协议以 `HEADFIGURE_FROZEN_PROTOCOL.json` 为准。看过新 seed effect 后不得换样本、seed、桥时间、主统计量或主 checkpoint。
