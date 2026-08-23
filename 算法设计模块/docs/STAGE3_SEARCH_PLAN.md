# 阶段3：HNEK 桥原生思想的有界搜索计划

> 本文件在阶段2跑完后执行。目标对象固定：对 amortized endpoint 条件律做“剩余时域归一化”。
> 不偏离到 DT/HJ 或一般多任务正则。

## 负参照

- HNEK frozen e50：macro PSNR delta **−0.7438 dB**（95% CI [−1.0567,−0.4356]）。
- 只有明显脱离该负区间的变体才值得深挖；连续 2~3 个变体失败/无方向即停止，写“无简单桥原生归一化修法能改善”。

## 单轴变体矩阵

每个变体 1 seed=2026，最多 e50，用 frozen adjudicator 的粗门控判定正/负/平。

| 轴 | 候选 | 定义 |
|---|---|---|
| 归一化指数 | γ ∈ {0.25, 0.5, 0.75, 1.0} | `R=(Y-X_t)/(1-t)^γ`，`Y=X_t+(1-t)^γ R`；γ=0.5 是 frozen HNEK |
| 熵坐标 | `(X_t,R)` vs `(X_t,Y)` | critic 输入的 endpoint 坐标 |
| 熵权重 | physical `h=1-t` / index `(T-i)/T` / 混合 | SB 熵项前面的时域权重 |
| 部分应用 | entropy-only / endpoint-only / all | 只改熵坐标、只改 endpoint 重建、或两者都改 |

## 执行约束

1. 不修改 frozen spec/阈值/schedule/数据口径；方法变体用独立搜索脚本或新 adapter，不改原 HNEK frozen 树。
2. 每次记录：变体定义、delta、CI、诊断量（drift / entropy-h / invariant）。
3. 1 seed、e50、单 GPU 串行；不与阶段2并行。

## 停止规则

- 连续 2~3 个变体失败或无明显方向 → 停止，写诚实结论。
- 任一变体明显脱离负区间 → 记录并停止发散，后续只围绕该轴做最小深挖。

## 待阶段2完成后确认

- 搜索 harness 落点：优先复用 refactor/baseline + HNEK shim 的新变体模型；评估需对齐 frozen T2/T3 口径或等价的干净基准。
