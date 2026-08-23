# 投稿级论文骨架（evidence-first）

> **历史结构模板：** 其“等待阶段 2/3”状态已过时，且当前尚无跨 seed/未触碰数据确认的最终方法。仅保留为写作结构参考，不得填入开发数字后直接宣称投稿结论。

> 本文件是结构模板，不是结论。所有数字必须来自统一 harness，并在对应结果文件中可审计。
> 状态：等待阶段2 `CLEAN_CORE_RESULTS.md` 与阶段3 `summary.tsv` 后填充。

## 1. 一句话贡献

我们研究的是：在无配对、受限端点条件下，**如何为 Schrödinger Bridge 的 amortized endpoint 条件律选择坐标与介入时机**。

贡献必须落在桥原生主线上：

- UNSB 受限端点律；
- 无配对 EROT；
- 剩余时域归一化坐标与具体介入原则。

不声称首创：

- preconditioning；
- gradient surgery；
- 不确定性校准；
- 新参考设计；
- 新噪声 schedule。

## 2. 方法

### 2.1 桥原生候选

令 `h=1-t`，定义：

```text
R = (Y - X_t) / h^γ
Y = X_t + h^γ R
```

固定候选轴：

1. `γ ∈ {0.25, 0.5, 0.75, 1.0}`；
2. 熵坐标 `(X_t,R)` vs `(X_t,Y)`；
3. 熵权重 `physical h` vs `index (T-i)/T` vs `mix`；
4. 部分应用：entropy-only / endpoint-only / all。

### 2.2 应用层证据

- DT：对 amortized endpoint 响应几何做有界函数正则/信任域一致性约束。
- HJ：PatchNCE forward 不变，对结构有害方向施加约束的梯度修正。

这两者不冒充桥原生贡献。

## 3. 实验

### 3.1 确定性

- 手工确定性反射 pad；
- CuBLAS workspace + strict deterministic；
- 同 seed 两次 GPU smoke `3_net_G.pth` SHA256 相同。

证据：`refactor/_runs/DETERMINISM_FIX.md`。

### 3.2 核心干净重跑

DT plain/best、HJ plain/true/roll，3 seed=2026/2027/2028。

结果槽：`refactor/_runs/metrics_clean_core/CLEAN_CORE_RESULTS.md/json`。

### 3.3 桥原生搜索

每个变体 1 seed=2026、最多 e50，粗门控；连续 2~3 个非正变体停止。

结果槽：`refactor/_runs/hnek_search/summary.tsv`。

## 4. Novelty 边界与诚实降级

- 若阶段3全部或多数为负：只报告“剩余时域归一化在该设置下不改善”，不包装成桥贡献。
- 若 HNEK 负结果在修正变体中仍不脱离负区间：桥原生主结论按负结果写。
- DT/HJ 只作为应用层证据，不用于支撑桥原生方法创新。

## 5. Limitation 清单

- 单卡、single-seed development；
- reflection-pad 非确定项已通过 deterministic pad 消除；但 CuBLAS/deterministic 仍属于实现层面设置，论文写清楚；
- HNEK T3 为 saturated paired development，不是 confirmatory；
- 最终投稿级多 seed 需在更好服务器上补做。
