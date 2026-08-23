# 算法设计模块（Algorithm Design）

> 本模块是无配对 All-in-One 多天气图像恢复在 Schrödinger Bridge（UNSB）框架上的算法设计与实验收口，全部为 clean-room 实现 + 统一 harness 的可复现证据。口径：single seed=2026 paired-development，**非 confirmatory**。

## 一、目标（一句话）

在无配对、多天气统一恢复的 UNSB（Schrödinger Bridge）上，找出并验证一个真正「桥原生」的改进，使其足以支撑毕业论文章节与投稿级方法贡献，而不是停留在通用训练技巧。

## 二、我们做过哪些探索（时间线）

1. **clean-room 重构 + 统一 harness**：把历史 DT-CovMatch / HJ-PatchNCE 从混杂实现重构为干净包，建立数据身份 / 配置冻结 / 确定性 RNG / checkpoint / 配对 bootstrap。
2. **确定性修复**：手工确定性反射 pad + CuBLAS workspace + strict deterministic；同 seed 两次 smoke 的模型权重 SHA256 逐位一致。
3. **DT / HJ 干净核心**：在确定性口径下重跑，发现早期收益基本消失（见下）。
4. **桥原生缺陷定位**：发现官方 UNSB 熵项用「索引时域 (T-i)/T」而非真实「剩余时域 1-t」（C003），以及生成器显式时间注入被覆盖（time-dead，A2）。
5. **HNEK 变体搜索 + e200 确认**：围绕「剩余时域归一化」做 9 变体 e50 搜索，再对最强两个做 e200 确认；最终 `hnek_g0.25` 存活。

## 三、关键结果

### 1) DT / HJ 干净核心（确定性，单 seed=2026）

| 对比 | Δ PSNR |
|---|---:|
| DT best − plain | −0.2677 dB |
| HJ true − plain | +0.0381 dB |
| HJ roll − plain | −0.7521 dB |
| HJ true − roll | +0.7901 dB |

### 2) HNEK 9 变体（e50，最强项）

| 变体 | e50 ΔdB | 正域 | e50 verdict |
|---|---:|---:|---|
| hnek_coord_y | +3.1481 | 4/5 | PROCEED |
| hnek_g0.25 | +2.6173 | 4/5 | PROCEED |
| hnek_endpoint_only | +1.3484 | 4/5 | PROCEED |
| hnek_entropy_only | +0.9671 | 5/5 | PROCEED |
| hnek_g1.0 | +0.7860 | 4/5 | PROCEED |
| hnek_horizon_mix | +0.7861 | 4/5 | PROCEED |

### 3) e200 最终 verdict

| 变体 | e200 ΔdB | 95% CI | 正域 | verdict |
|---|---:|---:|---:|---|
| hnek_coord_y | −1.2164 | [−1.4174, −1.0153] | 2/5 | FAIL |
| hnek_g0.25 | **+0.7884** | [**+0.5916, +0.9933**] | 4/5 | **PASS** |

## 四、最终结论

- **唯一存活候选：`hnek_g0.25`**（γ=0.25、residual 坐标、physical horizon 权重、全量应用）。
- e200 macro PSNR delta **+0.7884 dB**，SSIM delta **+0.0355**，4/5 域为正；但 LowLightTrafficData 仍为 −0.18 dB。
- 确定性口径下，DT/HJ 相对 plain 无稳定正收益；桥原生「物理剩余时域归一化」方向有正信号，但只是单 seed development。

## 五、怎么看这个模块

```text
算法设计模块/
├── README.md         # 本文件（总览 + 结论）
├── code/
│   ├── baseline/     # clean-room UNSB 基线 + HNEK shim + 确定性 pad
│   ├── dt_covmatch/  # DT-CovMatch（应用层方法）
│   ├── hj_patchnce/  # HJ-PatchNCE（应用层方法）
│   └── harness/      # 统一 harness（数据/配置/确定性/checkpoint/bootstrap）
├── docs/
│   ├── METHOD_GROUNDING.md             # 数理 grounding
│   ├── METHOD_NARRATIVE.md             # 叙事与 novelty 边界
│   ├── FINDINGS.md                     # 必要机制 vs 工程惯性
│   ├── BASELINE_DECISION.md            # 收益基准口径
│   ├── EXPERIMENT_PLAN.md              # 消融/自适应计划
│   ├── BRIDGE_NATIVE_REASSESSMENT.md   # 桥原生缺陷定位（C003/A2）
│   ├── PAPER_SKELETON.md               # 论文骨架
│   ├── DETERMINISM_FIX.md              # 确定性修复审计
│   ├── DIAGNOSTIC_ANALYSIS.md          # 诊断分析
│   ├── PROJECT_WORK_SUMMARY.md         # 阶段性工作总结
│   └── ...
└── evidence/
    ├── CLEAN_CORE_RESULTS.md/json      # 干净核心结果
    └── hnek_search/
        ├── summary.tsv                 # 9 变体 e50 汇总
        ├── E200_CONFIRMATION.md/json   # e200 确认
        └── state/<variant>/...         # 权威 verdict / SUMMARY
```

阅读顺序建议：本 README → `docs/PROJECT_WORK_SUMMARY.md` → `docs/METHOD_GROUNDING.md` → `docs/BRIDGE_NATIVE_REASSESSMENT.md` → `evidence/hnek_search/E200_CONFIRMATION.md`。

## 六、Limitation

1. 单 seed=2026、paired-development、非 confirmatory，不能写稳定性/泛化结论。
2. 反射 pad 非确定性已用手工确定性 pad 消除；CuBLAS/deterministic harness 仍属实现层设置。
3. T3 评估集为 saturated paired development，不是未污染确认集。
4. `hnek_g0.25` 在 LowLightTrafficData 上仍为负，跨域稳健性未闭合。
5. 本机 4090 只做单 seed development，多 seed 未做。

## 七、下一步建议

1. 迁移到更好服务器：先同 seed=2026 复现 `hnek_g0.25`，再补 2~4 个独立 seed，报告 mean±CI。
2. 论文落点：围绕「amortized endpoint 条件律的剩余时域归一化（γ=0.25 + residual 坐标 + 真实物理 h 权重）」。
3. 若多 seed 失败，按负结果诚实收尾，不再为保历史叙事继续调参。
