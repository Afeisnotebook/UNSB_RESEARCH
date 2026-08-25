# 算法模块当前权威状态

> 更新：2026-08-24
> 口径：clean deterministic，single seed=2026，paired-development，非 confirmatory。
> 跨模块总状态见 [../../CURRENT_STATE_CN.md](../../CURRENT_STATE_CN.md)。

## 1. 确定性基座

- 手工确定性 reflection pad 已取代会触发非确定 CUDA backward 的实现。
- CuBLAS workspace、deterministic algorithms、数据与辅助 RNG 隔离已统一。
- 同 seed、同配置的 3-epoch GPU smoke 两次 `3_net_G.pth` SHA256 逐位一致。
- 因此，“reflection pad 造成不可约 1–2 dB 运行方差”只是修复前历史，不是当前 limitation。

证据：[DETERMINISM_FIX.md](../../foundation/canonical/contracts/determinism/DETERMINISM_FIX.md)。

## 2. DT / HJ clean core

| 对比 | Δ PSNR |
|---|---:|
| DT best − plain | **−0.2677 dB** |
| HJ true − plain | **+0.0381 dB** |
| HJ roll − plain | **−0.7521 dB** |
| HJ true − roll | **+0.7901 dB** |

裁决：

- DT 在确定性 clean 口径下为负，不是当前有效方法。
- HJ 相对 plain 的收益基本消失。`true−roll` 说明 rollout 更差，不能被转述为 HJ 相对 plain 改进 `+0.7901 dB`。
- 旧非确定阶段的 DT `+0.8875 dB` 与 HJ 大幅正结果仅作历史轨迹证据，不再作权威性能结论。

机器可读证据：[clean-core evidence.json](../../experiments/L1-local/EXP-L1-DT-HJ-CLEAN-CORE-20260824/evidence.json)。

## 3. HNEK 搜索与 e200 延伸

9 个变体在同一 seed=2026 开发协议中进行 e50 搜索，最强两个延长到 e200：

| 变体 | e50 ΔdB | e200 ΔdB | e200 图像级 95% CI | 正域 | 开发裁决 |
|---|---:|---:|---:|---:|---|
| `hnek_coord_y` | +3.1481 | −1.2164 | [−1.4174, −1.0153] | 2/5 | FAIL |
| `hnek_g0.25` | +2.6173 | **+0.7884** | **[+0.5916, +0.9933]** | 4/5 | DEVELOPMENT_PASS_SINGLE_SEED |

`hnek_g0.25` 的当前精确定义：

```text
--model hnek_search
--hnek_gamma 0.25
--hnek_coord residual
--hnek_horizon_mode physical
--hnek_partial all
```

LowLightTrafficData 仍为 −0.1813 dB。CI 仅描述固定训练 seed 与开发数据下的配对样本不确定性，不包含训练 seed 方差。因为经过 9 变体搜索且 T3 已饱和，该结果必须被称为“开发候选”，不是确认、泛化或稳健结论。

机器可读证据：[E200_CONFIRMATION.json](../../experiments/L2-medium-4090/EXP-L2-HNEK-SEARCH-E200-20260824/evidence/E200_CONFIRMATION.json)。文件名中的 `CONFIRMATION` 是历史阶段名，不表示 confirmatory study。

## 4. 当前方法裁决

- 当前没有已确认的最终方法。
- 唯一可进入最后一轮的候选是 `hnek_g0.25`。
- 最后一轮先做同 seed 跨环境复现，再做冻结配置的多 seed，最后使用未触碰数据一次性确认。
- 任一门禁失败时收口为负结果，不再在饱和开发集上增加变体。

## 5. 历史文档警告

`ABLATION_RESULTS.md`、`ADAPTIVE_SCHEDULE.md`、`BASELINE_DECISION.md`、`DIAGNOSTIC_ANALYSIS.md`、`EXPERIMENT_PLAN.md` 和旧版 `REPRODUCE.md` 曾记录确定性修复前的当时判断。它们不得覆盖本文和机器可读 evidence。
