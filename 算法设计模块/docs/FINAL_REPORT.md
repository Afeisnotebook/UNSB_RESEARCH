# Final evidence report（自动收口）

> 本文件是证据收口，不是投稿终稿。所有数字来自 harness 证据文件；全部为 single seed=2026 paired-development，不构成 confirmatory 结论。

## 1. 确定性

- 证据文件：`DETERMINISM_FIX.md`
- 结论：同 seed 两次 3-epoch smoke 的 `3_net_G.pth` SHA256 一致。

## 2. 核心干净重跑（单 seed=2026）

| seed | DT delta | HJ true−plain | HJ roll−plain | HJ true−roll |
|---|---:|---:|---:|---:|
| 2026 | -0.2677 | +0.0381 | -0.7521 | +0.7901 |
- DT delta: -0.2677 dB (single seed, no CI)
- HJ true−plain: +0.0381 dB (single seed, no CI)
- HJ roll−plain: -0.7521 dB (single seed, no CI)
- HJ true−roll: +0.7901 dB (single seed, no CI)

## 3. HNEK frozen 负参照

- macro PSNR delta: -0.7438 dB
- positive domains: 1

## 4. 阶段3桥原生搜索（e50）

| variant | delta_db | ci_low | ci_high | positive_domains | verdict |
|---|---:|---:|---:|---:|---|
| hnek_g0.5_ref | -1.232843 | -1.499757 | -0.975635 | 0 | STOP_E50_CLEAR_FAIL |
| hnek_g0.25 | 2.617316 | 2.351880 | 2.885277 | 4 | PROCEED_DIRECT_TO_E200 |
| hnek_g0.75 | -1.313877 | -1.503621 | -1.121398 | 0 | STOP_E50_CLEAR_FAIL |
| hnek_g1.0 | 0.785955 | 0.538221 | 1.035796 | 4 | PROCEED_DIRECT_TO_E200 |
| hnek_coord_y | 3.148084 | 2.852841 | 3.441472 | 4 | PROCEED_DIRECT_TO_E200 |
| hnek_horizon_index | -2.582525 | -2.954044 | -2.208901 | 1 | STOP_E50_CLEAR_FAIL |
| hnek_horizon_mix | 0.786130 | 0.516163 | 1.059355 | 4 | PROCEED_DIRECT_TO_E200 |
| hnek_entropy_only | 0.967119 | 0.741349 | 1.202221 | 5 | PROCEED_DIRECT_TO_E200 |
| hnek_endpoint_only | 1.348399 | 1.135305 | 1.562566 | 4 | PROCEED_DIRECT_TO_E200 |

## 5. 阶段3 e200 确认（最终）

| variant | e50 delta | e200 delta | e200 95% CI | positive domains | e200 verdict |
|---|---:|---:|---:|---:|---|
| hnek_coord_y | +3.1481 | -1.2164 | [-1.4174, -1.0153] | 2 | DEVELOPMENT_FAIL_SINGLE_SEED |
| hnek_g0.25 | +2.6173 | +0.7884 | [0.5916, 0.9933] | 4 | DEVELOPMENT_PASS_SINGLE_SEED |

- `hnek_g0.25` 是唯一在 e200 后仍保持正向、95% CI 不含 0、positive domains ≥3/5 的变体；SSIM delta +0.0355。
- `hnek_coord_y` 在 e200 翻负，判定 `DEVELOPMENT_FAIL_SINGLE_SEED`。
- 汇总：`refactor/_runs/hnek_search/E200_CONFIRMATION.md/json`。

## 6. 收口判定

- 阶段1 确定性：完成
- 阶段2 干净核心（单 seed=2026）：完成
- 阶段3 搜索（e50）+ e200 确认：完成
- 唯一存活候选：`hnek_g0.25`
- 状态：E200_CONFIRMED（single seed=2026 paired-development，非 confirmatory）

## 7. 投稿前仍需人工确认

- 桥原生 novelty 边界是否仍成立；
- `hnek_g0.25` 正结果是否能在更好服务器上跨 seed 复现；
- DT / HJ 干净核心与 `hnek_coord_y` 的负结果是否诚实降级；
- limitation 与 mean±CI 口径是否统一。
