# HNEK e200 confirmation

> 由 `summarize_e200_confirmation.py` 自动生成。只做单 seed=2026 的 paired-development 确认，不构成 confirmatory 结论。

| variant | e50 delta | e200 delta | e200 95% CI | e200 positive domains | e200 verdict | change (e200−e50) |
|---|---:|---:|---:|---:|---|---:|
| hnek_coord_y | 3.1481 | -1.2164 | [-1.4174, -1.0153] | 2 | DEVELOPMENT_FAIL_SINGLE_SEED | -4.3645 |
| hnek_g0.25 | 2.6173 | 0.7884 | [0.5916, 0.9933] | 4 | DEVELOPMENT_PASS_SINGLE_SEED | -1.8289 |

## 判定

e200 后仍保持正向且 CI 不含 0、positive domains ≥ 3/5 的变体：hnek_g0.25。

## 下一步建议（本机不做多 seed）

- 若至少一个变体通过上述判定，下一步应在更好的服务器上以同 seed=2026 复现，再补 2~4 个独立 seed；
- 优先验证最强且实现解释最清晰的变体；
- 多 seed 报告使用 mean±std/CI，并继续注明 reflection/adaptive pool 的 limitation 口径。

