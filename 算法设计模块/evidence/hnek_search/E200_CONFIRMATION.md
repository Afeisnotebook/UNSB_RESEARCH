# HNEK e200 开发延伸（历史文件名保留）

> 由历史 `summarize_e200_confirmation.py` 自动生成。这里只做 single seed=2026 的 paired-development 延伸，不构成 confirmatory 结论；文件名中的 confirmation 是阶段名。
>
> 表中 CI 是固定训练 seed 和开发集条件下的配对样本 bootstrap，不包含训练 seed 之间的不确定性；9 变体搜索与开发集反复使用造成的选择偏差也不在 CI 内。

| variant | e50 delta | e200 delta | e200 95% CI | e200 positive domains | e200 verdict | change (e200−e50) |
|---|---:|---:|---:|---:|---|---:|
| hnek_coord_y | 3.1481 | -1.2164 | [-1.4174, -1.0153] | 2 | DEVELOPMENT_FAIL_SINGLE_SEED | -4.3645 |
| hnek_g0.25 | 2.6173 | 0.7884 | [0.5916, 0.9933] | 4 | DEVELOPMENT_PASS_SINGLE_SEED | -1.8289 |

## 判定

e200 后仍保持正向且 CI 不含 0、positive domains ≥ 3/5 的变体：hnek_g0.25。

## 下一步建议（本机不做多 seed）

- 若至少一个变体通过上述判定，下一步应在更好的服务器上以同 seed=2026 复现，再补 2~4 个独立 seed；
- 只验证冻结的 `hnek_g0.25`，不再按本开发集新增变体；
- 多 seed 报告以 seed 为统计单位使用 mean±std/CI；图像级 bootstrap 只作 seed 内辅助；
- 使用未参与搜索的确认集做一次性最终评估。

