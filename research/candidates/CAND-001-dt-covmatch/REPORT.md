# DT-CovMatch 重构报告

## 结论

第一阶段的 L0/L1 已通过。核心算法被抽成一个自包含的
`dtcov.dtcovmatch.DTCovMatch` 类，训练循环的接入点压缩成
`dtcov.model.SBModelDTCovMatch` 中的一个子类覆写。没有跑 GPU。

## 交付状态

- `SPEC.md`：完成。
- `TRICK_LEDGER.md`：完成。
- `dtcov/`：完成，见下。
- `tests/test_dtcovmatch.py`：7 个纯 CPU 测试，全部 PASS。
- `REPORT.md`：本文件。

## 保住了什么

- 最优分支的核心机制：按 domain 分组的 MC endpoint 分歧统计、冻结 first-use
  teacher、domain×time EMA z-score 匹配、smooth-L1、grouped-domain 等权聚合。
- 训练协议：warmup、ramp-hold-decay、plain continuation、eval-off plain。
- 原有 RNG 账目语义：辅助 MC 采样不偏移主训练 RNG。
- 目标：仍以 `TARGET_RESULTS.json` 中的 +1.0439 dB 为后续 L2 门槛。

## 删掉了什么（按分类）

- `inertia_legacy`：scheme12/123 low-rank covariance、side-car uncertainty head
  (`netU`)、risk weighting / bridge gating、大量诊断字段和 JSON flush。
- `hack_artifact`：`ua_scheme=12` + `ua_train_rollout=True` 在训练桥构造中的
  “看似启用、实则走 plain netG”的开关假象；改为显式在 `compute_G_loss` 加正则。
- 未用 option 分支：`ua_train_match_norm=none`、domain/time allowlist、
  `u_head`、`u_band`、`u_cap` 等。

## 关键工程简化

原实现把状态挂在 `model` 的 `_ua_*` 属性上，用四个自由函数改字典。现在：

- `DomainTimeStats`：一个类取代 4 个函数，负责 EMA mean/var 与 mu/sigma。
- `DTCovMatch`：一个类持有 `netG`、`teacher`、`stats`、`iter`，把数据通过参数
  传入，没有隐藏全局状态。
- `scheduled_lambda`：把 epoch schedule 单独抽出来，可直接单元测试。
- `SBModelDTCovMatch`：只覆写 `modify_commandline_options`、`__init__`、
  `compute_G_loss`、`optimize_parameters`，不复制整个 baseline。

## 测试覆盖

| 测试 | 验证内容 |
|---|---|
| `test_scheduled_lambda_ramp_hold_cosine_decay` | ramp-hold-decay 在关键 epoch 的数值 |
| `test_scheduled_lambda_fixed_and_cosine` | fixed / cosine 语义 |
| `test_compute_direction_statistics_matches_reference` | MC 分歧公式与手写 reference 一致 |
| `test_domain_time_stats_ema_and_unknown` | EMA 更新、未知 key 的 mu=0/sigma=1 |
| `test_dtcovmatch_eval_off_returns_zero` | `lambda=0` 时严格零 loss 且不建 teacher |
| `test_dtcovmatch_enabled_forward_and_backward` | 启用时标量 loss、diagnostics、backward 到 netG |
| `test_domain_key_and_time_norm` | domain key 与实际时间归一化 |

## L1 等价性边界

- 已对齐：schedule、方向统计、domain×time EMA、eval-off 零路径、teacher 冻结语义。
- 有意简化：移除了诊断落盘和未用分支；核心数值公式保持一致。
- 未做：`SBModelDTCovMatch` 与原始 sb_model.py 的端到端 forward/loss 逐位对齐。
  这需要真实 dataloader/网络/GPU，属于 L2。

## 还缺的证据 / 下一步

1. L2 GPU 干净实现复现：用 `SBModelDTCovMatch` 走
   `fin6srv_b16_dtcov_grouped_ramp5hold15decay25_l001_all6_plain_e200_s2026`
   协议，目标是 >= +1.0439 dB，否则 `blocked` 并回报。
2. 最多 2 个 knock-out：例如 `grouped_domain` 改 batch 统一、teacher 改 self，
   确认 grouped-domain / frozen teacher 是否真的必要。
3. 若 GPU 复现不达门槛，回退只报告代码等价性，不宣称保住了收益。

当前阶段：`L0=PASS, L1=PASS（核心等价）, L2=PENDING`。
