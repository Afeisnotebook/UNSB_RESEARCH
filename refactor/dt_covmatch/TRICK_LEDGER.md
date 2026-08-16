# DT-CovMatch trick 分类账

分类法见 `../CONTRACT.md`：

- `necessary_algorithmic`：删掉就不再是 DT-CovMatch。
- `necessary_robustness`：数值稳定、收敛或确定性所必需。
- `inertia_legacy`：历史惯性，可删。
- `hack_artifact`：绕 bug 或历史事故，可删或改写。
- `unknown`：当前无法判定，需消融。

## 账本

| 项目 | 原代码位置 | 作用 | 分类 | 理由 / 建议 |
|---|---|---|---|---|
| MC endpoint proposals `ua_samples=4` | `sample_endpoints` | 估计随机 proposal 分歧所需的样本数 | necessary_algorithmic | 保留，参数改为 `dtcov_m` |
| 方向统计 `(Y-X_t)/(1-t)` | `compute_direction_statistics` | 把 endpoint 分歧变成桥方向分歧 | necessary_algorithmic | 保留 |
| 信号归一化 `U_reg_norm = U_reg/(signal+eps)` | `compute_direction_statistics` | 避免亮区域天然主导 U，尺度可比 | necessary_algorithmic | 保留 |
| region pooling `ua_region_patch=32` | `_region_grid` + adaptive pool | 降低空间自由度，稳定匹配 | necessary_robustness | 保留，改名 `dtcov_region_patch` |
| frozen first-use teacher | `ensure_frozen_teacher` | 阻断“当前模型既产 U 又被自己 U 推”的闭环 | necessary_algorithmic | 保留，改成 `DTCovMatch.teacher` |
| teacher no_grad/detach | `with torch.no_grad()` | teacher 只作目标，不回传 | necessary_robustness | 保留 |
| current `detach_uncertainty=False` | `covariance_match_regularizer_grouped_domain` | 让损失能反传到当前 generator | necessary_algorithmic | 保留 |
| log-U floor `ua_train_u_floor=1e-8` | `clamp_min(floor)` | log 数值下限 | necessary_robustness | 保留 |
| domain×time EMA `ua_train_match_norm=domain_time_ema` | `_dtcov_*` | 按 (domain,time) 归一化 teacher log-U 分布 | necessary_algorithmic | 保留 |
| EMA momentum `0.98` | `_dtcov_update_stats` | 统计稳定与跟踪平衡 | necessary_robustness | 保留 |
| sigma/eps `1e-4` | `ua_train_match_norm_eps` | 方差下限，防止除零 | necessary_robustness | 保留 |
| z-score clip `3.0` | `ua_train_match_norm_clip` | 防止离群 log-U 拉爆 loss | necessary_robustness | 保留 |
| grouped-domain aggregation | `covariance_match_regularizer_grouped_domain` | 混合 batch 下按 domain 等权聚合 | necessary_algorithmic | 保留 |
| lambda `ua_train_reg_lambda=0.001` | `add_uncertainty_losses` | 正则强度 | necessary_algorithmic | 保留，改名 `dtcov_lambda` |
| schedule `ramp_hold_cosine_decay` | `train.py::scheduled_ua_train_reg_lambda` | 控制 DT 只在中段启用并平稳退出 | necessary_algorithmic | 保留，函数名 `scheduled_lambda` |
| warmup 300 iters | `_train_uncertainty_enabled` | 等 teacher 统计有一点积累后再开 loss | necessary_robustness | 保留，改名 `dtcov_warmup_iters` |
| `ua_scheme=12` 但训练期不启用 scheme12 | `maybe_rollout_endpoint` | 最优分支训练 bridge 构造仍走 plain netG；scheme12 只在测试路径有意义 | inertia_legacy | 删除；DT 只保留 u_match loss |
| `ua_train_rollout=True` | `maybe_rollout_endpoint` | 训练桥构造标志，实际 train_phase 分支直接返回 netG，等于 plain | hack_artifact | 删除，改为在 compute_G_loss 显式加正则 |
| `ua_train_match_domain_balance` choices 含 `grouped_domain` | `sb_model.py` | 旧版/新版切换痕迹，最优分支走 grouped_domain | inertia_legacy | 保留一个明确参数 `dtcov_domain_balance`，不再保留未用分支 |
| `ua_train_match_norm=none` 兜底 | `covariance_match_regularizer` | 非 domain_time_ema 的 legacy absolute log-U 匹配 | inertia_legacy | 删除，DT 只支持 domain_time_ema |
| `ua_train_domain_allowlist` / `ua_train_time_allowlist` | `train_aux_domain_mask` / `train_aux_time_mask` | 最优分支 allowlist 为 all6/空，等于不做筛选 | hack_artifact | 删除；若将来需要域过滤，用显式 domain list 参数 |
| `ua_train_uhead_*` / `netU` | `u_head_teacher_loss` 等 | side-car uncertainty head 诊断 | inertia_legacy | 删除，与最优分支无关 |
| scheme12/123 low-rank covariance | `compute_scheme12/123`, `_solve_low_rank_shrink` | 早期 test-time rollout 方案 | inertia_legacy | 删除，与 DT 训练正则无关 |
| bridge MSE weighting / risk gating | `bridge_mse_loss`, `bridge_corr_gate` 等 | 训练期 MSE 加权或 gate | inertia_legacy | 删除 |
| `loss_UA_*` 大量诊断字段 | `sb_model.__init__` | 历史显示字段 | inertia_legacy | 删除，只保留 `U_match` |
| `record_diagnostics` / JSON flush | `record_diagnostics`, `flush_diagnostics` | 逐图诊断落盘 | hack_artifact | 核心实现删除，诊断可单独做 |
| `SimpleUncertaintyHead` | `SimpleUncertaintyHead` | side-car head | inertia_legacy | 删除 |
| `preserve_torch_rng` 包裹 MC 采样 | `preserve_torch_rng` | 防止辅助采样偏移主训练 RNG | necessary_robustness | 保留，重构为 `_preserve_rng_state` |
| `train/test` 多套 option 打印 | `sb_model.modify_commandline_options` | 原实现的命令行膨胀 | inertia_legacy | 只保留 DT 需要的参数，前缀 `dtcov_` |

## 最关键的结论

1. “grouped-domain” 是这个最优分支真正区别于同批 ``homog``/``eqdom`` 的地方；
   它必须按 domain 分组采样与聚合，不能简化成 batch 内统一 EMA。
2. ``ua_scheme=12``、``ua_train_rollout=True`` 在训练桥构造里实际不改变
   forward，是历史遗留的“开关开着但没作用”的假象；重构后把这个假象删掉。
3. 大量 ``ua_*`` 字段与 side-car head、risk gate 均不参与最优分支，属于惯性代码。
