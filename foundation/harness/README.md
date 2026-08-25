# Harness（统一底座）

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 与模型解耦的数据身份、配置 hash、RNG 隔离、checkpoint 审计和配对统计工具。 |
| 当前结论 | CPU 自测 12/12；适合所有新候选复用，但不能替代真实 GPU smoke 或训练 seed 确认。 |
| 时间线位置 | T1 早期实验纪律萌芽，T5–T7 随 deterministic canonical 固化。 |
| 先看哪里 | 先运行 `python foundation/harness/self_test.py`，再按下方模块职责接入新实验。 |

这是 DT/HJ 重构前必须依赖的工程底座。CPU 核心已通过 `self_test.py`（12/12）。自测使用临时元数据 fixture，不依赖旧 `/home/yc` 数据或 checkpoint。

## 模块

- `data.py`：manifest 身份/切分/零重叠审计，不读像素。
- `config.py`：版本化冻结配置 + 规范 hash。
- `determinism.py`：seed、辅助 RNG 隔离、无碰撞子 seed。
- `checkpoint.py`：net 四件套 + training_state 的完整状态审计。
- `metrics.py`：逐图配对对齐 + bootstrap CI。

## 使用约定（算法重构必须遵守）

- 数据身份一律用 `data.zero_overlap` / `data.audit_manifest` 校验。
- 每个实验的完整有效配置用 `config.freeze_config` 记录并保存 hash。
- 辅助随机（probe/MC/对照）必须用 `determinism.rng_scope` 包裹，不污染主训练 RNG。
- 子流 seed 用 `determinism.sub_seed`，避免历史那种 1280 次碰撞。
- checkpoint 用 `checkpoint.audit_checkpoint` 核验完整性，不只看目录名。
- 比较用 `metrics.paired_bootstrap`，逐图配对，不拿汇总均值硬拼。

## 自测

```bash
python foundation/harness/self_test.py
```

该命令只验证 manifest/checkpoint 元数据契约、RNG 隔离和 bootstrap，不代替真实数据身份审计或 GPU 训练 smoke。
