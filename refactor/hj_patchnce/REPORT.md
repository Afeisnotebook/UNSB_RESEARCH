# HJ-PatchNCE 重构报告

## 结论

L0/L1 已过（9 个纯 CPU 测试 PASS），未上 GPU。

## 交付

- `SPEC.md`：完成。
- `TRICK_LEDGER.md`：完成。
- `hj/`：核心实现（projection / structure / core）。
- `tests/test_hj.py`：9 个 CPU 测试，全部 PASS。

## 保住了什么

- 核心机制：结构切方向 → 反事实探针 → delta/risk → gate → forward 恒等 backward 投影。
- 最好分支配置：continuous layer0、joint 方向、central_consensus、strength 0.5、gate_quantile 0.75、min_risk 0.05、boundary_scale 0.001、update_mode remove。
- eval-off 不变量：关闭干预时 projected loss == raw loss。

## 删掉了什么

- 大量未用模式（bridge_*、structure_risk/relational/pixel_project、corr_*、curation、side-car netU）。
- `_structure_event_weight` 的事件状态（最好配置 z=0，恒为 1）。
- 诊断累积/落盘从核心移出。

## 还缺的证据 / 下一步

1. L2 复现锚点：用冻结 `true_constant@200` / `plain@200` / `roll_constant@200` 在 val-O 推理，复现 +1.4729 和归因 SSIM -0.0202。
2. 干净实现训练复现 + knock-out（boundary_scale / min_risk / central_consensus）。

状态：`L0=PASS, L1=PASS（核心等价）, L2=PENDING`。
