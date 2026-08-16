> **历史/已废弃**：本文早期用 nohup，已被 systemd 持久服务取代（`systemd-run --user`）。后台运行现状以 `PLAN.md` 和 `STATUS.json` 为准。

# 后台运行与续跑约定（解决“长任务会停”）

## 机制

长训练用 `nohup ... &` 启动为服务器后台进程，PID 记录到 `refactor/_runs/pids.txt`，日志写到 `refactor/_runs/*.log`。这样训练不依赖 agent 回合是否结束。

## 已修 baseline 坑

- netG/netD/netE choices 补 `_cond`；
- `--mode` 与 `--model` argparse 前缀冲突，加 `allow_abbrev=False`；
- 补 `--seed` 与 train.py 确定性 seed。

## 干净模型注册

- `--model dtcov` → refactor/baseline/models/dtcov_model.py（已端到端跑通）
- `--model hj` → refactor/baseline/models/hj_model.py（集成已写，runtime bug 待修）

## 下一步

1. 修 HJ 集成：真实 netF 特征形状对齐（重点查 sample_ids 与 batch 维度）。
2. 用后台进程跑 DT 完整复现（warmup → dtcov 窗口 → plain 续训 → eval-off）。
3. HJ 修好后跑完整复现 + 消融。
