# 复现指南（单 seed=2026，干净框架）

本文给出从零复现 DT-CovMatch / HJ-PatchNCE 全部实验的确切顺序。所有脚本都在
`refactor/_runs/` 下，统一使用 `refactor/baseline/train.py` / `test.py` 和
`refactor/harness/`。收益基准以干净 plain 为参照，不追旧 modified 绝对数。

## 环境与路径

- Python：`/home/yc/anaconda3/envs/unsb_cov/bin/python`（torch 2.5.1+cu121）。
- GPU：单卡 RTX 4090，`CUDA_VISIBLE_DEVICES=0`。
- 训练数据（DT）：`/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired`。
- 训练数据（HJ）：`/home/yc/UNSB_Patch/datasets/final6_train160_test40_unpaired`。
- val-O 数据（HJ 归因/评测）：`/home/yc/UNSB_Patch/datasets/final6train_valO5x16_offset560_unpaired`。
- checkpoint：`refactor/_runs/checkpoints/`。
- 指标：`refactor/_runs/metrics*/`，每分支写 `metrics_summary.json`。
- 评测器：`/home/yc/UNSB_Cov5/tools/evaluate_restoration.py`。

## 顺序

1. **warmup（e20）**：`run_dtcov_repro.sh` 的 stage 1 会复用已存在的
   `checkpoints/dtcov_clean_warmup_e20`；如需重跑，先确保该目录不存在。

2. **干净 plain 基线**：`bash run_plain_baseline.sh`（从 warmup 续训到 e200）。

3. **干净 DT（best）**：`bash run_dtcov_repro.sh`（25-epoch ramp5/hold15/decay25
   窗口 + 续训 plain 到 e200）。

4. **DT 评测**：`bash run_eval_both.sh`，再
   `python summarize_dt_comparison.py`，得到 +0.8875 dB 的带 CI 相对收益。

5. **HJ true/roll/plain**：`bash run_hj_experiments.sh`
   （layer0 continuous HJ true 与 roll，然后 val-O 评测 plain/true/roll），
   生成归因证据与 `HJ_ATTRIBUTION_RESULT.json`。

6. **knock-out 消融**：
   - DT A1/A4：`bash run_ablations.sh`（前两个 DT 项）。
   - HJ A1-A4：`bash run_ablations.sh`（后四个 HJ 项）。
   - DT A2/A3/A5：`bash run_dt_abl_extra.sh`（frozen→self、domain×time→global、
     signal norm→raw U）。

7. **自适应 schedule**：
   - DT adaptive：`bash run_dt_adaptive.sh`（含诊断 `--dtcov_diag_out`）。
   - HJ adaptive：`bash run_hj_adaptive.sh`。
   - HJ per-location risk amplitude：`bash run_hj_risk_amplitude.sh`。

8. **诊断日志**：上述 adaptive 脚本会把只读诊断写到
   `refactor/_runs/diagnostics/*.jsonl`（DT drift / HJ gate / SB 熵梯度范数）。

## 多 seed 复现注意

脚本把 `--seed` 写死为 `2026`，且 `--name` / `--checkpoints_dir` 是固定目录。
做多 seed 时请对每个 seed 使用独立的 `--name`（如 `dtcov_clean_best_e200_sX`）
和独立 `--checkpoints_dir`，否则会复用/覆盖同一 checkpoint。评测时保持相同的
`--num_test`（DT=240 test40，HJ=80 val-O）与统一 harness。

每个 seed 的最小闭环（以 DT 为例，`S` 为 seed）：

```bash
python train.py ... --name dtcov_warmup_s$S --model sb --n_epochs 20 --seed $S
python train.py ... --name dtcov_plain_s$S --model sb --n_epochs 200 \
  --continue_train --pretrained_name dtcov_warmup_s$S --seed $S
python train.py ... --name dtcov_best_s$S --model dtcov --n_epochs 25 \
  --continue_train --pretrained_name dtcov_warmup_s$S --dtcov_lambda 0.001 \
  --dtcov_lambda_schedule ramp_hold_cosine_decay --seed $S
# 继续 plain 到 200 并 eval，读取 overall psnr 得到 delta_S
```

收集各 seed 的 PSNR delta 后：

```bash
python aggregate_multiseed.py 0.8875 0.74 1.05 ...
# 输出 n / mean / std / sem / ci95
```

## 确定性边界（重要）

- `train.py` / `test.py` 已设置 `cudnn.deterministic=True`、`cudnn.benchmark=False`、
  `cudnn.allow_tf32=False`、`cuda.matmul.allow_tf32=False`，这是能做到的最好确定性。
- 但仍存在不可约的运行间方差：生成器使用的 `reflection_pad2d` 在 PyTorch 中
  **没有确定性 backward 实现**（`reflection_pad2d_backward_cuda does not have a
  deterministic implementation`），因此同一 seed 两次训练也不会逐位一致。
- 结论：单 seed 数值存在约 1 dB 级别的运行间方差，**不能用单 seed 下结论**；
  投稿级结论必须多 seed 统计（均值 + CI）。本目录单 seed 结果仅作方向参考。
- 为尽量降低方差，多 seed 复现建议加 `--num_threads 0`（单进程数据加载）。

## 结果汇总入口

- DT 补充消融一键汇总：`python summarize_dt_abl_extra.py`（输出
  `DT_ABL_EXTRA_RESULT.md`）。
- 跨 seed 聚合：`python aggregate_multiseed.py delta1 delta2 ...`。
- 各结论表：`ABLATION_RESULTS.md`、`ADAPTIVE_RESULT.md`、
  `HJ_ADAPTIVE_RESULT.md`、`HJ_ATTRIBUTION_RESULT.md`。
