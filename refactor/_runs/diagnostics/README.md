# 只读诊断日志约定

诊断钩子不改变训练语义（loss、RNG、参数更新、forward 都不变），只额外记录介入机制证据。

## DT

- flag：`--dtcov_diag_out <path>`
- 每 epoch 落一行 JSONL：`epoch, drift(=teacher-student mismatch EMA), plateau, lambda_value, sb_entropy_grad_norm`。

## HJ

- flag：`--hj_diag_out <path>`
- 每 epoch 落一行 JSONL：`epoch, gate_hit_rate, risk_mean, probe_agreement, risk_positive, sb_entropy_grad_norm`。

## 约定

- 所有诊断输出统一放在本目录，文件名用 `dtcov_<run>.jsonl` / `hj_<run>.jsonl`。
- 未设置 `*_diag_out` 时零开销，不记录、不做额外 backward。

## 机制证据（介入有效发生点）

- DT（dtcov_adaptive.jsonl，25 epoch 窗口）：drift 从 warmup 后 epoch 6 开始 >0，到窗口末增至 0.519，仍在收敛；SB 熵项梯度范数全程波动。
- HJ（hj_adaptive.jsonl，200 epoch）：gate 命中率自 start_epoch 后 epoch 6 开始 >0，稳定在约 0.12（约 12% 位置被投影），conflict risk 约 0.12。

即两个介入都在 epoch 6 附近“生效”，诊断量把这一机制事实记录了下来。

程序化核对：`python refactor/_runs/analyze_diagnostics.py`（读上面两个 JSONL，输出
首次激活 epoch 与末段均值，便于审计“epoch 6 生效”这一结论）。
