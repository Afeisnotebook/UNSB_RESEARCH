# DT-CovMatch 干净版完整复现结果（2026-08-14）

## 训练

- 完成：warmup 20e → DT 窗口 20e（ramp5hold15decay25, λ=0.001）→ plain 续训到 200e。
- 状态：200/200 完成，checkpoint `checkpoints/dtcov_clean_best_e200/200_net_G.pth`。

## 评测（test40，240 张，eval-off）

- overall PSNR = 17.9555
- overall SSIM = 0.5893
- overall LPIPS = 0.3064
- overall NIQE = 7.357

## 对照

- 原 plain 基线 = 18.7360
- 原 DT 最好 = 19.7800（+1.0439）

## 结论

未复现。干净版 DT 得到 17.96，既没到 +1.0439，也低于原 plain。需要查根因：

1. 干净 baseline（原始 UNSB）是否与原 modified plain 一致；
2. DT schedule/正则接线是否正确；
3. 先训一个干净 plain 基线做公平对照。
