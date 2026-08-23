# 当前进度（当前权威，2026-08-24）

## 一句话状态

**阶段1确定性修复、阶段2单 seed=2026 干净核心、阶段3 HNEK 桥原生搜索与 e200 确认均已全部完成。** e200 确认后唯一存活变体是 `hnek_g0.25`（macro PSNR delta +0.7884 dB，CI [+0.5916,+0.9933]，4/5 域为正，SSIM +0.0355）；`hnek_coord_y` 翻负。所有结论均为 single seed=2026 paired-development、**非 confirmatory**；下一步只剩迁移到更好服务器补 2~4 个独立 seed 复现。

## 已完成

- 干净 DT vs 干净 plain：PSNR +0.8875 dB（CI95 [0.4992,1.2923]）、LPIPS -0.0557。
- HJ 集成 bug 修复（netG encode_only 单 layer 返回结构错误），GPU smoke 通过。
- 数理 grounding 成文：refactor/METHOD_GROUNDING.md。
- 实验计划成文：refactor/EXPERIMENT_PLAN.md（knock-out + adaptive schedule）。
- 基准口径落盘：refactor/BASELINE_DECISION.md。
- HJ 归因结果（单 seed=2026）：true vs plain +2.7533、true vs roll +2.7612，四项 PASS，roll≈plain。**但 3-seed 试点显示该归因未跨 seed 复现**（true−roll 3-seed CI 含 0），见 `HJ_MULTISEED_PILOT.md`。
- 数理 grounding 成文：refactor/METHOD_GROUNDING.md（DT=有界函数正则/信任域一致性；HJ=前向不变结构有害方向梯度修正 + 可验证 gate）。
- DT 自适应介入 schedule 已实现：`--dtcov_lambda_schedule adaptive`（EMA plateau 检测 + ramp→hold→plateau 退出），文档 refactor/ADAPTIVE_SCHEDULE.md。
- 只读诊断日志钩子已加（不改变训练语义）：DT drift / HJ gate 命中率、conflict risk、probe agreement / SB 熵项梯度范数；输出约定见 refactor/_runs/diagnostics/README.md。
- HJ knock-out 4/4 完成：central_consensus / boundary / min_risk / strength 全部必要（见 refactor/_runs/ABLATION_RESULTS.md）。
- DT knock-out 2/5 完成：grouped_domain（+0.3492）、schedule 固定 λ（+0.7377）均低于基线，必要。
- 统一叙事 + novelty 边界成文：refactor/METHOD_NARRATIVE.md。
- DT adaptive λ 对比手调已启动（`unsb_train_adaptive.service`，带诊断日志）；HJ adaptive α 已实现（conflict EMA/peak 驱动）。
- DT adaptive 结果：adaptive PSNR 18.8911（+0.9332）≥ 手调 18.8453（+0.8875），满足 adaptive ≥ 手调；诊断日志记录 drift/plateau/SB 梯度范数。
- HJ adaptive 结果：adaptive（+2.1190）低于手调 strength 0.5（+2.7533），结论是 HJ 的全局 strength 应固定、局部性由 gate 保证；DT 自适应有益、HJ 自适应有害，作为分析点写进 refactor/_runs/HJ_ADAPTIVE_RESULT.md。
- 监控恢复：`monitor_once.sh` 曾丢失可执行位（664），导致 03:13 后 STATUS/HEARTBEAT 停更；已恢复 +x 并手动触发一次，STATUS 现为 stage=HJ_DONE / service=inactive，心跳恢复正常。
- 新增 DT 消融开关（默认语义不变）：`--dtcov_teacher frozen|self`、`--dtcov_norm_mode domain_time|global`、`--dtcov_signal_norm on|off`；单测 10/10 pass，GPU smoke 验证三个路径无崩溃。
- 全量 CPU 回归通过：harness 12/12、DT 10/10、HJ 9/9（新增消融开关后无回归）；诊断 JSONL 已确认真实非空（DT drift 0→0.52、HJ gate 命中率≈0.097）。
- 复现总纲：`refactor/_runs/REPRODUCE.md`（完整顺序 + 多 seed 防覆盖注意）；自动收尾：`unsb_finalize.timer` 每 15 分钟检测 `dt_abl_extra.done`，成功后自动生成 `DT_ABL_EXTRA_RESULT.md`。
- determinism 修复：`train.py`/`test.py` 增加 `cudnn.deterministic=True`、`cudnn.benchmark=False`、关 TF32；`base_model.py` 的 `benchmark=True` 加了 `not deterministic` 保护；`REPRODUCE.md` 写清确定性边界。
- 3-seed DT 试点完成：配对 delta +0.8875 / +0.7426 / +0.4687，mean +0.6996，95% CI [0.1712,1.2280]（不含 0）。详见 `refactor/_runs/DT_MULTISEED_PILOT.md`。
- v1.0.0 冻结：git commit `0ca9075` + tag `v1.0.0`，`VERSION.md`、`MANIFEST.sha256`（198 项）已生成并自校验；排除 checkpoints/生成图片/日志/legacy 目录。
- 收口文档：`refactor/_runs/FINAL_STATUS.md`（已确认收益/机制现状/动机/limitation）、`refactor/_runs/DIAGNOSTIC_ANALYSIS.md`（drift=发散信号、SB 梯度=噪声、HJ gate 指标随 α 单调但与 PSNR 无线性关系）。
- 桥原生重评估与对齐：`refactor/_runs/BRIDGE_NATIVE_REASSESSMENT.md`、`HNEK_ALIGNMENT_CHECK.md`、`HNEK_RUN_PLAN.md`；HNEK shim 已落地（`refactor/baseline/models/hnek/` + `--hnek` 标志）。
- **随机性修复（阶段1，CPU 完成）**：新增 `refactor/baseline/models/det_pad.py`（确定性反射 pad，slicing+cat），`networks.py`/`ncsn_networks.py` 全部 `nn.ReflectionPad2d` 已替换；CPU 单测 4/4（forward 与 `F.pad(mode='reflect')` 逐位一致、backward 正确且可复现）；回归 harness 12/12、DT 10/10、HJ 9/9。
- **HNEK frozen e50 verdict**：`STOP_E50_CLEAR_FAIL`，macro PSNR delta −0.7438 dB（CI [−1.0567,−0.4356]），positive domains 1/5；PLAIN 17.9731 vs METHOD 17.2293。作为阶段 3 桥原生搜索的负参照。
- **阶段1 GPU 逐位复现通过**：`refactor/baseline/models/det_pad.py` 替换反射 pad 后，进一步定位并修复 CuBLAS `torch.bmm` 非确定；`train.py`/`test.py` 在 import torch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并在 `--seed>=0` 时启用 `torch.use_deterministic_algorithms(True)`。同 seed 2026 两次 3-epoch smoke 的 `smoke_det_e/f` `3_net_G.pth` SHA256 相同，loss 行仅耗时秒数差异。
- **阶段2 启动**：新增 `refactor/_runs/run_core_clean_deterministic.sh`，每个 seed 先练 `clean_warmup_s$s`（e20，num_threads=0），再串行练/评 DT plain、DT best、HJ plain、HJ true、HJ roll；全程单 GPU、`num_threads=0`、方法臂开诊断。运行单元：`unsb_clean_core.service`，主日志 `refactor/_runs/metrics_clean_core/core_clean_run.log`。
- **阶段2 中断与恢复**：`clean_dt_best_s2026` 在 DT 正则器 `F.adaptive_avg_pool2d` 的 CUDA backward 上触发 `deterministic_algorithms(True)` RuntimeError。已将 DT 的 adaptive pooling 替换为 `_deterministic_adaptive_avg_pool2d`（整除时等价 `avg_pool2d`，否则切片+mean 回退；CPU/GPU forward 与 `F.adaptive_avg_pool2d` 一致、backward 确定），DT 单测 10/10 通过。`run_core_clean_deterministic.sh` 增加 `CORE_RESUME_SEED`/`CORE_RESUME_STAGE` 断点续跑能力，已从 seed 2026 `dt_best` 恢复；失败日志保存为 `clean_dt_best_s2026_train_failed_adaptive_pool.log`。
- **阶段3 搜索脚手架**：新增 `STAGE3_SEARCH_PLAN.md`、`models/hnek/hnek_search.py`、`models/hnek_search_model.py`、`run_hnek_search_variant.py`、`run_stage3_search_queue.sh`，并启动 `unsb_stage3_watch.service`；阶段2完成后会自动等待 GPU 空闲并串行运行 HNEK 桥原生变体（γ / 熵坐标 / 熵权重 / partial），每个 e50、single seed、写 trace/eval/adjudication。
- **核心结果汇总器**：新增 `summarize_clean_core.py`，阶段2完成后先生成 `metrics_clean_core/CLEAN_CORE_RESULTS.md/json`（3-seed mean±CI），再进入阶段3搜索。
- **自动收口链路**：新增 `finalize_report.py` 与 `run_final_report_watch.sh`；`unsb_final_report_watch.service` 会在阶段3搜索结束并写 `hnek_search/stage3_queue.done` 后自动生成 `refactor/_runs/FINAL_REPORT.md` 和 `FINAL_REPORT_STATUS.json`。
- **停止 3-seed，只保留 seed=2026**：已优雅停止 `unsb_clean_core.service`，丢弃 seed 2027 半成品（checkpoints 与 train log 已删除，seed 2026 clean 结果保留）。`summarize_clean_core.py` 改为单 seed=2026 口径，生成 `metrics_clean_core/CLEAN_CORE_RESULTS.md/json`，明确不写 3-seed mean±CI、不把单点写成稳定性结论。已写 `core_clean.done`（单 seed 完成口径）触发阶段3 watcher。

- **阶段3 e200 确认完成**：`hnek_coord_y` e200 翻负（macro PSNR delta −1.2164 dB，CI [−1.4174,−1.0153]，positive domains 2/5，verdict `DEVELOPMENT_FAIL_SINGLE_SEED`）；`hnek_g0.25` e200 通过（+0.7884 dB，CI [+0.5916,+0.9933]，positive domains 4/5，SSIM delta +0.0355，verdict `DEVELOPMENT_PASS_SINGLE_SEED`）。汇总见 `refactor/_runs/hnek_search/E200_CONFIRMATION.md/json`。
- **阶段3 全部完成**：9 个 HNEK 变体 e50 搜索 + 2 个 e200 确认均已落盘；`hnek_g0.25` 是唯一 e200 存活变体。

## 进行中

- （无进行中的训练/评估任务；阶段3 与 e200 确认均已结束。）

## 下一步

1. 本机不再跑训练/评估。将 `hnek_g0.25` 作为唯一候选迁移到更好服务器：先用同 seed=2026 复现，再补 2~4 个独立 seed，报告 mean±CI。
2. 围绕 `hnek_g0.25`（γ=0.25、residual 坐标、physical horizon、全量应用）做最小深挖与论文落点；保留“single seed paired-development、非 confirmatory”边界。

## 阻塞 / 风险

- **HJ 3-seed 结论（已出）**：true−plain mean +2.07（95% CI [−1.64,+5.78] 含 0）；true−roll mean +1.49（CI [−1.37,+4.35] 含 0）；roll−plain 在 3 seed 里 ≈0 / +2.58 / −0.82，极不稳定。→ HJ 的收益与“结构方向特异”归因**在 n=3 下不能成立**，必须扩到 ≥5 seed；当前 HJ 结果只能作方向参考。
- **已解决**：原 `reflection_pad2d` backward 非确定问题已通过手工确定性 pad 消除；另发现 PatchNCE 的 `torch.bmm`（CuBLAS）在未设置 `CUBLAS_WORKSPACE_CONFIG` 且未启用 deterministic-algorithms 时仍会带来运行间差异，已在 `train.py`/`test.py` 统一修复并逐位 smoke 验证。
- **后果**：旧的单 seed=2026 DT/HJ 消融与 adaptive 数字仍按既有 mean±CI / direction-only 口径解读；新一轮 `metrics_clean_core` 才是确定性口径下的干净基准。
- **结论（2026-08-19 更新）**：投稿级结论先以确定性 `metrics_clean_core` 的单 seed=2026 数值为准，但不作稳定性声明；跨 seed 稳健性留到更好的服务器。当前单 seed 干净数值已汇总到 `CLEAN_CORE_RESULTS.md/json`。
