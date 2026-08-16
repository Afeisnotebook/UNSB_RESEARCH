# 当前进度（当前权威，2026-08-15）

## 一句话状态

**多 seed 已停止**（DT/HJ 各 3 seed，方向性为正已足够当前阶段证据）。现转向介入优化：先分析诊断 JSONL 找切入点，再按确认后的 1~3 个假设做最小单变量实验；reflection_pad2d 非确定性不修，数字按 mean±CI 报告并写 limitation。

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

## 进行中

- 无训练在跑。正在分析 `diagnostics/*.jsonl` 并整理介入优化假设，等用户确认后再跑。

## 下一步

1. 确认优化假设后：DT 先试“延长窗口 / 修正 plateau 退出”，HJ 只调 gate 阈值（strength 固定 0.5）。
2. 每个优化只改一个变量，统一 harness，结果写 mean±CI（单 seed 的 paired bootstrap）。

## 阻塞 / 风险

- **HJ 3-seed 结论（已出）**：true−plain mean +2.07（95% CI [−1.64,+5.78] 含 0）；true−roll mean +1.49（CI [−1.37,+4.35] 含 0）；roll−plain 在 3 seed 里 ≈0 / +2.58 / −0.82，极不稳定。→ HJ 的收益与“结构方向特异”归因**在 n=3 下不能成立**，必须扩到 ≥5 seed；当前 HJ 结果只能作方向参考。
- **已定位**：生成器 `reflection_pad2d` 的 backward 在 PyTorch 中无确定性实现；即使 `cudnn.deterministic=True` + 关 TF32 + `num_threads=0`，同一 seed 两次训练也不逐位一致（冒烟实测 loss 仍有差异）。
- **后果**：单 seed=2026 的 DT +0.8875 与 DT 消融（+0.35~+1.07 dB）都在约 1 dB 噪声量级内，**单 seed 不能证明 DT trick 必要**；HJ +2.75 dB 效应更大、更稳健，但同样需多 seed 确认。
- **结论**：当前单 seed 结果统一标记为“方向参考”，不作为投稿级结论；多 seed 复现是用户侧在更好服务器上的既定步骤，`REPRODUCE.md` 已给出确定性边界与防覆盖约定。
