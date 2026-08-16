# 决策与漂移账本

## 2026-08-16（续）

- 决策：更好服务器不可得，改在本机 4090 上跑多 seed。分三阶段：Tier1 核心 5-seed（DT plain/best + HJ plain/true/roll）、Tier2 各 trick 5-seed 消融、Tier3 adaptive 5-seed，总预估约 105–110 小时。已启动 `unsb_core5seed.service`（先 DT 2029/2030，再 HJ 2029/2030）。

## 2026-08-16

- 叙事收紧：`METHOD_NARRATIVE.md` 把“证明必要性”改为“检验必要性”，并新增“经验证据状态（诚实）”，明确 DT 3-seed 方向稳健、HJ 3-seed 归因未复现、均需 ≥5 seed 后才能写投稿结论。
- **HJ 3-seed 试点完成**：true−plain +2.75/+3.10/+0.36（mean +2.07，CI [−1.64,+5.78] 含 0）；true−roll +2.76/+0.52/+1.18（mean +1.49，CI [−1.37,+4.35] 含 0）；roll−plain ≈0/+2.58/−0.82。结论：HJ 收益与“结构方向特异”归因在 n=3 下不稳健，需 ≥5 seed；当前 HJ 结果降级为方向参考。见 `refactor/_runs/HJ_MULTISEED_PILOT.md`。
- **HJ 3-seed 试点（部分）**：seed 2027 plain=15.4467 / true=18.5474 / roll=18.0231，即 true−plain +3.10、true−roll 仅 +0.52（roll 相对 plain 也涨 +2.58）。这与 seed 2026 的 roll≈plain（true−roll +2.76）不一致，提示“结构方向特异”归因可能不稳健；等 2028 结果后做 3-seed 判定。
- 启动：`unsb_hj_multiseed.service` 跑 HJ 3-seed 归因试点（2027/2028 的 plain/true/roll，同 warmup 配对，val-O），脚本 `run_hj_pilot_multiseed.sh`，用于多 seed 验证 true-vs-roll 的结构方向特异归因。
- **结果**：3-seed DT 配对试点完成。delta +0.8875 / +0.7426 / +0.4687，mean +0.6996，95% CI [0.1712,1.2280]（不含 0）→ DT 相对 plain 方向稳健；plain 绝对基线跨 seed 波动约 1.5 dB，故各 trick 必要性/adaptive 仍需更大规模多 seed。见 `refactor/_runs/DT_MULTISEED_PILOT.md`。

## 2026-08-15

- 决策：因“单 seed=2026 稳定”这一前提已被 reflection_pad 非确定性证伪，改跑一个**有界 3-seed DT 试点**（新增 2027/2028，复用 warmup，脚本 `run_dt_pilot_multiseed.sh`），量化 +0.8875 的跨 seed 方差；最终大规模多 seed 仍由用户在更好服务器跑。
- 工具：新增 `analyze_diagnostics.py`，从诊断 JSONL 程序化算出 DT/HJ 的首次激活 epoch（均为 6）与末段均值，把“介入有效发生点”变成可审计证据。
- 工具：新增 `aggregate_multiseed.py`（跨 seed 均值/CI），并在 `REPRODUCE.md` 补每个 seed 的最小闭环命令；`summarize_dt_comparison.py` 的过时 target 字符串改为干净框架相对 +0.8875 口径。
- **关键发现（可复现性缺口）**：`train.py` 未设 `cudnn.deterministic`，`base_model.py` 又设 `benchmark=True`；进一步定位到生成器 `reflection_pad2d` 的 backward 无确定性实现，导致单 seed=2026 有约 1 dB 运行间方差。DT-A2（self-teacher，`U_match` 恒 0）实测 +1.07 dB，判定为噪声而非“frozen teacher 不必要”。已停掉非确定性三连跑，改为修正 determinism（`cudnn.deterministic=True` + 关 TF32），并把多 seed 作为唯一可靠结论路径写入 `REPRODUCE.md`/`PROGRESS.md`。
- 自动化：新增 `unsb_finalize.timer`（15 分钟）+ `finalize_dt_abl.sh`，在 `dt_abl_extra.done` 出现且训练服务退出后自动运行 `summarize_dt_abl_extra.py`；只写新结果文件，不碰 STATUS/HEARTBEAT/MONITOR 和现有 md。
- 产出：新增 `refactor/_runs/REPRODUCE.md`，串起 warmup→plain→DT→HJ→消融→自适应→诊断的完整复现顺序，并注明多 seed 需独立 `--name`/`--checkpoints_dir` 避免覆盖；`SUBMISSION.md` 已指向该文件。
- 修正：`METHOD_GROUNDING.md` 的 HJ gate 描述与代码对齐（原“risk≥τ ∧ boundary significant”是简写，真实实现为 `r=sqrt(r_dir·b)`，`b=4σ(m/0.001)(1-σ)`，再走 TopQ⁺(r,0.75) ∧ r≥0.05 ∧ Q(δ₊)≥δ_min）；并把自适应小节从“下一步”改成已出结论。
- 修复：`monitor_once.sh` 丢失可执行位（664），导致 systemd `203/EXEC Permission denied`、STATUS/HEARTBEAT 从 03:13 起停更；恢复 +x 并手动触发一次，三个自动写文件恢复正常。
- 审计：`EXPERIMENT_PLAN.md` 的 DT knock-out 计划 5 项，但 `metrics_abl/` 只跑了 A1（grouped→equal）和 A4（固定 λ），A2（frozen teacher→self）、A3（domain×time EMA→global）、A5（signal norm→raw U）未跑；这是“每个 trick 必要性”在 DT 侧的真实缺口（HJ 4/4 已完整）。
- 实现：给 clean DT 增加默认语义不变的三个消融开关 `--dtcov_teacher frozen|self`、`--dtcov_norm_mode domain_time|global`、`--dtcov_signal_norm on|off`；单测 10/10 pass，GPU smoke 验证 self 路径（损失恒 0，符合“去冻结参考”语义）与 global+raw 路径（非零 loss）。
- 启动：`unsb_train_all_dt.service`（systemd transient）串行跑 `dt_abl_a2_self` / `dt_abl_a3_global` / `dt_abl_a5_nonorm`，补齐 DT 必要性证据。

## 2026-08-14

- 根因：DT 复现失败的一个明确 bug 是 DT 窗口写成了 20 epoch，而原始 grouped 分支是 25 epoch（ramp5 hold15 decay25 需要 25 窗口，plain 从 26 续训）。已修正并重跑。
- 机制：确认 `systemd-run --user` 持久服务可靠（nohup 会随工具调用被杀）。现在用 `unsb_train_all_dt.service` 顺序跑 plain 基线 + 修正后的 DT。
- 进展：也确认 `unsb_monitor.timer` 每 30 分钟心跳正常（日志有 23:30/00:00 时间戳）。

- 重要发现：在 exec_command 里用 nohup 起的后台进程，会在工具调用结束后被杀（DT 训练和 watchdog 都验证了这一点）。所以“后台任务没跑”的根因是启动方式错了，不是没做。
- 修正：改用常驻 subagent（dt_train_worker）来执行训练，训练在该 agent 的回合里持续运行；由它每 30 分钟写 heartbeat 到 logs/HEARTBEAT.log。

- 行动：真正用 nohup 启动了 DT 干净版完整复现训练（后台进程 PID 2816581，日志 refactor/_runs/dtcov_warmup.log）。已用 pgrep 确认 train.py 正在跑。
- 修正：上一轮只跑了 smoke、未真正启动后台训练；已纠正。

- 进展：DT 干净实现 `--model dtcov` 通过 GPU 端到端 smoke（6 图 1 epoch，训练+保存成功），重构代码首次真正跑起来。
- 进展：HJ 干净实现 `--model hj` 已写并注册，但 smoke 触发 CUDA index out-of-bounds，根因是真实 netF 特征形状与合成单测不一致，待修。
- 修复：baseline argparse 历史 bug（netG/netD/netE choices 缺 _cond、--mode 与 --model 前缀冲突、缺 --seed），加 allow_abbrev=False。

- 里程碑：全量 CPU 测试通过（harness 12/12，DT 7/7，HJ 9/9）；DT 集成层 `SBModelDTCovMatch` 可正常 import baseline。
- 状态：L2 GPU 阶段的前置已就绪，下一步是接入 train/eval runner 做最小训练复现与消融。

- 验证：harness.metrics 在真实 HJ val-O per-image CSV 上复现 TRUE-PLAIN PSNR +1.4729 / LPIPS -0.0619 / SSIM +0.0164，以及 TRUE-ROLL SSIM -0.0202。
- 产出：refactor/FINDINGS.md 总结“必要机制 vs 工程惯性/假象”。

- 进展：HJ-PatchNCE 重构完成 L0/L1（9 个 CPU 测试 PASS）。因 hj_refactor subagent 两次停摆未产出，改由根 agent 直接按已读源码完成。
- 决策：两套 clean-room 重构都只保留最好分支的真实机制，把大量死代码和惯性开关移出核心。

- 进展：DT-CovMatch 重构完成 L0/L1（7 个 CPU 测试 PASS）。核心结论：best 分支的真正机制是 grouped-domain + 冻结 first-use teacher + domain×time EMA z-score + ramp-hold-decay；ua_scheme=12 / ua_train_rollout 等是训练期无效的历史开关。
- 验证：harness.metrics 用真实 DT per-image CSV 复现 plain 18.7360 / best 19.7800 / delta +1.0439，并给出 CI [0.7475, 1.3475]。
- 决策：hj_refactor 之前停在问方向，已重新派活要求直接落盘交付物。

- 进展：搭好 harness CPU 核心（data/config/determinism/checkpoint/metrics），self_test.py 12/12 PASS，验证了真实 train160/test40 与 val-O 的身份和零重叠。
- 决策：确认 DT 旧流水线只有 net 四件套、没有 training_state；HJ 有 full-state。harness 的 checkpoint.audit_checkpoint 同时兼容两种。
- 决策：重新派活 dt_refactor、hj_refactor，要求它们按 harness 约定产出 SPEC / TRICK_LEDGER / 最小实现 / CPU 测试。

- 决策：把执行顺序从“先重构算法”改为“先搭 harness，再重构”。
  - 原因：长序任务会漂移，harness 是可复现/可比较的底座，缺了它后面所有数字都不可信。
- 决策：门控分两层，科学阈值放松为方向信号，运营/成本/数据身份类硬门保持。
- 决策：暂停 dt_refactor、hj_refactor 两个 subagent，待 harness 接口稳定后重新派活。
- 决策：以 ICLR 级证据标准作为努力方向，但明确“标准不等于承诺”，保留诚实归因与 novelty 边界。
