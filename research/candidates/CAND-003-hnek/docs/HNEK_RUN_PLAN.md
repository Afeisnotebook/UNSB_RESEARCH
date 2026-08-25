# HNEK frozen 最小实验运行计划（2026-08-17）

> **已执行并被后续结果覆盖的历史计划：** 当前唯一 e200 存活开发候选是 hnek_g0.25；legacy 冻结 gamma=0.5 参照已失败。见 [当前裁决](../../../../decisions/CURRENT.md)。

## 0. Shim 状态（已完成）

- `hnek_kernel.py` / `hnek_adapter.py` 已原样拷入
  `refactor/baseline/models/hnek/`，并加 `__init__.py`。
- `SBModel.modify_commandline_options` 新增 `--hnek`（默认 False），
  构造后若开启则 `install_hnek_model(self)`。
- 只读验证：compile OK、shim import OK、`bridge_schedule(5)` 与 baseline `times` 一致、
  adapter 所需 SBModel 属性全部命中、0 新参数。

## 1. Frozen 协议（不改公式/阈值/schedule）

依据 `HNEK_FROZEN_SPEC.json` 与 `run_hnek_decisive.py`：

- 对照：`HNEK_PLAIN`（official time-dead）vs `HNEK_METHOD`（HNEK endpoint-kernel）。
- seed=2026；training 用 frozen T2 unpaired（900 steps/epoch）；development 用
  frozen five-domain T3（64 stems/domain × 4 replicates，paired）。
- 顺序：effect-free invariants → zero-training retrofit(e0) → coupled e50 →
  `hnek_adjudicator` 冻结门控。
- 门控（冻结）：
  - retrofit delta ≤ −0.50 dB → STOP_CATASTROPHIC_RETROFIT；
  - e50 delta ≥ +0.15 且 CI 下界 >0 且 ≥3/5 域为正 → PROCEED_DIRECT_TO_E200；
  - e50 delta ≤ −0.15 且 CI 上界 <0 → STOP_E50_CLEAR_FAIL；
  - 否则 PROCEED_TO_E100_INCONCLUSIVE。
- 口径：只给 single-seed development verdict；不写 confirmatory / SOTA / 桥求解正确性。

## 2. 数据 / harness 现实（关键，需定夺）

- Frozen 数据在 `data_root=/home/yc/UNSB_C21/dataset_all`：
  - 训练 T2 manifest：`.../specs/h2/T2_MANIFEST.json`；
  - 开发 T3 manifest：`.../specs/h2c/T3_CONFIRMATORY_MANIFEST.json`。
- 能加载 T2/T3、且带 coupled RNG + full-state + adjudicator 的 harness 在
  `UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806` 树（`scripts/final1` + `scripts/hnek`
  同树、T2/T3 数据 7.9G 存在）。
- `refactor/baseline` 现有数据管线是 final6 / val-O（cycleGAN-style `manifest.csv`），
  **不是 T2/T3**。因此 shim 只解决了“方法能装在干净 SBModel 上”，尚未解决
  “用哪份 frozen 数据/哪套 harness 跑生死测试”。

## 3. 两条可执行路径

### Path A（推荐，faithful，改动最小）

在 `UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806` 树里直接跑
`scripts/hnek/run_hnek_decisive.py`（其 `scripts.final1` 依赖同树存在）。

- 优点：T2/T3 + coupled RNG + full-state + adjudicator 已冻结，改动为 0。
- 缺点：不是 `refactor/baseline` 的 clean 重实现（但方法代码相同）。

### Path B（干净 shim，近似 coupled）

在 `refactor/baseline` 上：
1. 加一个 T2/T3 manifest 数据加载器（读 `UNSB_C21/dataset_all` + manifest）；
2. `--hnek` 训练 HNEK、关闭训练 PLAIN，同 seed、同 warmup 前缀、其余超参一致；
3. 用 T3 开发集评测，套 `hnek_adjudicator` 门控；
4. 补 hnek drift / invariant / entropy-h 只读诊断。

- 优点：落在 clean harness，`--hnek` shim 直接使用。
- 缺点：需新增 T2/T3 加载器（不只是 shim）；且是“近似 coupled”，不是严格共享 RNG bundle。

## 4. 诊断输出（每步）

- HNEK invariant：t=0/t=1 恒等、参数/state-key 不变、`layers/encode_only` 不变。
- entropy-h：用真实 `h=1-t` 的 `loss_SB` 熵项（对应修复 C003）。
- drift：teacher/参考漂移（如沿用 DT 的 drift 概念则需重新定义；HNEK 至少输出
  `horizon` / `r` 范数统计作为机制证据）。

## 5. 下一步

待定夺：Path A（faithful final1 harness）还是 Path B（refactor/baseline + T2/T3 加载器）。
确认后即可启动；启动后按门控逐关回传 verdict + 关键诊断。
