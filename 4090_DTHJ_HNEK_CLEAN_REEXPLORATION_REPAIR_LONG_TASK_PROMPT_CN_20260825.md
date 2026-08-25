# 4090 DT/HJ/HNEK 干净再探索修复与条件续跑长任务合同

> 文档身份：本文件是服务器 Agent 的唯一执行授权，不是讨论稿。  
> 任务性质：修复 2026-08-24 clean-reexploration 的实现偏差，先校准尺度与训练协议，再机械选择“复用既有资产”或“统一主干重建”，最后完成必要训练、冻结、评估和唯一回传。  
> 研究边界：single seed=2026、paired-development、非 confirmatory；不得据此宣称算法、创新性、ICLR 可行性或 Schrödinger Bridge 特异性已经成立。  
> 最终回传：`DTHJ_HNEK_CLEAN_REEXPLORATION_REPAIR_RETURN_20260825.zip` 与同名 `.sha256`。  
> 执行原则：除本文定义的 hard stop 外，不得中途请求作者选择；不得自行修改算法、门槛、split 或 claim boundary。

---

## 0. 作者裁决与本轮目标

作者已裁决：2026-08-24 回传包不能作为原长任务完成件。现有结果只可记为固定日程、单 seed 的暂定线索：

```text
canonical plain e200 = 13.603177819449025 dB
DT e200              = +0.019225495572989075 dB vs plain
HJ e200              = -0.1636381622244826 dB vs plain
HNEK FULL e200       = +0.2662758832126487 dB vs plain
```

已确认的实现事实如下，服务器不得重新争论或弱化：

1. 四条 lane 均以 `restore_path=None` 从 e1 各自构造，未按合同从 canonical full-state 分叉。
2. 训练循环只创建 controller，没有调用 `observe/record/decide`；DT/HJ/HNEK controller history 全为空。
3. DT 未实现完整的 `MC_floor_DT/E_DT/E_plain/R_DT`；HJ 未实现 RAW/TRUE/ROLL/SIGN 虚拟步；HNEK 未实现 `repeat_floor_H/B_H/safety`。
4. 当前 HNEK handoff 用未减 floor 的非负 energy distance 检查 `upper<=0`，使退出条件近乎不可达。
5. controller 代码存在“先过滤 invalid，再检查 invalid”的不可达分支。
6. HJ 在应当尚未启用的 e1 已与 plain 出现约 `-0.824 dB` 差异，当前 HJ 方法归因无效。
7. `ACCESS_LEDGER.csv` 只含 denied self-test，evaluator 直接 `Image.open(target)`，未强制匹配 `TRAINING_FROZEN.ok`。
8. 新 canonical plain 13.603 dB 与相同 T2/T3 身份、seed=2026 的历史 plain 约 18.8467 dB 存在约 5.24 dB 缺口；必须先区分 evaluator 偏差与训练 harness 偏差。
9. 既有 `full_state_e5.pt` 是完成 e5 后的 `post-e5`，不是 HJ 所需的 `pre-e5`。禁止把它直接当作 HJ 分叉点。

本轮目标不是追加参数搜索，而是回答下面四个机械问题并形成可审计结果：

1. 当前 evaluator 是否与历史权威尺度一致？
2. 当前训练数据顺序、sampler、配置和共同锚点是否足以复用既有 canonical/HNEK checkpoint？
3. 修复后的 DT/HJ/HNEK controller 是否真实参与训练、能恢复并能按冻结规则产生 OFF/HANDOFF？
4. 在唯一 canonical 主干上，DT、HJ、HNEK FULL/HANDOFF 的 paired-development 结果是什么？

---

## 1. 权威文件、路径和优先级

### 1.1 仓库与历史运行

默认服务器路径；若实际路径不同，可通过只读 Git/文件搜索定位，但必须记录 resolved absolute path 与 SHA-256：

```text
研究仓库：
/home/yc/unsb_tired

原始 2026-08-24 Prompt：
/home/yc/unsb_tired/4090_DTHJ_HNEK_CLEAN_REEXPLORATION_LONG_TASK_PROMPT_CN_20260824.md

本修复 Prompt：
/home/yc/unsb_tired/4090_DTHJ_HNEK_CLEAN_REEXPLORATION_REPAIR_LONG_TASK_PROMPT_CN_20260825.md

原运行目录（只读）：
/home/yc/unsb_tired/runtime_4090/clean_reexploration_20260824

新运行目录：
/home/yc/unsb_tired/runtime_4090/clean_reexploration_repair_20260825

历史 EvidenceFirst authority：
/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806
```

已知原运行身份：

```text
run_id = clean-reexploration-s2026-20260824-1db252fecfe97f43
frozen spec canonical SHA-256 = 1db252fecfe97f43339c6e79e010d0a07b0952c42c92afa0fc14c081ad709f44
executed code SHA-256 = 9cb41d24d4720e48df34d46615f2859e69b6ee45a0f09afe9e6b46eaff8857cc
T2 training manifest SHA-256 = f6049e7c1565d8e00e1baca1821b67b56d33bd78b064c596dbbc17d3d6e02
T3 paired-development manifest SHA-256 = 71b4eb92822166d67a97c15f9c5b2bbd8b4d70a24173d1ae03fe5c20596ddb0c
canonical post-e20 netG SHA-256 = c1fbf21758d96fc8169c965fab27877e5af008c641fed5b13014da8ff7c55d56
```

若任何已知身份与服务器实物不一致，触发 `HARD_STOP_INPUT_IDENTITY_MISMATCH`，不得猜测或选择“看起来最新”的文件。

### 1.2 科学优先级

发生冲突时按以下顺序：

1. 本修复 Prompt；
2. 原始 2026-08-24 Prompt 中未被本文件明确修正的算法公式、门槛和 claim boundary；
3. 原 frozen spec；
4. `CURRENT_STATE_CN.md` 与仓库中权威机器证据；
5. 历史实现代码；
6. 聊天上下文、日志叙述和 Agent 推断。

本文件没有授权重新设计 DT、HJ 或 HNEK。原 Prompt 中的算法配置、统计定义和退出规则保持不变；本轮只允许补齐缺失实现、共同锚点、恢复、隔离和报告。

---

## 2. 保护、分支和不可变输入

执行前必须：

1. 完整读取本文件、原 Prompt、`CURRENT_STATE_CN.md`、原 frozen spec 和原执行代码。
2. 记录当前 HEAD、branch、remote、`git status --porcelain=v1`、环境、GPU、磁盘和数据 root。
3. 从包含本 Prompt 的最新 HEAD 新建：

```text
repair/clean-reexploration-r1-20260825
```

4. 在项目外创建并 `git bundle verify`：

```text
/home/yc/UNSB_Long/UNSB_research_git_backup_20260825_clean_reexploration_repair.bundle
```

5. 原 runtime、旧 checkpoints、旧 return ZIP、历史 evidence、`CURRENT_STATE_CN.md`、原 Prompt 和原 frozen spec 全部只读；不得覆盖、移动、删除或“整理”。
6. 新代码、日志、checkpoint、spec、ledger 和报告全部写入新 runtime 或本轮新增模块。
7. 不得删除 `.git`；不得清理用户历史文件；只可清理新 runtime 内本轮可再生的临时缓存。

创建：

```text
state/REPAIR_INPUT_INVENTORY.json
state/PROTECTION_RECORD.json
state/REPAIR_ATTEMPT_LEDGER.jsonl
```

每次失败、修复、重试必须追加 attempt ledger，禁止最终打包时写空数组冒充“无失败”。

---

## 3. 执行状态机与禁止越级

唯一允许的阶段顺序：

```text
R0_PROTECT_AND_INVENTORY
  -> R1_IMPLEMENT_AND_SYNTHETIC_AUDIT
  -> R2_EVALUATOR_AND_HARNESS_CALIBRATION
  -> R3_PATH_ADJUDICATION
  -> R4_FREEZE
  -> R5_SOURCE_ONLY_DIAGNOSTICS_AND_TRAINING
  -> R6_TRAINING_FREEZE
  -> R7_PAIRED_DEVELOPMENT_EVALUATION
  -> R8_RECOMPUTE_REDTEAM_PACKAGE
```

规则：

- R4 之前不得读取当前 lane 的 paired effect。
- R4 之后不得修改算法、controller、门槛、sampler、evaluator 数学或 adjudicator 规则。
- R7 之前不得用当前 lane 的 paired target 做 checkpoint 选择、早停、调参或路径选择。
- 服务器不得基于已知 `+0.266 dB` 曲线修改 HNEK handoff 规则；只能实现原 Prompt 已冻结的规则。
- 工程路径只由 R2 的身份/等价性证据机械决定，不由 PSNR 高低决定。

---

## 4. R1：先修实现，再读真实诊断信号

所有真实 checkpoint controller signal 读取前完成本节，并提交代码。

### 4.1 统一 controller 接口必须真实接入

实现并在训练循环调用：

```text
observe(epoch, lane_state, canonical_plain_state, diagnostic_manifest)
    -> SignalRecord

record(SignalRecord)
decide(history)
    -> ACTIVE | OFF | HANDOFF

state_dict() / load_state_dict()
```

硬要求：

- `observe` 只在冻结 cadence 执行；不得推进主训练 RNG、optimizer、scheduler、sampler。
- 每次 scheduled audit 无论 valid/invalid 都必须写入 history；invalid 不得预先过滤。
- `decide` 必须以完整 history 为输入，检查连续 invalid 时能实际触发。
- OFF/HANDOFF 单向不可逆。
- controller history、连续计数、status、reason、frozen epoch、bootstrap draw identity 进入 full-state。
- 恢复后下一 audit 与不中断轨迹逐位一致。
- 训练日志每次 audit 输出 `method/epoch/status_before/status_after/reason/history_length`。

### 4.2 合成状态机硬测试

effect-blind synthetic fixtures 至少覆盖：

#### DT

- 两次 `E_DT.upper<=0` -> `DT_SIGNAL_EXHAUSTED/OFF`；
- 三次 `R_DT.lower<=0` -> `DT_NO_TARGET_BLIND_RESPONSE/OFF`；
- 一次 engineering-invalid -> lane engineering stop；
- invalid 记录必须保留在 history；
- countable epoch 前不退出。

#### HJ

- 连续两次 VALID -> ACTIVE；
- 连续两次 invalid -> `HJ_SIGNAL_NOT_ALIVE/OFF`；
- valid/invalid/valid 不得错误退出；
- e20 前不计数；最早 e30 才可能 OFF。

#### HNEK

- 两次 `C_H.upper<=0` -> `HNEK_SIGNAL_EXHAUSTED/HANDOFF`；
- 两次 `B_H.lower<=0` -> `HNEK_NO_LONGER_BEATS_PLAIN_UPDATE/HANDOFF`；
- 两次 safety lost -> `HNEK_NATIVE_SAFETY_LOST/HANDOFF`；
- engineering-invalid 能触发；
- e30 前不计数；默认 cadence 下最早 e40 handoff。

断言必须检查：精确 status、reason、frozen_epoch、history length、序列化后等价和恢复后下一决策等价。仅“不抛异常”不算通过。

### 4.3 三条真实诊断实现

不得使用 2026-08-24 简化实现作为验收标准。必须逐项实现原 Prompt：

#### DT

```text
MC_floor_DT
E_DT(e)
E_plain(e)
R_DT(e) = E_plain(e) - E_DT(e)
```

- teacher/reference bank 严格来自 canonical post-e20；
- teacher checkpoint SHA、netG SHA、参数冻结状态进入 spec；
- audit active age 为 `2,4,6,...,24,25`；
- canonical plain 同 epoch、同面板、同 random bundle；
- 保存 raw cluster、repeat floor、999 draws、point、单侧 bound。

#### HJ

必须执行不提交的完整 generator virtual step：

```text
RAW / TRUE / ROLL / SIGN
V_HJ
C_ROLL
C_SIGN
```

- 四条 virtual lane 从完全相同 G/F/D/E、RNG、batch、lr 开始；
- 只改变 PatchNCE backward 方向；
- D/E/F 不更新；
- 结束后科学状态与 RNG hash 恢复；
- 保存 one-sided/central agreement、gate/risk mass、removed norm、direction norm 和 gradient hash；
- ROLL/SIGN null 与 threshold 在真实 effect 前冻结。

#### HNEK

必须实现：

```text
C_H(e) = cross-time energy discrepancy - repeat_floor_H
DeltaC_H
DeltaC_P
B_H(e) = DeltaC_H - DeltaC_P
counterfactual plain objective safety envelope
```

- `repeat_floor_H` 必须实际相减；禁止把非负 raw ED 直接拿去检查 `upper<=0`；
- H_STEP/P_STEP 是同一 HNEK state、同一 batch/RNG 的完整 G virtual step；
- HNEK OFF 必须走 plain forward/E/G objective；
- snapshot/restore 包含全部网络、optimizer、scheduler、RNG 和 controller；
- 保存 raw units、repeat estimates、999 draws、CI、safety null 和状态 hash。

### 4.4 单元测试、真实模型 smoke 与独立 red-team

R1 至少完成：

- 全量 Python 语法检查；
- 原测试全过；
- 新 controller 状态机测试全过；
- snapshot/restore 与 RNG 隔离测试；
- DT `lambda=0`、HJ `strength=0`、HNEK OFF 的 plain 等价测试；
- HJ forward 不变测试；
- target guard 正负测试；
- wrong run/spec/code/checkpoint/anchor 拒绝测试；
- 真实 SBModel 单 batch CUDA smoke；
- 100-step identical twin；
- 独立进程 step50 checkpoint resume -> step51–100 bitwise。

首选 `STRICT_CUDNN`。失败后只允许一次统一 `STRICT_NATIVE_NO_CUDNN` fallback；两者都失败才 hard stop。禁止 tolerance、retry-until-match 或混合后端。

测试后服务器 Agent 必须重新读取实际 diff，而不是只读自己写的报告，执行第二遍 checklist。任何发现必须先修复、重新跑全套测试、重新 red-team，之后才进入 R2。

---

## 5. R2：校准 evaluator 和训练 harness

R2 只解决工程身份，不评价当前方法效果。

### 5.1 历史 evaluator oracle

优先使用已存在、hash-locked、已经公开过 paired-development 结果的历史 FINAL-1/HNEK plain checkpoint 与原始逐图 evidence。已知参考：

```text
T2 SHA = f6049e7c1563565d8e00e1baca1821b67b56d33bd78b064c596dbbc17d3d6e02
T3 SHA = 71b4eb92822166d67a97c15f9c5b2bbd8b4d70a24173d1ae03fe5c20596ddb0c
historical HNEK plain macro PSNR = 18.84673264733668
historical HNEK method macro PSNR = 19.63510470656946
historical delta = 0.7883720592327812
```

历史身份可从仓库：

```text
算法设计模块/evidence/hnek_search/state/hnek_g0.25/E200_EXECUTION_IDENTITY.json
算法设计模块/evidence/hnek_search/state/hnek_g0.25/eval_e200/SUMMARY.json
```

及其服务器 authority 路径解析。

校准过程：

1. 校验 checkpoint、manifest、历史 evaluator 和 raw evidence 哈希。
2. 若历史逐图 raw 存在，新 evaluator 用同 checkpoint、T3、4 replicates、seed/random bundle 重算。
3. 身份顺序、domain/stem、replicate 数必须完全相同。
4. 同一实现路径要求逐图 PSNR max abs `<=1e-5 dB`、SSIM max abs `<=1e-6`、macro PSNR abs `<=1e-6 dB`。
5. 若历史 raw 不存在，必须同时运行原 hash-locked evaluator 与新 evaluator；同 checkpoint 同输入的上述容差仍成立。
6. 仅比较 summary、不比较逐图或双 evaluator，不算通过。

该步骤允许通过隔离的 `CALIBRATION_ORACLE_ONLY` 进程读取已经饱和的历史 paired target，但必须满足：

- 只加载历史 checkpoint，绝不加载本轮/current lane；
- 方法/controller/sampler 代码在校准前已由 R1 固定；
- ledger 标明 `purpose=legacy_evaluator_calibration`；
- 校准结果只产生 PASS/FAIL，不得用于修改算法或阈值；
- 当前 lane paired effect 在 R7 前仍禁止读取。

最多允许两次 evaluator-scope 修复。仍不通过：`HARD_STOP_EVALUATOR_ORACLE_MISMATCH`。

### 5.2 训练配置与 sampler oracle

不得把当前 `build_deterministic_pairs()` 的排序下标绑定自动视为 unpaired sampler 等价。

建立两条只读/短程 oracle：

1. `AUTHORITATIVE_FINAL1`：由 hash-locked FINAL-1/baseline 数据加载与训练 harness 产生；
2. `REPAIR_CANDIDATE`：本轮拟采用的 loader/harness。

在相同 T2、seed=2026、batch=1 下记录至少前 100 个主训练 step：

```text
step
A domain/stem/path identity
B domain/stem/path identity
data2 A/B identity
sampler position
crop/resize/flip parameters或变换后输入 hash
time index
bridge noise hash
z hash
PatchNCE sample-id hash
```

通过条件：

- 两条 transcript 的所有身份与 random bundle 逐行一致；
- `pairing_used_for_training=false`；
- 不得读取 paired target；
- 两个独立进程重复 transcript bitwise 一致；
- resume 后 transcript 与不中断一致。

若 FINAL-1 本身没有可调用的 loader oracle，必须直接复用其真实 loader/sampler 代码，不得凭文档重写后自证。

### 5.3 当前 canonical 的自重放审计

从原运行的 canonical `pre-e1/full_state_e0` 恢复，使用原执行代码和原 loader 精确重放：

1. e1–e4，保存真正的 `pre_e5=post_e4`；
2. 再运行 e5，与原 canonical `full_state_e5.pt` 比较；
3. 再运行到 e20，与原 canonical `full_state_e20.pt` 比较。

“科学状态”比较包括：

```text
G/F/D/E 参数与 buffers
所有 optimizer state
所有 scheduler state
Python/NumPy/Torch CPU/CUDA RNG
sampler epoch/position
global step
physical epoch
下一 batch identity
下一 random bundle hash
```

只允许忽略明确列出的非科学元数据字段，例如 lane name、创建时间和绝对输出目录；忽略列表必须冻结在比较前。所有科学状态必须 bitwise 相等，禁止容差。

重要：原 `full_state_e5.pt` 只能作为 `post-e5` 比较对象，永远不能直接作为 `pre-e5`。

### 5.4 现有 HNEK FULL 的复用身份门

现有 HNEK FULL 只有全部满足才允许复用：

1. 将 canonical pre-e1 加载到 HNEK wrapper，先设 HNEK OFF；除允许的 controller/identity 元数据外，科学状态与 canonical pre-e1 bitwise 一致。
2. 切换 HNEK ON 不改变任何参数、buffer、optimizer、scheduler、RNG、sampler 或 state-dict key，只改变可序列化 mode flag。
3. 从 canonical pre-e1 按冻结 HNEK 配置重放 e1，科学状态与原 HNEK `full_state_e1.pt` bitwise 一致。
4. 至少继续 spot-check e5、e20；与原 HNEK checkpoint 科学状态 bitwise 一致。
5. 下一 batch 与 random bundle identity 一致。

任何一项失败，现有 HNEK FULL 只能作为历史线索，不得作为本轮 handoff parent。

### 5.5 R2 输出

生成：

```text
state/EVALUATOR_ORACLE_AUDIT.json
state/TRAINING_HARNESS_ORACLE_AUDIT.json
state/CANONICAL_REPLAY_AUDIT.json
state/HNEK_REUSE_IDENTITY_AUDIT.json
state/PAIRING_SEMANTICS_AUDIT.md
```

---

## 6. R3：机械选择 SALVAGE 或 REBUILD

不得询问作者。只允许以下 truth table：

### 6.1 `SALVAGE_PARTIAL_RETRAIN`

只有以下全部成立：

- evaluator oracle PASS；
- repair candidate 与 authoritative FINAL-1 sampler/config transcript PASS；
- 原 canonical 自重放 e5/e20 bitwise PASS；
- canonical pre-e1/pre-e5/post-e20 身份完整；
- HNEK reuse identity PASS。

动作：

- 复用原 canonical plain checkpoint；
- 复用原 HNEK FULL checkpoint；
- 从 canonical pre-e5 重跑 HJ e5→e200；
- 从 canonical post-e20 重跑 DT e21→e200；
- 用现有 HNEK FULL 做完整 source-only controller 审计，冻结 e_star 后只跑 HNEK HANDOFF continuation；
- 不重训 canonical plain、不重训已证明身份等价的 HNEK FULL。

### 6.2 `REBUILD_COMMON_TRUNK`

条件：evaluator oracle PASS，但 sampler/config、canonical replay 或 HNEK reuse 任一失败。

动作：

- 使用已通过 oracle 的 authoritative FINAL-1 loader/config 建立唯一 canonical pre-e1；
- canonical plain 只训练一次 e1→e200；
- 保存真正 pre-e5、post-e20 和所有所需 matched audit checkpoint；
- HNEK 从 canonical pre-e1 full-state 分叉；
- HJ 从 canonical pre-e5 分叉；
- DT 从 canonical post-e20 分叉并加载该 state 的冻结 teacher；
- 所有分叉继承 sampler/RNG/global step，不得重新 `_seed_all()` 冒充共同锚点；
- 完成 HNEK FULL 及 controller-triggered HANDOFF。

这是预授权的条件 fallback，不需要作者中途确认。不得因为它更费时而退回独立 e1 初始化。

### 6.3 hard stop

若 evaluator oracle 失败，或 authoritative loader/data identity 无法建立，触发 hard stop。不得在未知尺度上训练。

R3 写：

```text
state/REPAIR_PATH_ADJUDICATION.json
```

必须包含每个 truth-table 输入、证据路径、SHA 和唯一选择。

---

## 7. R4：重新冻结

在读取真实 controller signal、训练方法 lane 或当前 paired effect 前冻结：

```text
specs/repair/CLEAN_REEXPLORATION_REPAIR_FROZEN_SPEC.json
state/CODE_SHA256.json
state/SPEC_CANONICAL_SHA256.txt
state/RUN_ID.txt
state/PRE_EFFECT_AUDIT.json
state/PRE_EFFECT_RED_TEAM.md
```

run_id：

```text
clean-reexploration-repair-s2026-20260825-<spec_sha前16位>
```

frozen spec 必须列出：

- 选择的 SALVAGE/REBUILD 路径；
- 所有输入 checkpoint path/payload SHA/scientific-state SHA；
- pre-e1、pre-e5、post-e20 身份；
- evaluator/loader/code SHA；
- 数据 manifests；
- backend；
- controller formulas、cadence、threshold、synthetic test hashes；
- HNEK FULL reuse 与否；
- target guard/freeze schema；
- 预算和自动降级顺序。

冻结后禁止 amend 旧提交来隐藏历史。代码若因工程 bug 必须修复：新 commit、追加 attempt ledger、重新跑全部 R1–R4 门、生成新 spec/run_id；最多两次同根因修复。

---

## 8. R5：source-only 诊断与训练

### 8.1 canonical matched states

DT 所需 canonical e22/e24/.../e45 若原 checkpoint 不存在：

- 从 canonical post-e20 确定性重放 plain；
- 保存 audit-only matched states；
- 在已有 canonical e30/e40 等交叉点验证科学状态 bitwise；
- 不得另训一条“DT plain”。

HJ 与 HNEK matched epoch 同理，所有对照来自唯一 canonical 主干。

### 8.2 DT lane

- parent = canonical post-e20 full-state；
- teacher = 同一 post-e20 netG，hash-lock、冻结；
- e21 第一个 batch 启用；
- 原 `lambda=0.001` 与 ramp/hold/cosine active-age 不变；
- 按原 cadence 在线调用 controller；
- OFF 后永久 `lambda=0`，沿同一 state 继续 plain objective 到 e200；
- controller state 与 checkpoint 一起保存。

### 8.3 HJ lane

- parent = 真正 canonical pre-e5/post-e4 full-state；
- e5 第一个 batch 启用；
- layer0/joint/central/strength=0.5/remove 等配置不变；
- e1–e4 不存在独立 HJ 训练；
- 按 e10,e20,...,e200 做完整虚拟步 audit；
- 连续两次 invalid 后 strength 永久归零并继续到 e200；
- 至少证明 parent、下一 batch、下一 random bundle 与 canonical 分叉点一致。

### 8.4 HNEK FULL 与 HANDOFF

- HNEK FULL 必须来自 canonical pre-e1；SALVAGE 时须通过复用身份门；REBUILD 时重新从共同锚点训练。
- 配置严格为 `gamma=0.25/residual/physical/all`。
- HNEK FULL 始终 ON 到 e200。
- 在 e10,e20,...,e200 做完整 `C_H/B_H/safety` 审计。
- 若复用资产缺少 e110/e130/e150/e170/e190 等 required audit state，只允许从最近 hash-locked HNEK checkpoint 确定性重放补齐，并在后续已有 checkpoint 处证明 scientific state bitwise；不得减少 cadence。
- 第一次按原规则形成 e_star 后，从该 HNEK FULL full-state 分叉 HANDOFF；OFF 后用 plain objective 继续到 e200。
- 若到 e200 没有 handoff，记录 `HNEK_HANDOFF_NOT_TRIGGERED`；不得用 paired 最佳 epoch 人工选择。
- 不执行新的 gamma、coord、partial、seed 搜索。

### 8.5 HNEK e50 描述性 handoff

只有预计总 GPU 时间仍低于 hard budget 的 85% 时，才执行原 Prompt 的固定 e50 handoff。预算不足自动取消，主 HNEK FULL 与目标盲 HANDOFF 不得取消。

### 8.6 每条 lane 的必要运行证据

每个主 step 至少以可压缩/分片形式记录：

```text
run/spec/code/lane/checkpoint identity
physical epoch/global step/sampler position
A/B/data2 identity
time/noise/z/PatchNCE sample-id hashes
controller status/history length
finite flags
```

不得把诊断 RNG 混入主训练 RNG。普通中断自动 full-state resume，不触发作者闸门。

---

## 9. R6：训练冻结与 target 访问门

只有以下全部成立才能创建新 runtime 中唯一的：

```text
TRAINING_FROZEN.ok
```

条件：

- canonical plain 到 e200；
- DT/HJ 到 e200或形成文档允许的 lane-local engineering stop；
- HNEK FULL 到 e200；
- 已触发的 HNEK HANDOFF 到 e200；
- controller histories 非空且 audit count 与 cadence 一致；
- 所有 checkpoint、scientific-state、code/spec/run 身份冻结；
- training target actual read count = 0；
- 不再有待运行的训练 lane；
- 不再允许修改方法、controller、sampler、evaluator 或 adjudicator。

marker 必须为 canonical JSON，至少包含：

```text
run_id
spec_sha256
code_sha256
training_manifest_sha256
paired_manifest_sha256
checkpoint_index_sha256
controller_state_index_sha256
terminal_lane_states
training_target_read_count
created_utc
```

evaluator 启动时必须重新计算并匹配 marker；任一不一致拒绝运行。

### 9.1 访问 ledger

实际使用 CSV schema，不得以 JSONL 冒充 CSV：

```text
timestamp_utc,phase,lane,epoch,purpose,role,path,resolved_path,stem,allowed,
training_frozen_sha256,run_id,spec_sha256,code_sha256
```

- 所有 target 打开都必须通过 `TargetAccessGuard.open_image()` 或等价唯一入口；禁止 evaluator 直接 `Image.open(target_path)`。
- freeze 前负向测试必须拒绝并记账。
- freeze 后每个实际 paired target read 必须记 allowed 行。
- 最终 ledger 的 allowed target identity/count 必须与 evaluator raw rows 可机械对应。
- source/input 训练读取可分片记账，但 paired target 不得采样省略。

发现 freeze 前真实 paired target read：`HARD_STOP_PAIRED_TARGET_LEAK`。

---

## 10. R7：统一 paired-development 评估与机械裁决

只使用 hash-locked T3 五域、320 images、4 replicates。RainDS-syn 缺失保持边界，不补 split。

评估：

- canonical plain 全部保存 checkpoint；
- DT/HJ 全部保存 checkpoint；
- HNEK FULL 全部保存 checkpoint；
- HNEK HANDOFF 的 e_star 与 e200；
- 如执行，HNEK E50_HANDOFF 的 e50 与 e200。

每图输出 PSNR/SSIM，并输出五域 macro、同 epoch plain 配对差、999-draw source-in-domain cluster bootstrap CI、正域数。禁止以 paired 指标回头选择 checkpoint。

### 10.1 机械标签

DT/HJ 保持原规则：

```text
DEVELOPMENT_GAIN
  iff e200 macro delta >= +0.15 dB
  and paired bootstrap lower > 0
  and positive domains >= 3/5
else DEVELOPMENT_NO_GAIN
```

为避免再次漏报，HNEK FULL 必须额外给出同口径的纯描述标签：

```text
HNEK_FULL_DEVELOPMENT_GAIN
  iff e200 macro delta >= +0.15 dB
  and paired bootstrap lower > 0
  and positive domains >= 3/5
else HNEK_FULL_DEVELOPMENT_NO_GAIN
```

该标签只表示本 single-seed paired-development lane 的数值状态，不替代 HNEK handoff 裁决。

HNEK HANDOFF 保持原规则：

```text
HANDOFF_OPTIMIZATION_GAIN
  iff handoff e200 - full e200 >= +0.10 dB
  and handoff e200 - canonical plain e200 >= +0.15 dB
  and both paired lower bounds > 0
  and handoff positive domains vs plain >= 3/5
else HANDOFF_OPTIMIZATION_NO_GAIN
```

必须同时报告：

- HNEK FULL 的全部同 epoch delta 曲线；
- controller signal 与下一保存区间 paired delta 的 Spearman/符号一致率，描述性；
- e200 之外的最佳 checkpoint仅标为 `POST_HOC_DESCRIPTIVE_MAX`，不得作为主结论；
- 新 canonical plain 与历史 18.8467 oracle 的差异及解释边界；
- SALVAGE/REBUILD 路径。

`adjudicate.py` 必须只从 portable raw evidence 纯函数重算所有 summary 和标签。

---

## 11. 长程 runner、自动恢复和预算

建立唯一入口：

```bash
bash LAUNCH_4090_CLEAN_REEXPLORATION_REPAIR.sh
```

要求：

- 在 tmux 中运行；
- heartbeat 至少每 5 分钟更新阶段、lane、epoch、global step、GPU/磁盘、最后 checkpoint 和 controller status；
- 重复启动同一 run_id 时幂等恢复；
- 已完成且身份匹配的阶段跳过；身份不匹配拒绝；
- 普通 warning、进程中断、已有缓存、source-only 信号 OFF/HANDOFF 不向作者提问；
- 不得在聊天中分阶段回传。

预算：

```text
GPU hard budget = 48 h
wall-clock hard budget = 72 h
new disk hard budget = 120 GiB
```

自动降级顺序：

1. 降低非主诊断的保存密度，但不减少 frozen controller cadence；
2. 删除可再生可视化，不删除 raw statistic/draws；
3. 取消固定 HNEK_E50_HANDOFF；
4. 不得取消 canonical、DT、HJ、HNEK FULL 或目标盲 HANDOFF。

预计超过 48 GPUh 才触发 `HARD_STOP_BUDGET_FORECAST`；不得静默降低 epoch、图片数、公式或主 lane。

---

## 12. Hard stop 与允许修复

只有以下情况可提前停止：

- `HARD_STOP_INPUT_IDENTITY_MISMATCH`
- `HARD_STOP_EVALUATOR_ORACLE_MISMATCH`
- `HARD_STOP_AUTHORITATIVE_HARNESS_UNAVAILABLE`
- `HARD_STOP_DATA_MANIFEST_MISMATCH`
- `HARD_STOP_PAIRED_TARGET_LEAK`
- `HARD_STOP_DETERMINISM_BOTH_BACKENDS_FAILED`
- `HARD_STOP_GLOBAL_NONFINITE`
- `HARD_STOP_BUDGET_FORECAST/EXCEEDED`
- `HARD_STOP_DISK_CAPACITY`
- 同一工程根因在允许的两次 scoped repair 后仍未闭合

以下不是 hard stop：

- DT/HJ 自动 OFF；
- HNEK HANDOFF 或未触发 handoff；
- paired 指标负收益；
- 某方法没有 DEVELOPMENT_GAIN；
- 普通断电/进程中断；
- SALVAGE 不成立而机械切换 REBUILD；
- 旧 2026-08-24 controller history 为空。

若 hard stop，仍需完成失败证据打包，不得只在聊天中描述。

---

## 13. 最终自审、fresh-directory 验收与唯一回传

### 13.1 最终独立复核

在查看最终报告后，服务器必须启动一个不依赖报告文字的独立复核脚本，从 raw evidence 检查：

- manifest/ZIP/file hashes；
- Prompt/spec/code/run/checkpoint identities；
- SALVAGE/REBUILD truth table；
- pre-e1/pre-e5/post-e20 锚点；
- controller audit counts、history 非空、状态转换和恢复；
- DT floor/plain response；
- HJ RAW/TRUE/ROLL/SIGN；
- HNEK floor/B_H/safety/handoff；
- target access 时序与 ledger；
- per-image row counts、bootstrap draws 和标签纯函数重算；
- 没有旧 paired result 混入新 raw；
- 没有 NaN/Inf 静默置零；
- attempt ledger 非空时报告如实列出。

### 13.2 ZIP 内容

唯一 ZIP 至少包含：

```text
本 Prompt 与原 Prompt
冻结 repair spec
实际执行代码与测试
REPAIR_INPUT_INVENTORY
PROTECTION_RECORD
REPAIR_PATH_ADJUDICATION
EVALUATOR_ORACLE_AUDIT
TRAINING_HARNESS_ORACLE_AUDIT
CANONICAL_REPLAY_AUDIT
HNEK_REUSE_IDENTITY_AUDIT
DETERMINISM_GATE
PRE_EFFECT_AUDIT/RED_TEAM
TRAINING_FROZEN marker
controller raw/history/draws/null/control
paired per-image metrics/bootstrap draws
mechanical adjudication
ACCESS_LEDGER.csv
REPAIR_ATTEMPT_LEDGER.jsonl
checkpoint/netG/scientific-state hash index
Git/code/spec provenance
logs/heartbeat/runtime summary
FINAL_REPORT.md
MANIFEST.sha256
```

不包含：

- 数据像素；
- checkpoint 本体；
- `.git`；
- `__pycache__`、`.pyc`、`.deps`；
- 嵌套 ZIP；
- 未经索引的大体积缓存。

checkpoint 不入 ZIP，但必须有 path、payload SHA、sidecar SHA、scientific-state SHA、parent anchor SHA 和存在性记录。

### 13.3 fresh-directory 验收

从最终 ZIP 解压到全新临时目录：

- `sha256sum -c` 全过；
- manifest 覆盖所有文件且仅自排除；
- JSON/CSV/UTF-8/相对链接通过；
- portable adjudicator 从 shipped raw 重算一致；
- 关键计数一致；
- 禁止文件扫描为空。

最终只回传：

```text
DTHJ_HNEK_CLEAN_REEXPLORATION_REPAIR_RETURN_20260825.zip
DTHJ_HNEK_CLEAN_REEXPLORATION_REPAIR_RETURN_20260825.zip.sha256
```

---

## 14. 最终机械六栏

服务器最终只允许输出：

1. **事实**：身份、SALVAGE/REBUILD、校准、共同锚点、controller、运行规模和指标；
2. **工程失败**：真实 attempt、修复和剩余工程问题；
3. **机械结果**：DT/HJ、HNEK FULL、HNEK HANDOFF 冻结标签；
4. **未决**：single seed、paired-development、RainDS-syn 和训练 seed 边界；
5. **下一动作**：只写“等待作者本地审查回传包”；
6. **是否触发作者闸门**：任务完成或 hard stop 后唯一一次触发。

禁止写：

- 算法“已经立住/已经失败”；
- “可投/不可投 ICLR”；
- “SB 特异性已证明”；
- 用 post-hoc 最佳 checkpoint 替换 e200 主裁决；
- 把没有执行的 controller 描述成 ACTIVE；
- 把空 history 描述为“信号持续存在”；
- 把 `post-e5` 写成 `pre-e5`；
- 隐瞒 REBUILD、失败尝试或 paired access。

---

## 15. 完成定义

只有下列全部成立，本任务才是 `COMPLETE`：

- evaluator 和 authoritative training harness 已校准；
- SALVAGE/REBUILD 由冻结 truth table选择；
- 所有方法来自唯一 canonical full-state 锚点；
- HJ 使用真正 pre-e5，DT 使用 canonical post-e20，HNEK 使用 canonical pre-e1；
- controller 在真实执行中 history 非空、状态可恢复、真实调用 observe/record/decide；
- DT/HJ/HNEK 完整诊断公式与 controls 已执行；
- HNEK FULL/HANDOFF 状态明确；
- target 只在新 TRAINING_FROZEN marker 后经 guard 读取；
- raw evidence 可复算；
- 最终报告没有漏掉 HNEK FULL；
- ZIP 与 sidecar fresh-directory 验收通过。

任何仅“修了类、跑了固定 lane、事后算四个 checkpoint signal、再生成报告”的实现都不算完成。

现在开始执行。除本文 hard stop 外，不要中途请求作者授权；结束时只回传唯一 ZIP、sidecar 和机械六栏。
