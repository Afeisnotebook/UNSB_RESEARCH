# DT / HJ / HNEK 干净工程再探索：4090 单次长任务授权合同

> 日期：2026-08-24  
> 目标硬件：NVIDIA RTX 4090 24 GiB  
> 任务类型：本地完成科学设计，服务器 Agent 负责实现、自审、冻结、长程执行、恢复、机械评估与唯一回传  
> 最终回传：`DTHJ_HNEK_CLEAN_REEXPLORATION_RETURN_20260824.zip` 及其 `.sha256`

---

## 0. 你的身份、唯一目标与禁止事项

你是**受约束的工程实现与实验执行 Agent**，不是论文方法设计者，也不是论文价值裁决者。

本任务接受以下研究前提，不再要求你判断三条路线是否“足以成为创新点”：

1. DT-CovMatch、HJ-PatchNCE 与 HNEK 都进入一次干净工程再开发；
2. 旧结果只作为历史线索，不作为新实验的控制轨迹；
3. 本轮要回答的是：在共同初始化、共同 plain 主干、显式物理 epoch、确定性状态和目标盲停止规则下，三条路线能否取得比旧工程实现更可靠的开发结果；
4. 服务器不得因为科学信号弱、收益为负或方法与历史不一致而提前向作者请求路线选择。科学信号弱只触发本文规定的自动 `OFF/HANDOFF`，任务仍继续到 e200。

你必须完成：

- 阅读当前 Git 项目和本文指定的权威文件；
- 在独立分支实现 clean-reexploration 模块、测试、冻结 spec、长程 runner、恢复和打包器；
- 在读取任何 paired effect 之前完成工程自审、red-team 与身份冻结；
- 在 4090 上运行完整任务，目标总预算不超过 48 GPU 小时；
- 全部训练冻结后才读取既有 paired-development；
- 只给出机械结果，不写 novelty、ICLR readiness、SOTA、论文故事或超出单 seed 开发实验的结论；
- 无论完成或 hard stop，都只回传一个 ZIP 与一个外部 SHA-256 sidecar。

严禁：

- 自行修改本文的算法、阈值、连续判据、epoch、seed、数据身份和预算降级顺序；
- 把 TA、KCK、CET、DTHS、task-vector fusion 或历史联合训练代码混入本任务；
- 每个方法重新训练一条自己的 plain 对照；
- 根据 paired PSNR/SSIM 决定在线停止、调参、改阈值或追加变体；
- 打开 official test、官方 confirmation 或任何未在本文授权的 split；
- 用“服务器认为更合理”为由替换当前 Git 中 DT/HJ/HNEK 的核心公式；
- 删除、覆盖或改写历史证据、`CURRENT_STATE_CN.md`、旧 Git 分支或旧 checkpoint；
- 为了通过确定性测试而反复重启碰运气、接受数值容差或重试直到命中某个 bit pattern；
- 在训练尚未完成时发送普通阶段性结果并要求作者选择下一步。

---

## 1. 权威顺序与必读材料

### 1.1 权威顺序

出现冲突时严格采用以下优先级：

1. **本文档**：本任务的算法、执行、门控、预算和回传合同；
2. 当前 Git HEAD 中的 clean-room 源码与算法规格；
3. FINAL-1 的基础训练配置、数据 manifest 和数据访问身份；
4. 其他历史报告、旧 Prompt、旧 return bundle 和聊天推断。

低优先级材料不得覆盖高优先级材料。旧文档中的 PASS/FAIL、固定窗口收益和旧自适应规则不是本任务的自动裁决依据。

### 1.2 当前 Git 项目必读顺序

在写代码前完整阅读：

1. `CURRENT_STATE_CN.md`
2. `README.md`
3. `算法设计模块/README.md`
4. `算法设计模块/docs/PROJECT_WORK_SUMMARY.md`
5. `算法设计模块/docs/FINAL_STATUS.md`
6. `算法设计模块/code/dt_covmatch/SPEC.md`
7. `算法设计模块/code/dt_covmatch/REPORT.md`
8. `算法设计模块/code/hj_patchnce/SPEC.md`
9. `算法设计模块/code/hj_patchnce/REPORT.md`
10. `算法设计模块/code/baseline/models/hnek/hnek_search.py`
11. `算法设计模块/evidence/CLEAN_CORE_RESULTS.json`
12. `算法设计模块/evidence/hnek_search/E200_CONFIRMATION.json`
13. `算法设计模块/evidence/hnek_search/state/hnek_g0.25/E200_EXECUTION_IDENTITY.json`
14. `算法设计模块/evidence/hnek_search/state/hnek_coord_y/E200_EXECUTION_IDENTITY.json`

然后阅读实际训练链：

- `算法设计模块/code/baseline/train.py`
- `算法设计模块/code/baseline/models/sb_model.py`
- `算法设计模块/code/baseline/models/base_model.py`
- `算法设计模块/code/dt_covmatch/dtcov/`
- `算法设计模块/code/dt_covmatch/dtcov/model.py`
- `算法设计模块/code/hj_patchnce/hj/`
- `算法设计模块/code/hj_patchnce/hj/model.py`
- `算法设计模块/code/baseline/models/hnek/`

### 1.3 历史资产只承担基础身份

服务器历史资产默认根目录：

```text
/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806
```

默认数据根目录：

```text
/home/yc/UNSB_Long/dataset_all
```

若环境变量 `DATA_ROOT` 已显式给出，优先使用它，但必须重新做只读 manifest 审计。

从历史资产中只允许复用：

- FINAL-1 的基础训练/优化器/数据参数；
- 原 T2 unpaired training manifest；
- 原 paired-development manifest；
- 数据访问策略、full-state checkpoint schema 和 deterministic runtime 经验。

必须定位并核对下列已记录身份：

```text
FINAL-1 frozen spec canonical SHA-256:
bb102af286f0d15f1f6b3bd0e562964d70cf86af463323373ae027e7194f4d86

training manifest SHA-256:
f6049e7c1563565d8e00e1baca1821b67b56d33bd78b064c596dbbc17d3d6e02

paired-development manifest SHA-256:
71b4eb92822166d67a97c15f9c5b2bbd8b4d70a24173d1ae03fe5c20596ddb0c
```

若无法定位这些身份，不得自行重新划 paired split；触发 `HARD_STOP_MISSING_BASE_AUTHORITY`，完成失败证据打包。

训练源码只允许使用当前 Git 的 `算法设计模块/code/baseline` 及当前 DT/HJ/HNEK clean-room 模块。不得直接以历史 FINAL-1 的 TA/KCK/CET 模型作为本轮 baseline。

---

## 2. 开工保护、分支和目录合同

### 2.1 Git 保护

1. 确认当前目录为 `UNSB_RESEARCH` Git 根目录，本文档存在于该 HEAD。
2. 记录 branch、HEAD、remote、`git status --porcelain=v1` 和所有已修改/未跟踪文件。
3. 不覆盖用户已有修改；若存在修改，只在不冲突的新目录工作，并把初始状态写入保护记录。
4. 从当前 HEAD 新建分支：

   ```text
   rebuild/dthj-hnek-clean-reexploration-20260824
   ```

5. 在 Git 项目外创建完整 bundle：

   ```text
   /home/yc/UNSB_Long/UNSB_RESEARCH_backup_20260824_clean_reexploration.bundle
   ```

6. 运行 `git bundle verify`，记录 bundle SHA-256；不得删除或移动 `.git`。

### 2.2 新实现落点

新代码统一放在：

```text
算法设计模块/code/clean_reexploration/
```

至少包含：

```text
frozen/
tests/
controllers.py
diagnostics.py
identity.py
full_state.py
run_long.py
adjudicate.py
package_return.py
README.md
```

仓库根目录新增唯一启动脚本：

```text
LAUNCH_4090_CLEAN_REEXPLORATION.sh
```

只允许对下列现有位置做最小接入：

- `算法设计模块/code/baseline/train.py`
- `算法设计模块/code/baseline/models/` 中必要的基类或方法注册文件
- `.gitignore`，仅用于排除本任务 runtime/checkpoint/cache

禁止把新逻辑复制进多个旧模型文件。算法控制、身份、诊断、恢复和裁决必须集中在新模块。

### 2.3 Runtime 落点

所有大工件写入：

```text
runtime_4090/clean_reexploration_20260824/
```

至少包含：

```text
authority/
state/
runs/
logs/
raw/
portable/
return_staging/
```

checkpoint 和逐步大日志只留服务器，不进入 Git，也不进入最终 ZIP；最终 ZIP 仅携带 checkpoint hash index。

---

## 3. 本轮为什么必须重建控制轨迹

把以下事实写入 `state/PRE_IMPLEMENTATION_FINDINGS.md`，但不得扩大解释：

1. `CLEAN_CORE_RESULTS.json` 中同为 seed=2026 的 DT plain 与 HJ plain 分别约为 18.8898 和 18.4452 dB。它们不是同一条可复用控制轨迹，不能跨实验直接拼接解释。
2. HNEK `g0.25` 与 `coord_y` 的 e200 plain 分别约为 18.8467 和 19.8533 dB，相差约 1.007 dB；二者虽记录相同顶层 seed 和 deterministic flags，但拥有各自的历史 plain 训练轨迹。
3. 当前 DT 代码的 teacher 是首次启用时 deep-copy；仅设置 iteration warmup 不能证明 teacher 等于物理 e20 checkpoint。
4. DT/HJ 当前内部 epoch 由 epoch 末的 `update_learning_rate()` 推进，存在与物理 epoch 标签错一拍的风险。
5. HNEK 只有 e50/e200 两个稀疏 paired 观测点；`g0.25` 从 +2.6173 dB 降到 +0.7884 dB，`coord_y` 从 +3.1481 dB 翻到 −1.2164 dB。它们支持检查停止时机，但不能证明真实最优 epoch。

本任务只据此重新建立共同控制和信号轨迹；不得把这些事实表述成“旧 DT/HJ 已被证明有效”或“旧负结果全部无效”。

---

## 4. 不可更改的共同实验合同

### 4.1 单一 canonical plain spine

只训练一条 canonical plain UNSB：

```text
seed = 2026
epoch = 1 ... 200
training manifest = 上述 hash-locked T2 unpaired manifest
lambda_NCE = 1
model source = 当前 Git clean baseline
```

从相同初始化 full-state 建立以下锚点：

- `pre_e1`：data-dependent initialization 完成、第一次 optimizer step 之前；
- `pre_e5`：物理 e4 完成、物理 e5 尚未开始；
- `post_e20`：物理 e20 全部 step 与 epoch-end scheduler 完成；
- `e10, e20, ..., e100, e120, e140, e160, e180, e200` 的 full-state；
- DT 激活期额外保存 e21–e45 每 5 epoch full-state。

full-state 必须包括 G/F/D/E、所有 optimizer、scheduler、Python/NumPy/Torch CPU/CUDA RNG、sampler epoch/position、global step、physical epoch、数据身份、run/spec/code identity 和 controller state。

DT/HJ/HNEK 的任何结果只能引用这条 canonical plain。不得为各方法或变体重新创建独立 plain 对照。

### 4.2 显式物理 epoch

在每个训练 epoch 的第一个 batch 前显式调用：

```python
model.set_train_epoch(physical_epoch)
```

并把 physical epoch 写入 checkpoint、日志和每行诊断。不得通过 scheduler 次数或 `_dtcov_epoch` / `hj_epoch` 猜测 epoch。

冻结语义：

- plain e1–e200：始终无 DT/HJ/HNEK；
- HJ：从 `pre_e5` 分叉，物理 e5 的第一个 batch 开始启用；
- DT：从 `post_e20` 分叉，物理 e21 的第一个 batch开始启用；
- HNEK：从 canonical `pre_e1` 分叉，物理 e1 开始启用。

### 4.3 共同随机性与数据顺序

- 各分叉从完整锚点恢复相同 sampler/RNG/global step；
- 方法的辅助 MC、反事实探针、controls 和诊断必须 snapshot/restore RNG，不得改变下一主训练 step 的随机流；
- 每个主训练 step 记录 input stem、unpaired B stem、domain、time index、bridge noise hash、z hash、PatchNCE sample-id hash、sampler position；
- 方法分叉与 canonical plain 在相同相对 step 上必须拥有相同的数据身份与训练随机 bundle hash；参数不同不影响随机账本比较；
- 禁止用 retry-until-match 建立轨迹。

### 4.4 固定目标盲诊断面板

从 hash-locked training manifest 内按 domain 分层、stem SHA-256 排序后，用 seed=20260824 选择：

- 每域 16 个 A/source 身份；
- 每域 16 个独立的 unpaired B 身份；
- 覆盖训练制度中的全部六域；
- 面板允许与 training manifest 重叠，因为它只承担无配对 schedule 诊断，但诊断执行不得提交梯度或 optimizer state；
- 严禁读取这些 A 的 same-stem target；
- 面板 manifest 在任何模型 effect 读取前冻结。

所有 bootstrap 使用 source identity nested in domain 为 cluster，六域等权聚合，固定 999 draws；控制器使用单侧 95% bootstrap bound。bootstrap seed 由 `(run_id, method, epoch, statistic)` 的 SHA-256 前 63 bit 唯一生成，不得使用 Python `hash()`。

### 4.5 目标访问闸门

创建 `TRAINING_FROZEN.ok` 前：

- DataLoader、诊断、controller、profile、smoke 和 runner 全部只可访问 training manifest 的 unpaired A/B；
- 文件访问 ledger 必须记录 exact path、stem、role、purpose；
- same-stem target、paired-development target、test、confirmation 路径均由 guard 主动拒绝；
- 负向测试必须证明伪装文件名、软链接、相对路径逃逸和错误 purpose 都被拒绝。

只有所有训练 lane 到达 e200 或按本文形成合法 lane-local failure、checkpoint 全部 hash-lock、controller state 冻结后，runner 才写 `TRAINING_FROZEN.ok` 并启动独立 paired evaluator。

---

## 5. 统一 intervention controller 接口

实现可序列化的统一接口：

```text
observe(epoch, lane_state, canonical_plain_state, diagnostic_manifest)
    -> SignalRecord

decide(history)
    -> ACTIVE | HANDOFF | OFF

state_dict() / load_state_dict()
```

要求：

- `observe` 不推进训练 RNG、optimizer、scheduler 或 sampler；
- `SignalRecord` 包含全部 raw unit values、floor、control、bootstrap draw identity、CI、连续计数和决定原因；
- `decide` 是纯函数；相同历史输入必须得到相同结果；
- 状态恢复后下一次 `observe/decide` 与不中断轨迹 bitwise 一致；
- OFF/HANDOFF 是单向状态，不允许重新激活；
- 科学信号失效是自动状态转换，不是作者闸门。

---

## 6. DT-CovMatch：固定教师与响应式退出

### 6.1 保留的算法

不修改当前 clean DT 核心：

- 按 domain 分组；
- 每组 M=4 endpoint proposals；
- 方向 `D=(Y-X_t)/(1-t_norm)`；
- signal-normalized region disagreement；
- log floor；
- domain×time 标准化；
- Smooth-L1；
- grouped-domain 等权；
- additive regularizer；
- eval-off。

固定：

```text
base lambda = 0.001
active age 1..5   = linear ramp
active age 5..15  = hold
active age 15..25 = cosine decay
```

### 6.2 教师身份

- DT teacher 必须由 canonical `post_e20` 的 netG state 构造；
- teacher checkpoint path、checkpoint SHA、netG state SHA 写入 frozen spec；
- teacher 创建后逐参数冻结，永不更新；
- 禁止 first-use 当前网络 deep-copy；
- 如果加载后 teacher state SHA 不等于 canonical post-e20 netG SHA，立即 lane hard stop；
- 训练损失仍保留当前 domain×time EMA 语义，不把诊断 reference bank 偷换进训练公式。

### 6.3 独立诊断 reference bank

在 canonical `post_e20` 和冻结诊断面板上：

- 每个 `(domain,time)` 使用 4 个独立 diagnostic repeat bundle；
- 每个 repeat 内 M=4；
- 固定 teacher `log U` 的 mean/scale；
- 用等价重复之间的距离构造 `MC_floor_DT`；
- floor 取同层次 null 距离分布的 99% 分位；
- reference bank 只用于 controller，不参与 DT loss 的反传和在线 EMA。

### 6.4 目标盲统计量

DT 从物理 e21 开始计 active age；在 active age `2,4,6,...,24,25`（对应物理 e22、e24、e26……e44、e45 结束）审计，定义：

```text
E_DT(e) = 六域等权 median[
  distance(logU_lane(e), teacher_reference) - MC_floor_DT
]

R_DT(e) = E_plain(e) - E_DT(e)
```

`E_plain(e)` 必须由 canonical plain 同 epoch checkpoint、同面板、同 diagnostic random bundle 计算。正的 `R_DT` 才表示 DT 比自然训练更有效地减少该教师分歧。

保存 `E_DT`、`E_plain`、`R_DT` 的 raw cluster values、999 draws 和单侧 95% bounds。

### 6.5 自动退出

DT 至少完成 5 个 active epoch；只有 active age >=6 的审计才进入连续计数。在此之后，满足任一条件时永久 OFF：

1. 连续 2 次审计中，`E_DT` 的 95% 上界不高于 0（因为 `E_DT` 已减去 `MC_floor_DT`）：`DT_SIGNAL_EXHAUSTED`；
2. 连续 3 次审计中，`R_DT` 的 95% 下界不大于 0：`DT_NO_TARGET_BLIND_RESPONSE`；
3. 非有限、teacher hash 改变、诊断无法纯函数复算：`DT_ENGINEERING_LANE_STOP`。

若均未触发，active age 25 后按原 schedule 归零。归零后从当前 DT 参数状态继续 plain objective 到 e200，不重新激活。

本轮不搜索新 lambda、M、floor、teacher epoch 或 domain weighting。

---

## 7. HJ-PatchNCE：结构信号存活门

### 7.1 保留的算法

固定为当前 clean HJ 最小核心：

```text
layer = 0
structure = joint edge + SSIM
probe = central_consensus
strength = 0.5
gate_quantile = 0.75
min_risk = 0.05
boundary_scale = 0.001
update_mode = remove
physical start epoch = 5
```

不搜索新 layer、direction、strength、quantile 或 proxy。HJ forward 必须与原 PatchNCE forward 完全一致，只允许 backward 投影变化。

### 7.2 虚拟一步诊断

从 canonical `pre_e5` 分叉，HJ 在物理 e5 第一个 batch 启用。在物理 `e10,e20,...,e200` 的 epoch 结束后，于冻结面板执行一次不提交的 virtual generator step：

1. `RAW`：完整原 G objective，无 HJ projection；
2. `TRUE`：唯一变化是使用 true HJ projection；
3. `ROLL`：相同 gate/dose，结构方向做冻结的 spatial roll control；
4. `SIGN`：相同 gate/dose，结构方向做冻结的 sign control。

四个虚拟 step：

- 从完全相同的 generator/F/D/E state 与 RNG bundle 开始；
- 使用当前实际 generator learning rate；
- G objective 包括 GAN、SB 和 PatchNCE，唯一差异是 PatchNCE backward 方向；
- D/E/F 在诊断 virtual step 中不更新；
- 可使用 stateless functional call 或完整 snapshot/restore；
- 结束后全部训练 state 必须恢复为审计前 hash。

在同一个 source-only 面板上重新生成输出，用当前 clean HJ 的 joint edge+SSIM source structure functional 计算：

```text
H_variant = L_structure(after virtual step) - L_structure(before)
V_HJ = H_RAW - H_TRUE
C_ROLL = V_HJ - (H_RAW - H_ROLL)
C_SIGN = V_HJ - (H_RAW - H_SIGN)
```

`V_HJ>0` 表示 true projection 避免了 raw step 导致的结构恶化；`C_ROLL/C_SIGN>0` 表示该效果不是相同剂量的 control 即可复制。

同时记录：

- one-sided 与 central probe 的逐 unit 符号一致率；
- risk/gate mass；
- removed-gradient norm ratio；
- direction norm；
- raw/true/control full G gradient hash 与有限性。

### 7.3 effect-blind null

在 `pre_e5` 上用等价重复、零 strength 和 identity direction 构造 null：

- probe agreement 阈值取 null 99% 分位；
- virtual-step 数值差异 envelope 取 null 99% 分位；
- 所有 null 在 HJ 参数 effect 和 paired target 读取前冻结。

### 7.4 信号有效与自动退出

一次 HJ 审计只有同时满足以下条件才记为 `VALID`：

1. one-sided/central 符号一致率高于冻结 null 99% 分位；
2. `V_HJ` 的单侧 95% 下界大于 0；
3. `C_ROLL` 与 `C_SIGN` 的单侧 95% 下界都大于 0；
4. gate mass、direction norm、removed-gradient ratio 和所有 loss/gradient 均 finite；
5. snapshot/restore hash 完全一致。

HJ 至少运行到物理 e15；只有 `e20` 及之后的审计进入连续计数。因此最早在 e30 的第二个连续无效审计后 OFF。连续 2 次审计不是 `VALID` 时，strength 永久设为 0：

```text
HJ_SIGNAL_NOT_ALIVE -> OFF
```

OFF 后从当前 HJ 参数状态继续原始 plain objective 到 e200，不重新激活。

废止当前仅依据 `conflict_ema/conflict_peak` 的 adaptive schedule；它可以作为描述性日志，但不得控制本轮 strength。

---

## 8. HNEK：FULL 与目标盲 HANDOFF

### 8.1 唯一方法配置

只允许：

```text
model = hnek_search
gamma = 0.25
coord = residual
horizon_mode = physical
partial = all
seed = 2026
```

禁止使用 legacy `--model sb --hnek true`，禁止重跑 `coord_y` 或新增 gamma/partial 变体。

### 8.2 ON/OFF 实现

为 HNEK search wrapper 实现可恢复、无新参数的状态开关：

- `ON`：当前 `gamma=0.25/residual/physical/all` forward、E loss、G loss；
- `OFF`：调用原始 plain generator forward、原始 E loss、原始 G loss；
- ON/OFF 不增加或删除 state-dict key，不重建 optimizer/scheduler，不改变已有参数；
- OFF 模式在同一状态、同一 RNG 下必须与 plain SBModel forward/loss/generator gradient bitwise 一致；
- `hnek_active` 和 handoff epoch 必须进入 checkpoint/controller state。

### 8.3 跨时间信号

在冻结面板上，对每个 source、domain 和所有非终端桥时间使用 common random numbers，计算：

```text
h(t) = physical remaining horizon
R_gamma(t) = (Y_t - X_t) / h(t)^gamma
```

终端 `h=0` 只做 identity control，不进入除法和 primary statistic。

用相邻非终端时间的 unbiased energy distance 定义：

```text
C_H(e) = 六域等权 sum_j ED(R_gamma(t_j), R_gamma(t_{j+1}))
         - repeat_floor_H
```

`repeat_floor_H` 由同一 checkpoint、相同状态、独立 diagnostic repeat 的等价估计构造，取 null 99% 分位。

### 8.4 HNEK 与 counterfactual plain 虚拟更新

在同一 HNEK lane state 上，使用同一 batch/RNG 做两个不提交的完整 G virtual step：

- `H_STEP`：HNEK ON objective；
- `P_STEP`：暂时 HNEK OFF 的 counterfactual plain objective。

定义：

```text
DeltaC_H = C_H(before) - C_H(after H_STEP)
DeltaC_P = C_H(before) - C_H(after P_STEP)
B_H(e)   = DeltaC_H - DeltaC_P
```

并用 counterfactual plain G objective 检查 H_STEP 相对 P_STEP 的安全差异。安全 envelope 来自两个等价 plain virtual step 的 effect-blind 99% null，不使用手调百分比。

### 8.5 自动 handoff

在 `e10,e20,...,e200` 的 epoch 结束后审计；只有 e30 及之后的审计进入 handoff 连续计数。因此默认 cadence 下最早在 e40 冻结 `e_star`。连续 2 次满足任一条件时，冻结首个目标盲 handoff epoch `e_star`：

1. `C_H` 的 95% 上界不高于 0（因为 `C_H` 已减去 `repeat_floor_H`）：`HNEK_SIGNAL_EXHAUSTED`；
2. `B_H` 的 95% 下界不大于 0：`HNEK_NO_LONGER_BEATS_PLAIN_UPDATE`；
3. H_STEP 对 counterfactual plain objective 的恶化超过冻结 null envelope：`HNEK_NATIVE_SAFETY_LOST`；
4. 非有限或 snapshot/restore 失败：对应 lane engineering stop。

HNEK 主训练先形成一条 `HNEK_FULL`：始终 ON 到 e200。第一次产生 `e_star` 时，从该 full-state 分叉一条 `HNEK_HANDOFF`，立即 OFF，并以 plain objective 继续到 e200。

若到 e200 仍没有目标盲 handoff 条件，记录 `HNEK_HANDOFF_NOT_TRIGGERED`，不得人为选择 paired 最佳 epoch。

预算允许时，额外从 HNEK e50 full-state 分叉：

```text
HNEK_E50_HANDOFF
```

它在 e50 固定 OFF 并以 plain objective 继续到 e200，只是检验历史 e50 峰值线索的 saturated-development control，不是无偏算法选择。

---

## 9. paired-development：只能在训练冻结后读取

### 9.1 解锁条件

只有以下全部成立才写 `TRAINING_FROZEN.ok`：

- canonical plain 到 e200；
- DT lane 到 e200或形成合法 lane-local engineering stop；
- HJ lane 到 e200或形成合法 lane-local engineering stop；
- HNEK FULL 到 e200，已触发的 HANDOFF lanes 到 e200；
- 所有 checkpoint hash index、controller state、代码/spec/run 身份冻结；
- training access ledger 证明 target read count 为 0；
- 不再允许修改算法、controller、阈值或训练代码。

### 9.2 评估对象

只使用 hash-locked 的既有 paired-development manifest，不创建新 split。评估：

- canonical plain 的全部保存 checkpoint；
- DT/HJ 的全部保存 checkpoint和 e200；
- HNEK FULL 的全部保存 checkpoint；
- HNEK HANDOFF / E50_HANDOFF 的 handoff checkpoint 与 e200。

对每个 checkpoint 输出：

- 每图 PSNR、SSIM；
- 五域既有 paired-development 的域均值和 macro mean；
- 相对同 epoch canonical plain 的逐图配对差；
- 999-draw image-cluster bootstrap CI；
- 正域数；
- checkpoint、netG、manifest、代码和 evaluator hash。

RainDS-syn 如不在既有 paired-development primary manifest 中，不得临时补入；只在身份账本中保留“训练覆盖、确认覆盖缺失”的边界。

### 9.3 机械标签

这些标签只用于整理，不触发本轮追加实验：

DT/HJ 相对 canonical plain：

```text
DEVELOPMENT_GAIN
  iff macro PSNR delta >= +0.15 dB
  and paired bootstrap lower bound > 0
  and positive domains >= 3/5
else DEVELOPMENT_NO_GAIN
```

HNEK HANDOFF 相对 FULL 的“停止优化成功”：

```text
HANDOFF_OPTIMIZATION_GAIN
  iff handoff e200 - full e200 >= +0.10 dB
  and handoff e200 - canonical plain e200 >= +0.15 dB
  and both relevant paired lower bounds > 0
  and handoff positive domains vs plain >= 3/5
else HANDOFF_OPTIMIZATION_NO_GAIN
```

同时报告无配对 controller 信号与下一保存区间 paired delta 之间的 Spearman 和符号一致率，但它们是描述性验证，不允许用于回头修改规则。

明确 claim boundary：单 seed=2026、saturated paired-development、非 confirmatory、不包含训练 seed 不确定性。

---

## 10. effect-blind 工程门、自审和冻结

### 10.1 CPU/静态测试

在任何真实 effect 训练前完成：

- 全量 Python 语法检查；
- 当前仓库既有 pytest；
- clean-reexploration 新单元测试；
- JSON canonicalization、seed64、manifest、访问 guard、controller 状态机、bootstrap 纯函数测试；
- 错 run_id/spec/checkpoint/teacher/manifest 的负向拒绝测试。

### 10.2 真实模型语义测试

必须证明：

1. physical e4 HJ OFF，physical e5 第一个 batch HJ ON；
2. physical e20 DT OFF，physical e21 第一个 batch DT ON；
3. DT teacher netG hash 等于 canonical post-e20 netG hash；
4. DT `lambda=0` 与 plain forward/loss/全部 G gradient/一次 optimizer update bitwise 一致；
5. HJ `strength=0` 与 raw PatchNCE forward/loss/全部 G gradient/一次 optimizer update bitwise 一致；
6. HJ true projection forward exactly equal raw；
7. HNEK OFF 与 plain forward、E/G loss 分项、全部 G gradient bitwise 一致；
8. HNEK ON/OFF 不改变参数数目和 state keys；
9. 所有 diagnostic snapshot/restore 后 canonical training state hash 不变；
10. target/confirmation/test 的正负访问测试全部通过。

### 10.3 确定性后端选择

在 import torch 前设置：

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=2026
```

首选 `STRICT_CUDNN`：

- `torch.use_deterministic_algorithms(True, warn_only=False)`；
- deterministic debug mode `error`；
- cuDNN deterministic=True、benchmark=False；
- TF32 全部关闭；
- 从同一 disposable anchor 建两条 identical plain twin；
- 100 个训练 step 每步 canonical training state bitwise equal；
- 保存第 50 step full-state，在独立进程恢复并证明第 51–100 step 与不中断轨迹 bitwise equal。

若首选失败，保留失败证据，只允许一次 fallback：`STRICT_NATIVE_NO_CUDNN`。关闭 cuDNN 后重建 disposable twins，执行同样 100-step 与跨进程恢复门。

禁止接受 `1e-3` 或其他数值容差。两条后端都失败则 `HARD_STOP_DETERMINISM`。

后端选择完全 effect-blind。通过后销毁 disposable twins，从原始 canonical initialization 重新加载正式 lane。

### 10.4 第二遍独立自审

工程门通过后、冻结 spec 前，服务器 Agent 必须停止写代码并做一次独立 red-team：

- 重新阅读本文与实际 `git diff`；
- 检查是否存在隐式 epoch、first-use teacher、variant-specific plain、target proxy 泄漏、Python hash、last-write 聚合、NaN 置零、错误 permutation tail、空 predictor、软链接逃逸、checkpoint identity 回退；
- 从 raw synthetic fixtures 手工构造每个 controller 的 ACTIVE、OFF、HANDOFF 正负样例；
- 确认 mechanical adjudicator 能检出旧 H2/H2C 类型的实现错误；
- 运行 production audit；
- 把审计写入 `state/PRE_EFFECT_RED_TEAM.md/json`。

发现工程缺陷时允许 effect-blind 修复，最多 2 个修复 commit；每次修复后重跑全部测试和 red-team。同一根因第三次出现则 hard stop。任何真实 method effect 或 paired effect 读取后不得修科学实现；只能停止并打包。

### 10.5 冻结

自审通过后生成：

```text
算法设计模块/code/clean_reexploration/frozen/CLEAN_REEXPLORATION_FROZEN_SPEC.json
runtime_4090/clean_reexploration_20260824/authority/CODE_SHA256.txt
runtime_4090/clean_reexploration_20260824/authority/SPEC_CANONICAL_SHA256.txt
runtime_4090/clean_reexploration_20260824/authority/RUN_ID.txt
```

run_id 格式：

```text
clean-reexploration-s2026-20260824-<spec_sha前16位>
```

冻结后 runner/worker/checkpoint/evidence 必须显式传入并三重校验 run_id、spec SHA、code SHA；禁止 silent fallback 到旧 spec。

---

## 11. 单一长程 runner 与恢复

### 11.1 唯一启动入口

实现：

```bash
bash LAUNCH_4090_CLEAN_REEXPLORATION.sh
```

launcher 必须：

- 自动建立或复用名为 `unsb_clean_reexploration_20260824` 的 tmux session；
- 启动 `run_long.py`；
- 立即返回 tmux 名称、日志和 heartbeat 路径；
- 重复运行时检测现有 state，并安全 resume/skip 已完成阶段；
- 不因启动命令返回而宣布任务完成。

heartbeat 每 5 分钟更新：当前阶段、lane、epoch、global step、最近 checkpoint、GPUh、wall-clock、磁盘、预计剩余时间和最后成功审计。

### 11.2 固定执行顺序

runner 依次执行：

1. protection / authority / data identity；
2. CPU、真实模型、确定性、resume、自审和冻结；
3. 实际 profile 与预算模型；
4. canonical plain e1→e200；
5. HNEK FULL，以及触发后的 HANDOFF、预算允许时 E50_HANDOFF；
6. DT post-e20→e200；
7. HJ pre-e5→e200；
8. 写 `TRAINING_FROZEN.ok`；
9. paired-development evaluator；
10. mechanical adjudication、纯函数复算、production audit；
11. 唯一打包与 fresh-directory 验收。

阶段科学结果不得改变后续顺序。DT/HJ 信号失效后只是该 lane 改为 plain continuation，HNEK HANDOFF 与 FULL 按计划继续。

### 11.3 恢复

- 每个 checkpoint 使用临时文件、fsync、原子 rename、payload SHA 和 sidecar；
- 恢复必须校验全部组件、身份、device count、sampler、controller、RNG 和 global step；
- net-only checkpoint 禁止作为恢复点；
- 中断后从最近合法 full-state 自动恢复，不触发作者闸门；
- 恢复后第一步的 batch/random bundle/decision 必须与中断前预期一致；
- 失败 attempt 永久保留在 attempt ledger，不覆盖。

---

## 12. 48 GPU 小时预算与自动降本

### 12.1 预算

```text
工程门与 profile                <= 2 GPUh
canonical plain                <= 6 GPUh
HNEK FULL/HANDOFF              <= 15 GPUh
DT                              <= 10 GPUh
HJ                              <= 12 GPUh
paired eval / packaging         <= 3 GPUh
总 hard budget                  <= 48 GPUh
wall-clock hard budget          <= 72 h
新增磁盘 hard budget            <= 120 GiB
```

profile 用真实模型和实际选定后端，覆盖 plain、DT active、HJ active、HNEK ON、三种诊断和评估。总预测包含 20% 安全系数。

### 12.2 唯一降本顺序

若 profile 预测总 GPUh >46：

1. 诊断面板每域从 16 A/16 B 降到 8 A/8 B；
2. DT 审计 cadence 从每 2 epoch 改为每 5 epoch；
3. HJ cadence 从每 10 epoch 改为每 20 epoch；
4. HNEK cadence 从每 10 epoch 改为每 20 epoch；
5. 重新 profile 和预算。

若仍 >46：取消 `HNEK_E50_HANDOFF`，保留 HNEK FULL 和目标盲 HANDOFF，再预算。

若仍 >48：触发 `HOLD_COST_BEFORE_EFFECT`。不得降低训练 epoch、删除 DT/HJ/HNEK 主路线、减少 M=4、改算法或偷偷减少数据。

运行中 GPUh 达 48、wall-clock 达 72h 或磁盘达到 120 GiB 时，安全保存当前 full-state，记录 `HARD_STOP_BUDGET` 并进入失败证据打包。

---

## 13. Hard stop、lane stop 与不应打扰作者的情况

### 13.1 全局 hard stop

仅限：

- `HARD_STOP_MISSING_BASE_AUTHORITY`
- `HARD_STOP_GIT_OR_CODE_IDENTITY`
- `HARD_STOP_DATA_MANIFEST`
- `HARD_STOP_TARGET_ACCESS`
- `HARD_STOP_DETERMINISM`
- `HARD_STOP_GLOBAL_NONFINITE`
- `HOLD_COST_BEFORE_EFFECT` / `HARD_STOP_BUDGET`
- 同一工程根因第三次出现
- return production audit 无法闭合

全局 hard stop 也必须执行保护、报告、轻量证据打包；不得只在终端留一句报错。

### 13.2 lane-local stop

单个 DT/HJ/HNEK lane 出现其专属非有限、snapshot 失败或 checkpoint 损坏时：

- 保存最近合法 checkpoint 和失败证据；
- 该 lane 标记 `ENGINEERING_LANE_STOP`；
- 其余 lane 继续；
- 不允许服务器为救该 lane 修改科学代码；
- 最终统一回传。

若相同故障同时影响 canonical plain 或两个以上方法，升级为全局 hard stop。

### 13.3 不触发作者闸门

以下情况都不得中途询问作者：

- 普通 warning；
- 单次进程/GPU/SSH/tmux 中断；
- 已有 cache 或 checkpoint；
- DT/HJ/HNEK 信号消失；
- paired PSNR 为负；
- 机械标签为 NO_GAIN；
- HNEK HANDOFF 未触发；
- 某个方法自动 OFF 后继续 plain；
- 服务器认为论文动机需要修改。

---

## 14. 机械裁决和服务器允许写的内容

生成：

```text
MECHANICAL_SUMMARY.json
FINAL_REPORT.md
DECISION_LOG.md
ATTEMPT_LEDGER.json
CHECKPOINT_SHA256.txt
ACCESS_LEDGER.csv
CODE_PROVENANCE.json
```

`adjudicate.py` 必须只从 raw evidence 纯函数生成 summary。fresh-directory 验收时重跑，数值和标签必须一致。

服务器最终只允许使用以下六栏：

1. **事实**：身份、工程门、运行规模、controller 状态、指标；
2. **工程失败**：实际发生的实现/环境/恢复问题；
3. **机械结果**：本文冻结标签，不扩展论文含义；
4. **未决**：单 seed、paired-development、缺失覆盖等边界；
5. **下一动作**：仅写“等待作者本地审查回传包”；
6. **是否触发作者闸门**：任务完成或 hard stop 后唯一一次触发。

不得写：

- “算法已立住 / 已失败 / 已证明创新”；
- “可投 ICLR / 不可投 ICLR”；
- “SB 特异性已经确认”；
- “跨 seed 稳定 / 泛化 / robust”；
- “建议服务器再试一个 lambda/epoch/seed/variant”；
- 任何不由 frozen mechanical rules 直接产生的科学裁决。

---

## 15. 唯一回传与 fresh-directory 验收

### 15.1 最终文件

无论完成、FAIL、HOLD 或 hard stop，只生成：

```text
DTHJ_HNEK_CLEAN_REEXPLORATION_RETURN_20260824.zip
DTHJ_HNEK_CLEAN_REEXPLORATION_RETURN_20260824.zip.sha256
```

### 15.2 ZIP 必须包含

- 本 Prompt 原文；
- frozen spec；
- 实际执行代码和测试；
- Git/环境/数据/authority 身份；
- pre-effect tests、determinism、resume、red-team 和 production audits；
- controller raw signals、null floors、controls、bootstrap draws；
- paired evaluator 的逐图指标和汇总；
- access ledger、attempt/decision ledger；
- logs 和 heartbeat 摘要；
- checkpoint/netG hash index；
- adjudicator、mechanical summary 和 final report；
- 内部 `MANIFEST.sha256`，覆盖除自身外所有 payload。

### 15.3 ZIP 禁止包含

- 训练/评估图像像素；
- checkpoint 本体；
- `.git`；
- `__pycache__`、`.pyc`、`.deps`、环境目录；
- 嵌套 ZIP、Git bundle、大型 cache；
- 未在 manifest 中的临时文件。

### 15.4 验收

在全新临时目录解压并检查：

- sidecar 与 ZIP SHA 一致；
- 内部 manifest 全覆盖且零错配；
- JSON/CSV/Markdown UTF-8 可读；
- 所有相对路径有效；
- 无禁止内容；
- raw 行数、checkpoint hash、access count 和 controller audit 数量一致；
- 从 shipped raw 重跑 tests、adjudicator 和 production audit，结果与 shipped summary 一致；
- paired access 只发生在 `TRAINING_FROZEN.ok` 之后；
- 所有路线只引用一个 canonical plain identity。

ZIP 在 staging 目录之外生成；ZIP 完成后再生成外部 sidecar。不要手工压缩一个未经验收的文件夹冒充最终回传。

---

## 16. 任务启动后的沟通合同

作者启动任务后，你应先回复：

```text
已读取并接受 4090_DTHJ_HNEK_CLEAN_REEXPLORATION_LONG_TASK_PROMPT_CN_20260824.md。
我将按单次长任务执行：保护与实现 → effect-blind 自审冻结 → 4090 长程运行 → 训练冻结后 paired evaluation → 唯一 ZIP 回传。
除合同定义的全局 hard stop 外不新增作者闸门，不自行修改科学设计。
```

随后直接工作。普通进度只写日志和 heartbeat，不需要作者确认。任务完全结束或发生合同定义的全局 hard stop 后，才发送唯一六栏汇报和两个最终文件的路径、字节数、SHA-256。

如果你发现本文某个实现细节需要读取源码才能落到具体函数名，应按第 1 节权威顺序完成最小接口适配；这不构成重新设计算法，也不触发作者闸门。若源码事实与本文的科学语义直接矛盾，保留证据并触发 `HARD_STOP_CONTRACT_SOURCE_CONFLICT`，不得自行选择替代语义。

---

## 17. 最终声明

本任务不是新的假设筛选轮，也不是让服务器 Agent 为已有收益寻找解释。它是一轮受控的工程再开发：用唯一 plain 主干消除对照漂移，用显式 physical epoch 消除启用错位，用 hash-locked teacher 消除 DT 身份漂移，用目标盲 controller 检查方法信号何时消失，并用 HNEK FULL/HANDOFF 区分长期干预和短程桥坐标预条件。

只执行本文，不追加实验，不中途请求普通科学选择，不在服务器写论文结论。
