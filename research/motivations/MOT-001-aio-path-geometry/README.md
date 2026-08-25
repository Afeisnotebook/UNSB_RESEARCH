# MOT-001：All-in-One 条件路径几何

> **当前中文阅读入口：** [READING_GUIDE_CN.md](./READING_GUIDE_CN.md)。它统一解释五域多 seed 证据、六域扩大实验、共享时钟遗憾、头图各面板及允许/禁止表述。

> 本 README 的“唯一对照”和五域内容描述最初的多 seed 主线；后续六域扩展恢复了 RainDS-syn，但仍属于同一个 MOT-001 观察性问题。两条证据链的关系以阅读指南和最新 `decisions/CURRENT.md` 为准。

本模块用于回答一个论文动机问题：

> 同一个 plain UNSB，从“每个天气域独立训练（Single-task）”扩展为“五个天气域共享一个模型（All-in-One）”之后，条件恢复方向的几何是否发生阶段依赖的变化？

我们不做新方法，也不做性能优化，只重建一条**纯基线动机证据链**。

## 唯一对照

- Single-task：5 个相互独立的 plain UNSB；
- Plain All-in-One：5 个域数据并集训练 1 个共享 plain UNSB。

天气域固定为 5 个（已去掉 RainDS-syn）：

FoggyCityscapes、LowLightTrafficData、RainCityscapes、RSCityscapes、SnowTrafficData。

DT 只作为后置 sanity check，HJ 不进入动机图。

## 目录结构

```text
research/motivations/MOT-001-aio-path-geometry/
├─ README.md / motivation.json        # 当前入口与机器可读状态
├─ protocols/                         # 冻结协议
├─ src/                               # 训练、测量、窗口审计、机制量脚本
├─ figures/                           # 精选主图与窗口审计图
├─ CLAIM_LEDGER.json                  # 主张账本（每条主张的状态与证据）
├─ MOTIVATION_FROZEN_SPEC.json        # effect-blind 冻结协议
├─ MEASUREMENT_MANIFEST.json          # 测量图像清单
├─ PATH_MAP.json / CODE_IDENTITY.json # 路径与源码身份
├─ DATA_MANIFEST.csv/.json            # 五域数据身份清单
└─ CHECKPOINT_INDEX.json / MANIFEST.sha256
```

完整报告和运行证据不再与动机定义混放，位于：

- `experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/`
- `experiments/L1-local/EXP-L1-MOTIVATION-HEADFIGURE-20260824/`
- `experiments/L2-medium-4090/EXP-L2-MOTIVATION-SIXDOMAIN-20260824/`

## 我们做过的探索

### 1. 纯基线重启与主测量（seed=2026）

- 重建干净数据目录和冻结协议；
- 训练 5 个 Single、1 个 AIO plain、1 个 AIO DT；
- 测量 panel b/c/d/e，核心量是方向分散度 `U` 和空间方向分散度 `U_reg`。

### 2. 固定窗口审计（seed=2026/2027/2028）

- 单看 seed=2026 时，Epoch 4–5 出现“AIO 更压缩”的负向窗口；
- 补 seed=2027、2028 后，seed=2028 在 Epoch 3–6 翻正；
- 审计认定 seed=2028 是真实 seed 差异，不是技术异常。

### 3. 机制量筛选与升级

- 先做方向秩、方向 cosine 代理、特征 CKA、综合 compression score；
- 结论：不能把“早期过度压缩”归因到共享表示冲突或瓶颈；
- 随后升级为真实跨域参数梯度冲突、M=64 多图方向谱结构。

### 4. 固定窗口终局投票（seed=2029/2030/2031）

- 再训练并测量 3 个新 seed，只做 Single 与 AIO plain；
- 结论：固定 Epoch 4–5 窗口没有立住，判定关闭。

## 取得的关键结论

最稳定的现象是：

> Epoch 1 的 AIO 初始方向发散，在 seed=2026/2027/2028/2029/2030/2031 中全部一致为正。

其它结论：

- Single 与 AIO 的条件方向几何确实不同，但差异具有阶段依赖、会发生符号反转、且因域而异。
- 曾想锁定的“Epoch 4–5 过度压缩窗口”不可复现：seed=2028、2031 翻正，seed=2030 只有 3/5 域同号。
- DT 作为路径尺度干预降低 `U` 的 sanity check：NOT_SUPPORTED。
- 局部结构冲突主张没有成立，不再作为必要性证据。
- `U / U_reg` 只能解释为“方向分歧/空间方向分散程度”，不是校准不确定性，也不是压缩的因果证据。
- 本动机证据本身**不足以指定唯一算法靶点**；候选推进必须依赖独立实验与决策。

## 如何看这个项目

建议按以下顺序读：

1. [WINDOW_FINAL_VOTE_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/WINDOW_FINAL_VOTE_CN.md)：最新终局投票，固定窗口关闭的最终结论。
2. [CLAIM_LEDGER.json](CLAIM_LEDGER.json)：每条主张的当前状态和证据文件。
3. [FINAL_GATE_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/FINAL_GATE_CN.md)：算法设计前门禁。
4. [WINDOW_NOT_SETTLED_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/WINDOW_NOT_SETTLED_CN.md) 和 [WINDOW_DECISION_REPORT_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/WINDOW_DECISION_REPORT_CN.md)：窗口未敲定的记录与窗口边界分析。
5. [SEED2028_AUDIT_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/SEED2028_AUDIT_CN.md)：seed=2028 反转审计。
6. [MOTIVATION_FIGURE_REPORT_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/MOTIVATION_FIGURE_REPORT_CN.md) 与 [MOTIVATION_FIGURE_INTERPRETATION_CN.md](MOTIVATION_FIGURE_INTERPRETATION_CN.md)：主图与可视化结果解释。
7. [MECHANISM_DESIGN_PLAN_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/MECHANISM_DESIGN_PLAN_CN.md)、[MECHANISM_SCREEN_REPORT_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/MECHANISM_SCREEN_REPORT_CN.md)、[MECHANISM_UPGRADE_REPORT_CN.md](../../../experiments/L1-local/EXP-L1-MOTIVATION-WINDOW-20260824/evidence/mechanism_upgrade/MECHANISM_UPGRADE_REPORT_CN.md)：机制量设计与筛选结论。

原始证据图在 `figures/` 下，原始测量数据与 checkpoint 因体积较大未上传；如需复算，可参考 `PATH_MAP.json` 中记录的本地绝对路径。

## 新环境的路径约定

`PATH_MAP.json`、manifest 和 identity 文件中的 `/home/yc/...` 是历史服务器溯源，保留原值。`src/` 下的启动脚本默认使用当前解释器和 `foundation/canonical/src`；新运行的临时输出进入 Git 忽略的 `runs/`，不能原位覆盖冻结实验。

按需要显式设置：

- `UNSB_DATA_ROOT`：五域原始数据根目录（运行 `discover.py` 时必需）；
- `UNSB_BASELINE_ROOT`：baseline 目录，供机制量脚本使用；
- `UNSB_DTCOV_ROOT` / `UNSB_HJ_ROOT`：DT/HJ 候选源码目录；
- `UNSB_MOTIVATION_ROOT`：本模块的工作根目录；
- `UNSB_MOTIVATION_RUN_ROOT`：新运行的 checkpoint/raw/report/figure 输出根目录；
- `UNSB_BOOTSTRAP_ROOT`：历史测量参考源码根目录；
- `UNSB_PYTHON`（Python orchestration）或 `PY`（shell orchestration）：指定解释器。
- `GPU`：选择物理 GPU；脚本设置 `CUDA_VISIBLE_DEVICES=$GPU` 后始终把可见设备作为逻辑 `gpu_ids=0` 使用。

这些变量只解决路径身份；完整 GPU 重算仍需要未上传的数据与 checkpoint，并必须先核对 manifest/hash。
