# 投稿级方法摘要与可复现清单

## 标题候选

Minimal Forward-Invariant Corrections to Amortized Endpoint Response Statistics for Unpaired Schrödinger Bridge Restoration

## 统一方法

在 unpaired UNSB 的 amortized 端点条件律上，对“局部响应统计”施加最小、前向不变、诊断量驱动的修正：

- DT：约束局部响应的分布尺度（端点 proposal 分歧 U），是对 amortized 端点条件律的有界函数正则/信任域一致性约束。
- HJ：约束局部响应的更新方向，保持 PatchNCE forward 不变，只对结构有害方向做梯度投影，并用可验证 gate 定位。

## 核心证据（单 seed=2026，干净框架）

> ⚠️ 单 seed 结果存在约 1 dB 运行间方差（`reflection_pad2d` backward 非确定），
> 下表的 DT 相对收益在噪声量级内，仅作方向参考；投稿级结论须多 seed（均值 + CI）。

| 证据 | 数值 |
|---|---|
| DT vs plain | +0.70 dB（3-seed 配对，95% CI [0.17,1.23]） |
| HJ true vs plain | +2.07 dB（3-seed，95% CI [−1.64,+5.78]，含 0） |
| HJ 归因（true vs roll） | 单 seed 四项 PASS；3-seed mean +1.49，CI [−1.37,+4.35]（含 0），**未跨 seed 复现** |
| DT knock-out | grouped_domain 等全部必要 |
| HJ knock-out | boundary/central_consensus/min_risk/strength 全部必要 |
| DT adaptive vs 手调 | +0.0458 dB（≥ 手调） |
| HJ adaptive vs 手调 | -0.6343 dB（全局强度自适应有害，应固定） |
| HJ risk amplitude vs 手调 | -1.7377 dB（per-location 剂量缩放也无效） |

## novelty 边界

不声称首次 preconditioning / gradient surgery / uncertainty 校准 / 新 reference / 新 noise schedule。贡献点是“UNSB 受限端点律 + unpaired EROT 上的最小前向不变修正 + 可验证 gate + 诊断量驱动的介入原则”。

## 可复现清单

- 环境：4090，torch 2.5.1+cu121，conda env `unsb_cov`。
- 数据：`final6_train160_test40_unpaired`（DT）与 `final6train_valO5x16_offset560_unpaired`（HJ）。
- 统一 harness：`refactor/harness/`（data/config/determinism/checkpoint/metrics）。
- 干净模型：`--model dtcov` / `--model hj`（baseline 注册）。
- 完整复现顺序与多 seed 注意：`refactor/_runs/REPRODUCE.md`。
- 结果 JSON：`refactor/_runs/metrics*/`、`metrics_abl/`、`metrics_adaptive/`、`metrics_hj*/`。
- 消融/归因/自适应结果：`refactor/_runs/*_RESULT*.md`、`ABLATION_RESULTS.md`。

## 仍缺（诚实）

- **多 seed / 更大规模复现**（用户将在更好的服务器自行跑）：这是给出 necessity 与收益稳定性结论的必要条件；单 seed 仅方向参考。
- HJ 的自适应已系统性证否：全局 strength 与 per-location risk amplitude 均劣于固定 0.5；局部性由 gate 保证，dose 应固定。
