# DEC-20260826: freeze SEARCH-001 local winner

## Decision

将 CAND-003 HNEK `gamma=0.25 + residual + physical + all` 冻结为 SEARCH-001 唯一本地候选，分类为 `positive_but_fragile`，下一门禁为 seed=2026 的 matched 4090 30k/60k/120k 验证。

LBST、PTQ、DCUM、AEB 的当前固定实现关闭为本轮负结果。HNEK+DCUM 和 DT 分别作为递补一、递补二；HJ 不因单个 1200-step 正点获得保护名额。

## Evidence

- [完整本地实验](../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/README.md)
- [唯一候选](../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/CANDIDATE.json)
- [机器可读逐域轨迹](../../experiments/L1-local/EXP-L1-SEARCH-001-DIRECTIONAL-20260826/RESULTS.json)

HNEK 完整视图复赛最后三点评分为 `+0.339392 dB`；延长后 8k/10k/12k 的 matched delta 为 `+0.392588 / -0.429242 / +0.055621 dB`，均值 `+0.006322 dB`。12k 时 3/6 域正、最差域 `-1.101658 dB`。该结果只证明“当前最值得继续投入”，不证明稳定正收益。

## Consequence

- 4090 阶段不得根据中间 paired PSNR 改 HNEK 配置或挑新窗口。
- confirmation20 保持封存，直到 4090 候选冻结后首次打开。
- 若 HNEK 在 4090 失败，按 HNEK+DCUM、DT 顺序验证；不回头围绕 DT/HJ/HNEK 做网格搜索。
- 本地结果只有 seed=2026，不允许用图像级统计替代后续训练 seed 稳定性。
