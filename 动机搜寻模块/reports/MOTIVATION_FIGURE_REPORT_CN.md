# UNSB 动机图纯基线重启 — 自动裁决草稿

- 生成时间（UTC）：`2026-08-18T22:18:33Z`
- 汇总文件：`/home/yc/UNSB_PASSION/motivation_baseline_restart/reports/MOTIVATION_SUMMARY.json`
- raw 行数：`6750`
- 汇总存在：`True`

> 这是**草稿**，不是最终科学结论。单 seed=2026，所有方向性结论必须先人工确认。

## 自动裁决结果

| claim_id | 状态 | 需人工确认 | 文本 |
|---|---|---|---|
| geometry_different_stage_dependent | 支持性筛查（单 seed/方向性，需人工确认） | 是 | Single 与 Plain All-in-One 的条件方向几何不同，且差异具有阶段依赖 |
| rain_and_multidomain_nonuniform | 支持性筛查（单 seed/方向性，需人工确认） | 是 | Rain/多域下路径几何发生改变，但并非全程同号 |
| dt_sanity_lower_U | 不支持 | 是 | DT 作为路径尺度干预 sanity check，相对 Plain AIO 降低 U |
| single_seed_scope | 支持 | 否 | 本旁路只允许 SUPPORTED / SUPPORTED_SCREEN 级别方向性措辞，禁止全称因果或校准结论 |

## 自动结果要点（仍需人工核对）

- 几何阶段依赖：状态 支持性筛查（单 seed/方向性，需人工确认）；sign_changes=True；ci_separation=True。
- Rain/多域非全程同号：状态 支持性筛查（单 seed/方向性，需人工确认）；sign_changes=True；ci_separation=True。
- DT sanity check：状态 不支持；fraction_dt_lower=0.0。

## 证据文件

- `raw/*.jsonl`：每 image / bridge-time 的 U、log U、u_map。
- `raw/panel_b_directions.npz` 与 `raw/panel_b_pca.json`：panel_b 联合 PCA。
- `reports/MOTIVATION_SUMMARY.json`：panel_c/d/e 汇总。
- `figures/panel_b.png`、`panel_c.png`、`panel_d.png`、`panel_e.png`。

## 作者睡醒后需要人工确认什么

1. 检查 `WAIT_MAIN`、GPU 占用与训练日志，确认没有抢占主线或静默降级。
2. 检查单 seed 轨迹：panel_c 中哪些域/哪个 bridge time 出现阶段反转或 CI 分离。
3. 检查 panel_e 的 paired bootstrap 是否只是少数图像/单一 bridge time 的结果。
4. 检查 panel_d 的 32px 区域图是否符合物理预期，且不是由显示尺度造成的伪影。
5. 检查 DT sanity check 是否只被用作后置机制一致性，而非主对照结论。
6. 逐条核对 `CLAIM_LEDGER.json` 中每个 `SUPPORTED_SCREEN`/`UNRESOLVED`/`NOT_SUPPORTED`/`FAILED` 项。
7. 禁止将结果改写成“AIO 全程更分散”“U 是校准不确定性”“HJ 修复局部冲突”。

## 结论保留边界

- 允许：Single 与 AIO 条件方向几何不同（stage-dependent）。
- 允许：Rain/多域下路径几何改变，但非全程同号。
- 允许：DT 作为路径尺度干预降低 U（仅机制一致性 sanity check）。
- 禁止：任何全称因果、校准 posterior/epistemic、AIO 全程更分散、HJ 修复局部冲突的表述。
