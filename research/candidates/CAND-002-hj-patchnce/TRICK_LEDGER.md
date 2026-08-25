# HJ-PatchNCE trick 分类账

分类法见 `../CONTRACT.md`。

| 项目 | 原代码位置 | 作用 | 分类 | 理由 / 建议 |
|---|---|---|---|---|
| `structure_project` 路径 | `sb_model._calculate_structure_project_nce` | 最好分支唯一使用的机制 | necessary_algorithmic | 保留 |
| 结构切方向 `source_structure_direction` | `sb_model._source_structure_direction` | 供投影对齐的局部方向 | necessary_algorithmic | 保留 |
| edge/SSIM 梯度 | `_source_edge_gradient` / `_source_ssim_gradient` | 定义“结构方向” | necessary_algorithmic | 保留 |
| 反事实 ±step | `perturbed_tgt / opposite_tgt` | 估计 PatchNCE 方向敏感度 | necessary_algorithmic | 保留 |
| `central_consensus` | `probe_mode` | 双侧一致性，去掉单侧噪声 | necessary_algorithmic | 保留（最好分支用这个） |
| `project_conflicting_gradient` | `patchnce._ProjectConflictingGradient` | forward 恒等、backward 投影 | necessary_algorithmic | 保留 |
| `update_mode=remove` | `project_conflicting_gradient` | 只去掉冲突分量，不重分配 | necessary_algorithmic | 保留 |
| `gate_quantile=0.75` | `positive_quantile_gate` | 只投影 top-quantile 正冲突 | necessary_robustness | 保留 |
| `min_risk=0.05` | `apply_absolute_evidence_gate` | 过滤弱证据，避免全图微投影 | necessary_robustness | 保留 |
| `boundary_scale=0.001` | `correspondence_boundary_instability` | 边界不稳度调制 risk | necessary_robustness | 保留，但作用量级较小，可做 knock-out |
| `strength=0.5` | `project_strength` | 投影剂量 | necessary_algorithmic | 保留 |
| `scales=1,2,4` | `_parse_scales` | 多尺度结构定义 | necessary_robustness | 保留 |
| `step=0.01` | `structure_step` | 反事实扰动步长 | necessary_robustness | 保留 |
| `deterministic / deterministic_strict / no_flip` | train.py | 保证可复现 | necessary_robustness | 保留 |
| `nce_uncert_start_epoch=5` | schedule | 等 PatchNCE 特征稍微收敛再介入 | necessary_robustness | 保留 |
| `nce_structure_event_z=0` | `_structure_event_weight` | 事件门控，最好配置为 0（恒 1） | inertia_legacy | 删除 |
| `bridge_*` 系列 | sb_model | 未用模式 | inertia_legacy | 删除 |
| `structure_risk/softpos/posmargin/relational/pixel_project` | sb_model | 未用模式 | inertia_legacy | 删除 |
| `corr_*` / `curation` | sb_model + correspondence_uncertainty | 未用模式 | inertia_legacy | 删除 |
| `side-car netU` / proposal uncertainty | sb_model | 未用模式 | inertia_legacy | 删除 |
| 大量 `loss_NCE_*` 诊断字段 | sb_model | 显示/落盘 | hack_artifact | 移出核心 |

## 关键结论

1. 最好收益只来自 `structure_project` 单路径，其余是历史堆叠。
2. `central_consensus` 是最好分支的关键，单侧探针本身不足以稳定归因。
3. `boundary_scale` / `min_risk` 属于“量级较小的稳健化”，是否真的必要需 GPU knock-out 消融确认。
