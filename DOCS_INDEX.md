# 文档索引（区分“当前权威”与“历史/已废弃”）

## 当前权威（以这些为准）

- `PLAN.md`：唯一当前目标与阶段。
- `PROGRESS.md`：当前进度与最新结论。
- `STATUS.json`：机器可读运行状态（自动写）。
- `refactor/BASELINE_DECISION.md`：收益基准口径（+0.8875 dB 相对基准）。
- `refactor/EXPERIMENT_PLAN.md`：消融与自适应实验计划。
- `refactor/METHOD_GROUNDING.md`：数理 grounding。
- `refactor/FINDINGS.md`：必要机制 vs 工程惯性（CPU 阶段结论）。
- `refactor/_runs/FINAL_REPORT.md`：确定性 → 干净核心 → 阶段3/e200 的收口证据总览。
- `refactor/_runs/hnek_search/E200_CONFIRMATION.md`：HNEK 阶段3 e200 确认最终结论（唯一存活变体 `hnek_g0.25`）。
- `refactor/_runs/hnek_search/summary.tsv`：阶段3 9 变体 e50 搜索汇总。
- `refactor/PATHS.md`：路径速查（只读原件）。

## 历史 / 已废弃（勿当当前目标）

- `refactor/VERIFY_PLAN.md`：早期“最小 GPU 验证 + 成本门”，已被 `EXPERIMENT_PLAN.md` 取代。
- `refactor/RUNNER.md`：早期 nohup 后台约定，已被 systemd 持久服务取代。
- `refactor/CONTRACT.md`：早期重构契约；trick 分类法仍有效，但成本门/旧门控已废弃。
- `README.md` 中的“两个算法最好结果”历史绝对数字（+1.0439 / +1.4729）仅为历史参考，当前基准是干净框架相对 +0.8875。

## 规则

- 目标变更时必须同步更新 `PLAN.md` 与 `PROGRESS.md`。
- 任何废弃文件应在头部标注“历史/已废弃”，或从 `DOCS_INDEX.md` 的“当前权威”列表中移除。
