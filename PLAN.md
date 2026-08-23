# 项目总计划（当前权威，2026-08-24）

## 目标

把已完成的算法重构升级为“可投稿 ICLR 的、有数理支撑的算法方法”。本机确定性修复、核心干净结果与 HNEK 桥原生有界搜索/e200 确认已全部完成；下一步是迁移到更好服务器补多 seed 复现。

确定性已修复：手工确定性反射 pad + CuBLAS workspace + strict deterministic，同 seed GPU smoke 逐位一致。

## 收益基准

- 干净框架相对收益：DT vs 干净 plain = **+0.8875 dB PSNR**（见 refactor/BASELINE_DECISION.md）。
- 不追原始 modified 基线绝对数值（19.78 那套）。

## 阶段

1. 阶段1（已完成）：确定性反射 pad + CPU 单测 + GPU 同 seed 两次逐位复现。
2. 阶段2（已完成，单 seed 口径）：确定性干净核心只保留 seed=2026（DT plain/best + HJ plain/true/roll + warmup）。跨 seed 稳健性留给更好的服务器，本机不再跑 2027/2028。结果落 `refactor/_runs/metrics_clean_core/CLEAN_CORE_RESULTS.md/json`。
3. 阶段3（已完成）：围绕“amortized endpoint 条件律的剩余时域归一化”完成 9 变体 e50 搜索与 2 变体 e200 确认；唯一存活候选为 `hnek_g0.25`（e200 macro PSNR delta +0.7884 dB，4/5 域为正，非 confirmatory）。详见 `refactor/_runs/hnek_search/E200_CONFIRMATION.md/json`。

## 门控与日志

- 日志治理：STATUS/HEARTBEAT/MONITOR_LOG 只由 monitor_once.sh 写；PLAN/PROGRESS/CHANGELOG 由主 agent 在实质里程碑更新。
- 当前权威 / 历史废弃划分见 DOCS_INDEX.md。
