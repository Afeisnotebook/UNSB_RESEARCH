# 项目总计划（当前权威，2026-08-15）

## 目标

把已完成的算法重构升级为“可投稿 ICLR 的、有数理支撑的算法方法”。已停止多 seed（DT/HJ 各 3 seed，方向性为正已足够当前阶段证据）；后续聚焦**介入优化（方法/时机/强度）**，用已有诊断数据找切入点，再做最小、单变量的优化实验。reflection_pad2d 非确定性不修，数字按 mean±CI 报告并在论文写 limitation。

## 收益基准

- 干净框架相对收益：DT vs 干净 plain = **+0.8875 dB PSNR**（见 refactor/BASELINE_DECISION.md）。
- 不追原始 modified 基线绝对数值（19.78 那套）。

## 阶段

1. 机制定位：单 seed knock-out 已完成（作方向参考，不再扩多 seed）。
2. HJ 归因：3 seed 均为 true>plain、true>roll，作当前阶段证据；sub-dB 差异不精确声明。
3. 数理 grounding：已完成（见 refactor/METHOD_GROUNDING.md）。
4. 介入优化（当前重点）：从诊断 JSONL 定位切入时机/停止时机，试“DT 延长窗口/plateau 退出”、“SB 熵梯度作时机信号”、“HJ 仅调 gate 阈值”，目标明确优于手调。
5. novelty 边界 + 投稿级方法描述（已基本完成，后续按优化结果微调）。

## 门控与日志

- 日志治理：STATUS/HEARTBEAT/MONITOR_LOG 只由 monitor_once.sh 写；PLAN/PROGRESS/CHANGELOG 由主 agent 在实质里程碑更新。
- 当前权威 / 历史废弃划分见 DOCS_INDEX.md。
