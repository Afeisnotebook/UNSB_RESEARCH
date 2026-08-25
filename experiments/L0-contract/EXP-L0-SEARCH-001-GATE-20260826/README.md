# EXP-L0-SEARCH-001-GATE-20260826

SEARCH-001 在本地真实六域数据上的工程门禁。源代码锁定为
`6b89c1d5568b4a067cfe8ba2cde471fb581ef7ee`，seed=2026，manifest SHA256 为
`1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b`。

结果为 PASS：plain 双跑完整状态精确一致；1 步中断后恢复到第 2 步与连续
训练精确一致；重复 discovery 评估精确一致；PTQ 的 50 步质量为
`[25,12,6,4,3]`。本门禁没有读取 confirmation20，也不构成算法效果证据。

大 checkpoint 留在本地 `runs/directional_search_20260826/engineering_gate/`，
仓库只保存小型协议锁和裁决事实。
