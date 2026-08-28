# Decision records

## 接任摘要

| 问题 | 回答 |
|---|---|
| 是什么 | 已发生关键裁决的不可变记录；新裁决追加文件，不原位改写旧判断。 |
| 当前结论 | T5 关闭旧 continuous DT/HJ；T7 接受新 canonical；T12 以完整状态 native handoff 将 CAND-002 晋级为 sustained-local。 |
| 时间线位置 | records 覆盖 T5、T7、T9–T12。 |
| 先看哪里 | 按下表从最新基座决策读回算法决策。 |

| 时间线 | 记录 | 含义 |
|---|---|---|
| T5 | [算法状态](./DEC-20260824-ALGORITHM-STATUS.md) | DT/HJ 关闭，HNEK development-frozen |
| T5 | [baseline 选择](./DEC-20260824-BASELINE-SELECTION.md) | 统一 clean deterministic 比较口径 |
| T7 | [clean re-exploration 包审计](./DEC-20260826-CLEAN-REEXPLORATION-AUDIT.md) | 数值自洽但门禁不全，不作为训练父节点 |
| T7 | [新 canonical 接受](./DEC-20260826-NEW-CANONICAL-ACCEPTANCE.md) | 本地真实数据微验证通过，新基座 READY |
| T9 | [SEARCH-001 本地总冠军](./DEC-20260826-SEARCH-001-LOCAL-WINNER.md) | HNEK 当时冻结为第一候选 |
| T10 | [有限期 HJ 候选](./DEC-20260827-HJ-FINITE-HORIZON-LOCAL-CANDIDATE.md) | HJ finite-horizon 取代 HNEK 成为当前第一候选 |
| T11 | [路线一停止](./DEC-20260828-SEARCH005-ROUTE1-STOP.md) | SEARCH-005 没有持续算子；PCOA 仅 weak fallback |
| T12 | [路线二 sustained-local](./DEC-20260828-SEARCH004-ROUTE2-SUSTAINED-LOCAL.md) | HJ native handoff 通过本地长程门禁，进入 full100 4090 |

L0 SEARCH gate 当前登记为工程实验和当前状态；未来 stage1/2/3 若改变候选身份，应新增独立 DEC record。
