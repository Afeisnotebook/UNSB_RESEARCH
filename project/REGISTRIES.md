# Registry contract

根状态只保存当前指针；完整实体分别登记在：

- `research/motivations/REGISTRY.json`
- `research/candidates/REGISTRY.json`
- `research/searches/REGISTRY.json`
- `experiments/REGISTRY.json`
- `decisions/DECISION_LEDGER.json`

每个候选必须列出来源动机、当前 iteration、实验 ID 和决策 ID。每个 output 必须能反向追到 decision、experiment、candidate iteration 和 canonical。
