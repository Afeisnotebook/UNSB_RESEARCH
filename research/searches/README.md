# Search controllers

`SEARCH` 是冻结的多 lane 选择/合成控制器，不是候选状态，也不是实验结论。代码、排序规则和数据访问边界在这里冻结；每次真实运行必须在 `experiments/` 创建独立记录，再由 `decisions/` 裁决。

当前搜索：

- [SEARCH-001 clean directional](./SEARCH-001-clean-directional/README.md)：plain 与 DT/HJ/HNEK anchors，加 DCUM/LBST/PTQ/AEB 新机制；当前为 `IMPLEMENTED_UNRUN`。
