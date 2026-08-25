# Research lifecycle

研究对象通过稳定 ID 连接，不依赖易变路径或阶段性文件名：

```text
MOT → CAND → ITER → SEARCH → EXP-L0/L1/L2/L3/L4 → DEC → OUTPUT
```

- `MOT`：研究问题和允许的科学表述。
- `CAND`：一个可辨认候选算法家族。
- `ITER`：公式、行为或关键实现发生变化后的冻结版本。
- `SEARCH`：冻结的多 lane 搜索控制器；它可以比较或合成 iteration，但自身不是有效性结论。
- `EXP`：一次不可变实验记录。
- `DEC`：根据指定实验作出的推进、修订、暂停、否决或确认。
- `OUTPUT`：仅引用已裁决证据的论文、图表、release 或 handoff。

实验放大等级：L0 contract、L1 local、L2 medium-4090、L3 scale-5090、L4 untouched/multi-seed confirmation。任何跨级推进都需要决策记录。
