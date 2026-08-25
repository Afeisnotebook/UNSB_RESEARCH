# DT、HJ、HNEK 的 Codex 互动形成史

> 整理日期：2026-08-26
>
> 范围：只回答“用户怎样与 Codex 互动，最终得到三个算法”。
>
> 不回答：三个算法是否正确、结果是否可靠、论文 claim 是否成立。

## 1. 总结

三个算法不是一次性让 Codex “生成三个创新点”得到的，也不是用户预先写好公式后让 Codex 代写代码。它们来自同一个反复循环：

```text
用户提出科学直觉与论文边界
→ Codex 把直觉翻译成可计算候选
→ 用户安排本机/4090/5090实验并返回结果
→ 用户质疑弱结果、假阳性和不合理解释
→ Codex 做归因、收缩或更换具体构造
→ 用户决定继续、停止或把边界进一步收紧
→ Codex 再推导公式、实现代码、打包协议
```

角色分工也很清楚：

- 用户主要提供研究问题、想保留的论文动机、不能越过的边界、算力安排、结果回传和最终取舍。
- Codex 主要完成源码审计、数学形式化、候选构造、命名、代码实现、实验包、结果归因和下一轮方案。
- 最终三个算法都属于人机共同迭代产物；只是三者中“用户直接给到的内容”和“Codex 自主推导的内容”比例不同。

## 2. DT-CovMatch：从协方差直觉到 domain-time 校准

### 2.1 用户先给出的核心问题

最早阶段，用户接手了一条 posterior-drift 思路，并明确追问：只使用 posterior mean、忽略 posterior covariance，是否能形成区别于 DDSB 的创新。Codex 先做数学审查，并将能被当前 UNSB 代码实际测到的量收缩为：固定 bridge state 与 time、改变 latent 后，多个 endpoint proposal 的方向分歧。

因此，DT 的最上游不是 Codex 凭空发明“协方差”，而是用户持续坚持的研究母题：

> 多域无配对 UNSB 在不同子群上可能对更新方向有不同程度的迷惑；同一状态多次采样若方向差异大，就应把它视为一种不确定性/协方差信号。

早期尝试把这个信号用于 test-time rollout 放缩或训练时全局 `u_match`。三域的 early regularization 有正信号，六域却出现明显域间 trade-off。用户没有接受“信号没用”或继续堆一般模块，而是要求从算法实现和分域结果解释为什么三域有效、六域不稳。

### 2.2 决定 DT 形态的关键互动

在任务“看一下UNSB_Cov3项目的内容，比较多所以主要关注近期的内容，总结一下目前的思路和已经取得的结论，大概的说一下后续可能的研究思路”（`019f417a-b467-7922-bcd8-29a2f8aa47c2`）中，用户给出了决定性的边界：

> “咱们核心的算法还是得紧扣不确定性或者协方差这种切入点，不然 paper 不好写啊，别最后调研调研改成专家系统或者门控了。”

Codex 随后把此前实验事实翻译为一个新的假设：六个域的绝对 `log-U` 不是同一把尺子，同一 bridge time 的含义也不同；问题不在于是否使用 U，而在于缺少 domain-time coordinate calibration。

由此 Codex 提出并命名：

```text
Domain-Time Calibrated Covariance Match
DT-CovMatch
```

其核心构造是对 teacher 的 `log-U` 按 `(domain, time)` 建立统计坐标，再比较 current 与 frozen teacher 的标准化相对偏离，而不是把六域的绝对 U 直接压到一起。

用户随后要求中文 roadmap、创建新的算法代码包、本地 smoke 和代码自审。Codex据此完成实现。之后出现的 frozen teacher、grouped/stratified estimator、短窗口和校准细节，是在用户持续返回服务器结果、质疑跨域不稳与工程堆叠后逐步形成的，不是 DT 首次提出时就完整存在。

### 2.3 DT 的形成归属

- 用户给出：协方差/不确定性母题，多域子群差异解释，不能滑向专家或门控的论文边界，以及实验反馈。
- Codex给出：从 proposal disagreement 到 domain-time calibration 的数学落地、`DT-CovMatch` 名称、frozen-teacher regularizer、实现与实验协议。
- 结论：DT 是“用户定研究轴，Codex根据负结果完成具体算法化”的共同产物。

## 3. HJ-PatchNCE：从“高不确定 patch 要谨慎”到 harmful-joint 梯度处理

### 3.1 用户主动开辟第二条独立算法线

在任务 `019f57a5-e828-7370-a64b-6686d10ed7fa` 中，用户先用一份 PatchNCE brief 要求把第二条线从 DT-CovMatch 干净拆开，只改 PatchNCE 相关部分，默认路径必须等价原始 UNSB。

初版 U-PatchNCE 很快没有通过。此时用户没有要求随便换一个第二创新点，而是明确冻结论文动机：

> “对于不确定性较大的 patch 块咱们应该更谨慎地更新……这是一个很关键的创新点。”

并进一步授权 Codex 从这个动机出发在本地 1660 自主寻找可行实现，而不被已有 MD 中的具体公式绑死。

### 3.2 用户不断否决“看起来有用、实际站不住”的版本

这条线的具体构造经过多次变化：

1. detached confidence 加权或简单降低高 U patch 权重；
2. 把“不确定”改写为“不确定且与恢复目标冲突”的 gradient risk；
3. 从整体降权转向只处理有害梯度分量；
4. 从 bridge-risk 转向直接预测结构伤害；
5. 从“少学”转向高风险 patch 使用更安全的关系保持目标；
6. 再从较重的关系替代收缩到只在特定层、特定 harmful-joint 条件下处理 backward。

推动这些变化的不是任意扩展，而是用户反复提出的具体反证问题：短 epoch 会不会错杀谨慎机制、为什么随机 seed 能放大十倍、为什么同样后续训练会把更好的状态训坏、3 dB 是否只是 plain 的瞬时塌陷、关系替代是否只是减少了 NCE 总量。

当简单干预失败后，用户再次要求：

> “从动机角度……结合咱们已有的内容再发散发散，大胆假设小心求证，不需要拘泥于咱的算法了。”

Codex 才提出“风险条件下的关系路由 PatchNCE”。后续服务器结果又迫使该路线经历质量守恒、窗口/剂量、handoff 和多 checkpoint 归因，最终收敛到 Layer-0 Harmful-Joint PatchNCE 的实现形态。

### 3.3 HJ 的形成归属

- 用户给出：第二创新点必须属于 PatchNCE、高不确定 patch 应谨慎、默认路径不得改变、失败后仍保留动机但允许重定义判据。
- Codex给出：uncertainty proxy、gradient-risk/harm oracle、harmful-joint 判据、layer-0 backward 操作、具体实现与命名。
- 结论：HJ 不是一次提案，而是用户用一连串质疑和实验结果把 Codex 的候选逐层筛窄后形成的算法。

## 4. HNEK：前两条路线受挫后，Codex 回到桥核本身推导

### 4.1 触发点

HNEK 的直接来源在任务“算法3”（`019febc5-a038-70b2-b93c-91686a5e035c`）。在 DT、HJ 以及后续 time-active/路径一致性方向相继暴露问题后，用户不断要求停止无效绕行，并把成本压到单 seed、一次性判决。

当 TA/KCK 类构造也没有给出可接受结果时，用户先质问这几天工作的意义，随后问：

> “所以接下来怎么办，等死吗”

Codex 在这里主动提出一个新的边界：不再加任意 time network、teacher、gradient surgery 或通用正则；保留表现更稳的 plain 主体，只修改 endpoint bridge kernel 的参数化，使新方法本身能够被识别为 Schrödinger Bridge 原生贡献。

用户只回复：

> “就这么做吧”

这句话是授权继续推导，不是用户给出了 HNEK 公式。

### 4.2 Codex 如何得到 HNEK

获得授权后，Codex重新检查官方 UNSB 的时间表、熵项和 endpoint residual 表达，抓住两点：

- rollout 使用非均匀 physical time；
- 原实现中的部分权重仍按均匀 index horizon 表达，且 endpoint residual 的尺度会随剩余时域变化。

Codex 先提出用真实剩余时域保护 endpoint，再进一步把目标写到 horizon-normalized residual 坐标中，形成 HNEK 的核心构造和名称。此后 `gamma`、coordinate、partial application 等变体才进入本地搜索。

### 4.3 HNEK 的形成归属

- 用户给出：停止泛化插件、回到 UNSB/SB 本体、控制时间成本、允许 Codex 按最小方案继续。
- Codex给出：重新阅读源码后的 horizon/endpoint kernel 问题、normalized residual 推导、HNEK 名称、公式、代码和变体设计。
- 结论：三个算法里，HNEK 的具体数学构造最偏 Codex 主导；用户的贡献主要是用前序失败和明确边界把搜索空间压回“桥原生、最小、可判死”的方向。

## 5. 三个算法不是平行出现，而是递进出现

| 算法 | 用户首先固定的内容 | Codex主要补全的内容 | 形成方式 |
|---|---|---|---|
| DT | 多域不确定性/协方差主线；不能变专家/门控 | domain-time 校准、teacher 正则、实现与命名 | 由跨域 U 尺度冲突推导 |
| HJ | 高不确定 patch 应谨慎；第二创新点必须落在 PatchNCE | harmful-joint 判据、梯度操作、层级选择、实现与命名 | 多轮负结果逐层收缩 |
| HNEK | 放弃通用插件，回到 SB 本体；成本必须可控 | horizon-normalized endpoint kernel、公式、实现与命名 | 前两条路线与 time-active 线受挫后的桥原生重启 |

所以最准确的说法不是“用户让 Codex 给了三个算法”，而是：

> 用户通过持续设定研究动机、返回真实实验、否决不可信解释和收紧论文边界，逐步把 Codex 的搜索空间压缩；Codex则在每个阶段把剩余空间转成数学构造、代码和实验协议。DT、HJ、HNEK 是这段持续互动的三个阶段性产物。

## 6. 主要历史任务索引

- DT 早期理论源头：`019ef46e-94b6-7113-8418-e5a739f5d25b`。
- DT 正式命名与实现任务：“看一下UNSB_Cov3项目的内容，比较多所以主要关注近期的内容，总结一下目前的思路和已经取得的结论，大概的说一下后续可能的研究思路”，`019f417a-b467-7922-bcd8-29a2f8aa47c2`。
- DT 后期回顾任务：“全面阅读 DT_CovMatch 项目”，`019f94a1-20b3-7481-a829-0b86ecf4d3c3`。
- HJ 主任务：`019f57a5-e828-7370-a64b-6686d10ed7fa`。
- HNEK 主任务：“算法3”，`019febc5-a038-70b2-b93c-91686a5e035c`。

上述索引来自对原始 user/assistant turns 的回读；本文引用它们是为了还原互动过程，不把对话中的阶段性性能判断当成新的科学裁决。
