# SEARCH-005 local operator-discovery report

SEARCH-005 修正了 SEARCH-003 把路线一退化为 controller/whole-state branch selection 的目标漂移。它只允许数学算子、无偏估计器、坐标变换或耦合训练动力学，禁止固定窗口、paired-PSNR 控制和 handoff。

本轮共运行 6 类 Generation-1 机制和 4 次由失败证据触发的 Generation-2 修订。所有实际训练候选先通过 endpoint identity、zero-intervention、full-state resume、paired blindness 等工程门禁。

唯一完成独立 2400-step 的新算子 PCOA 在 400/800/1200 为 `+0.0436/+0.0755/+0.1927 dB`，但 1600/2400 为 `−0.5698/−0.8608 dB`，峰值到终点绝对回撤约 `2.10 dB`。NPOOA 把 PCOA 改为严格保持每个 G/D/E 原生 Adam 位移范数的角向更新；其 400 为 `+0.2314 dB` 且护栏通过，800 已反转为 `−0.9664 dB`。因此 coupled-game 机制按两代上限关闭。

没有候选进入 full100、第二 seed 或 confirmation20。PCOA 仅是 weak fallback，不是论文方法；若未来运行 full100，必须明确标为高算力证伪，而不是已晋级验证。

权威机器证据位于：

- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/LOCAL_RANKING.json`
- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/HYPOTHESIS_LEDGER.json`
- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/REVERSAL_ATLAS.jsonl`
- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/CANDIDATE.json`
- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/ROUTE1_STOP.json`
