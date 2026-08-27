# G2-GAME-NPOOA — Norm-Preserving Orthogonal Optimistic Adam

PCOA 的 2400-step 轨迹证明了耦合博弈相位会被改变，但也暴露了原式把方向旋转与原生 Adam 更新幅值变化混在一起：反转前 G/E 的校正占比明显上升，单次 E 跃迁甚至可被近乎再加一遍。

对每个参与者令当前原生 Adam 位移为 $u$，上一原生位移为 $v$：

$$
\rho=\operatorname{clip}\left(\frac{\langle u,v\rangle}{\|v\|^2},0,1\right),\quad
d_\perp=(u-v)-\frac{\langle u-v,u\rangle}{\|u\|^2}u,
$$

$$
w=u+\rho d_\perp,\qquad u_*=\frac{\|u\|}{\|w\|}w.
$$

因此 $\|u_*\|=\|u\|$：算法只能旋转原生更新，不能加速或减速它。连续更新共线、不可预测、反向或处于固定点时均严格回到 plain。它不读取 paired target，不使用训练窗口或退出阈值，也不改变 UNSB 的目标、endpoint law 与 Adam moment recurrence。

判死条件：工程门禁必须逐步验证范数保持、first-step/zero-intervention/resume 精确一致；若 800-step 已负，或独立 2400-step 仍发生长程反转，则耦合博弈机制在本轮两代后关闭。
