# 方法数理 grounding（投稿级重表述）

本文件把两个重构算法从“工程实现语言”升级为“可投稿的数理语言”，同时把 novelty 边界写清楚。

## 共同背景

UNSB 把一族受限端点条件律 `Q_h(y | x_t)`（`h = 1-t` 是剩余时域）摊销进一个共享端点预测器 `G_θ(x_t, z)`。无配对训练只给了 PatchNCE 对应、GAN 与受限 SB 熵项这些代理信号，因此 `G_θ` 的端点响应在跨域、跨桥时刻的局部行为没有被直接约束。两个方法都只在这个“amortized endpoint 条件律”的局部响应上施加**最小、可验证、前向不变的干预**。

## DT：amortized 端点条件律的有界函数正则（trust-region 一致性）

定义归一化端点方向响应：

\[
R_θ(x_t, z)=\frac{G_θ(x_t,z)-x_t}{1-t},
\qquad
U_θ(x_t)=\mathrm{pool}\left(\mathrm{Var}_{z\sim p(z)}[R_θ]\right)
 \big/ \left(\mathrm{pool}\left(\|\mathbb E_z[R_θ]\|^2\right)+\epsilon\right).
\]

冻结一个 first-use 参考律（不再更新），在其 `(domain, time)` 图表上估计 `\log U` 的均值与尺度 `(\mu_{d,t},\sigma_{d,t})`。DT 损失是：

\[
\mathcal L_{\mathrm{DT}}
=\mathbb E_{x_t,d,t}\left[
  \rho\left(
    \frac{\log U_θ(x_t)-\mu_{d,t}}{\sigma_{d,t}},
    \frac{\log U_{\mathrm{ref}}(x_t)-\mu_{d,t}}{\sigma_{d,t}}
  \right)\right],
\]

其中 `ρ=smooth L1`，并按 domain 等权聚合。它只在 `λ>0` 的短窗口内加入，`λ` 很小，参考端 stop-gradient。

### 措辞边界

- 它约束的是“当前模型相对固定参考条件律的**标准化响应漂移**”，是**有界函数正则 / 信任域一致性约束**。
- **不声称**：估计真实后验协方差、估计校准不确定性、teacher 是 clean oracle、或跨时刻 covariance 校正。
- 参数不变，`G_θ` 的结构与端点插值不被替换。

## HJ：前向不变的“结构有害方向”梯度修正（带可验证 gate）

PatchNCE forward 是 `L_NCE(f_q,f_k)`。HJ **不改变 forward 特征和 loss 数值**，只在 backward 做约束。

定义源图结构切方向 `d`（edge 与 SSIM 梯度的单位 RMS 平均，逐图归一化）。对目标做 `±εd` 扰动，复用原 PatchNCE 编码器得到探针特征，记单侧与中心差分方向敏感度：

\[
\Delta_{-} = \frac{L(f_q(-ε d),f_k)-L(f_q,f_k)}{ε},
\qquad
\Delta_{c} = \frac{L(f_q(-ε d),f_k)-L(f_q(+ε d),f_k)}{2ε},
\]

取保守方向敏感度 `δ = min(Δ_-, Δ_c)`，并只保留正的部分 `δ_+ = max(δ, 0)`。方向风险为逐图分位数归一化：

\[
r_{\mathrm{dir}} = \min\!\left(1,\ \frac{\delta_+}{Q_{0.75}^{\mathrm{per-image}}(\delta_+)}\right).
\]

边界不稳定度由 PatchNCE 双向 margin `m = min(m_fwd, m_rev)` 定义：

\[
b = 4\,\sigma(m/s)\,\bigl(1-\sigma(m/s)\bigr),\qquad s=0.001.
\]

最终 risk 是二者的几何组合，gate 为：

\[
r = \sqrt{r_{\mathrm{dir}}\cdot b},
\qquad
\mathrm{gate} = \mathrm{TopQ}_{0.75}^{+}(r)\ \land\ (r \ge 0.05)\ \land\ \bigl(Q_{0.75}^{\mathrm{per-image}}(\delta_+) \ge \delta_{\min}\bigr),
\]

其中 `TopQ⁺` 表示逐图取“正且不小于该图 75% 分位数”的位置，默认 `δ_min=0`。

只在 gate 命中的位置，对 PatchNCE 梯度施加投影：

\[
\nabla_f L_{\mathrm{NCE}}
\;\leftarrow\;
\nabla_f L_{\mathrm{NCE}}
-\alpha\, \frac{\langle \nabla_f L_{\mathrm{NCE}}, d\rangle_+}{\|d\|^2}\, d,
\qquad
\alpha=0.5.
\]

`α` 是投影强度（最好配置 0.5），`d` 与 gate 均在 forward 之后、backward 时使用，forward 恒等。

### 可验证 gate

gate 完全由 `f_q`、`f_k`、`d`、分位数 `q=0.75`、`s=0.001`、`min_risk=0.05`、`δ_min=0` 的纯函数决定，可从 raw 特征重算；roll-control 把 `d`（或 gate）按 patch 平移后再投影，用来检验“收益是否依赖真实结构方向，而非只是通用稳健化”。

### 措辞边界

- 贡献点是“在 UNSB 的 unpaired EROT + amortized 端点律上，用**源图结构切方向**做前向不变的结构有害梯度约束，并给出可验证 gate”。
- **不声称**：首次 preconditioning、首次 gradient surgery、首次不确定校准、新参考设计、新噪声 schedule。

## 自适应介入（结论）

介入时机/强度已由可观测诊断量驱动，替代手调 `λ=0.001` + `ramp5 hold15 decay25`：

- DT（`--dtcov_lambda_schedule adaptive`）：用 teacher-student 分布距离的 EMA plateau 检测决定退出，PSNR 18.8911（+0.9332）≥ 手调 18.8453（+0.8875）。
- HJ：全局 strength 与 per-location risk 剂量自适应均系统劣于固定 strength 0.5（+2.1190 / +1.0156 vs +2.7533），说明 HJ 的局部性由 gate 保证、dose 应固定；作为负结果/分析点呈现。

机制证据由只读诊断日志记录：DT drift 0→0.52、HJ gate 命中率≈0.097、conflict risk≈0.11，介入在 epoch 6 附近生效（见 `refactor/_runs/diagnostics/`）。
