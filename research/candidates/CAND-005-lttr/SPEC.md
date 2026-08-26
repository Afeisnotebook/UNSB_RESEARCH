# LTTR 冻结规格

对训练桥状态 `x_t` 和固定 antithetic latent pair `z,-z`，定义：

\[
 m_\theta=\tfrac12[G_\theta(x_t,z)+G_\theta(x_t,-z)]-x_t,
 \qquad
 a_\theta=\tfrac12[G_\theta(x_t,z)-G_\theta(x_t,-z)].
\]

在 32×32 region 上定义 latent tangent ratio：

\[
 q_\theta=\log\frac{\operatorname{pool}\|a_\theta\|^2+\epsilon}
 {\operatorname{pool}\|m_\theta\|^2+\epsilon}.
\]

冻结 first-use generator `theta_0`。用每张图参考 `q_0` 的 MAD 建立局部尺度，直接匹配 `(q_theta-q_0)/s_0`，不使用 batch/domain EMA，也不把两端 clip 到同一饱和值。风险权重由参考 latent tangent ratio 与源图结构梯度共同构成。

`safe` 分支进一步对 region mean residual 的夹角施加单侧 barrier：

\[
 L_{safe}=\mathbb E[w\,\max(0,0.5-\cos(m_\theta,m_0))^2].
\]

总损失为 `L_UNSB + 0.001 * schedule * (L_tangent + 0.25 L_safe)`。schedule 继承历史有效的短窗口原则：总更新量 20% 时激活，覆盖 50% 更新量，窗口内 20% ramp、40% hold、40% cosine decay，之后回到 plain continuation。推理完全关闭 LTTR。

两条冻结 lane：

- `lttr_tangent`：只有 latent tangent chart；
- `lttr_safe`：tangent chart 加 one-sided direction barrier。
