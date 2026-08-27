# G2-HNEK-PHRSUP: Physical-Horizon Rate-Safe Update Projection

PHCRP confirmed that the pathwise horizon defect can be eliminated, but direct contraction of the deterministic endpoint mean degraded SSIM before its PSNR reversal. PHRSUP keeps the observable and changes the mathematical object to which it is applied.

Let

\[
\phi=\left[\log e_i-\log\left(e_{i-1}\frac{h_i}{h_{i-1}}\right)\right]_+,
\quad a=\nabla_\theta\phi,
\quad g_0=\nabla_\theta L_{\mathrm{UNSB}},
\]

and let (M) be Adam's positive diagonal gradient-to-update metric. The projected update is

\[
g_*=\arg\min_g\|g-g_0\|_M^2
\quad\text{s.t.}\quad \langle a,g\rangle_M\geq0.
\]

Thus (g_*=g_0) when the native update is safe; otherwise

\[
g_*=g_0-\frac{\langle a,g_0\rangle_M}{\langle a,a\rangle_M}a.
\]

Because the Adam displacement is (-Mg_*), the first-order defect change is nonpositive. The native descent alignment is also nonnegative by Cauchy-Schwarz. Unlike PHCRP, the endpoint law, rollout and inference outputs are exactly plain at the current parameters.
