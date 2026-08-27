# G1-GAME-PCOA: Predictability-Calibrated Optimistic Adam

The endpoint and regularizer families repeatedly produced locally beneficial updates that were later counteracted by the evolving D/E game. PCOA therefore changes the coupled optimizer dynamics rather than another endpoint object.

For each player (p\in\{G,D,E\}), let (u_t^p) be the displacement canonical Adam would make at the current state. Define

\[
\rho_t^p=\left[\frac{\langle u_t^p,u_{t-1}^p\rangle}
{\|u_{t-1}^p\|^2}\right]_{[0,1]},
\qquad
u_{*,t}^p=u_t^p+\rho_t^p(u_t^p-u_{t-1}^p).
\]

At (ho=1), this is the (2u_t-u_{t-1}) optimistic update with bilinear last-iterate convergence evidence. The regression coefficient suppresses optimism when the previous stochastic displacement is not predictive. The first step is native; an unchanged field is exactly native because the innovation vanishes. G, D and E use the same rule; the feature sampler F remains canonical.

Primary basis: Daskalakis et al., *Training GANs with Optimism* (arXiv:1711.00141), and Liang & Stokes, *Interaction Matters* (AISTATS 2019).
