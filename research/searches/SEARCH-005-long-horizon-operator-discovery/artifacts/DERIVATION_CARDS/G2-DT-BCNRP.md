# G2-DT-BCNRP: Block-Confidence Normalized Rate Preconditioner

CNDRP was the only new 800-step operator essentially tied with plain, but its elementwise scales approached (10^{-6}) and damaged SSIM and the worst domain. BCNRP keeps the same evidence route while replacing the coordinate metric with one positive scalar per generator parameter tensor.

For block (b), two independent DT sensitivity gradients give

\[
s_b=\left\|\frac{a_{1b}+a_{2b}}{2}\right\|^2,
\qquad v_b=\frac{1}{2}\|a_{1b}-a_{2b}\|^2,
\qquad p_b=\frac{v_b+\epsilon_b}{s_b+v_b+\epsilon_b},
\qquad g_b^*=p_b g_{0b}.
\]

The scalar (p_b>0) preserves every within-block gradient direction. In any positive diagonal Adam metric, the native descent alignment remains strictly positive for every nonzero block. There is no endpoint change, added loss, schedule, paired target, or exit rule.
