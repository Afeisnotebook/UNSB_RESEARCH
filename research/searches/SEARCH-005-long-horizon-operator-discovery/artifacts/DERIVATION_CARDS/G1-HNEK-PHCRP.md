# G1-HNEK-PHCRP: Pathwise Horizon-Consistent Residual Projection

For the actual bridge rollout, write (h_i=1-t_i), (P_i=G_\theta(X_i,z_i)), and (e_i=\operatorname{mean}(P_i-X_i)^2). Starting from the exact native endpoint (Y_0=P_0) and (q_0=e_0), define

\[
c_i=q_{i-1}\frac{h_i}{h_{i-1}},\qquad
a_i=\min\left(1,\sqrt{\frac{c_i}{e_i}}\right),\qquad
Y_i=X_i+a_i(P_i-X_i),\qquad q_i=e_i a_i^2.
\]

This is the Euclidean projection of the current native residual onto the pathwise cone (q_i\leq q_{i-1}h_i/h_{i-1}). It replaces HNEK's always-active fixed exponent with a physical constraint computed from each actual trajectory.

If the native residual already decays no slower than the remaining physical horizon, (a_i=1) and the code returns (P_i) exactly. No step counter, score, paired target, plain branch, or learned exit threshold is available to the operator. Training rollout, the gradient-carrying final endpoint, and evaluator rollout use the same recorded pathwise scale.

The construction is specifically routed from the clean baseline's determinism: BCAVP measured (z/-z) endpoint variance at (2.37\times10^{-17}), so latent-variance control is a no-op here. PHCRP instead observes the deterministic residual change induced by the real stochastic bridge states.
