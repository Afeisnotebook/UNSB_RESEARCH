# Causal audit protocol

## State and correction

Let `S` contain G/F/D/E, all optimizer and scheduler states, method co-state,
samplers and RNG. For matched native and probe updates define

\[
u_i(S;\xi)=u_0(S;\xi)+c_i(S;\xi).
\]

The audit distinguishes correction validity from correction propagation. A
positive checkpoint does not establish either, and a later negative PSNR does
not identify which one failed.

## A. Same-state correction field

At positive, near-reversal and negative checkpoints, evaluate native and probe
updates from the same full state on matched and independent unpaired batches.
Record:

- Adam-metric norm and block/time/domain decomposition of `c_i`;
- cosine and directional derivative against independent native gradients;
- local loss curvature by symmetric finite difference where feasible;
- across-batch mean, covariance and sign stability.

This determines whether the correction mean reverses, its scale becomes
ill-conditioned, or variance dominates while the mean remains useful.

## B. Native-flow propagation

Apply a controlled small perturbation in the measured correction direction and
then continue both perturbed and unperturbed states with identical native UNSB
updates. At horizons 1/8/32/200 measure the finite-difference action

\[
J_{0,H}(S)c \approx
\frac{\Phi_0^H(S\oplus\epsilon c)-\Phi_0^H(S)}{\epsilon}.
\]

Record retention, rotation and amplification in parameter, endpoint, bridge and
unpaired-objective spaces. Paired discovery metrics label completed branches
only and are not exposed to an algorithm.

## C. Coupled-state attribution

Use diagnostic hybrid states to separate G/F parameters, D/E parameters,
optimizer moments and method/rollout co-state. Hybrid states are causal probes,
not trainable candidates. Report which component is necessary for the later
loss of a beneficial perturbation and whether full-state evolution is stable.

## D. Bias--variance attribution

Across independent batches, domains, bridge times and latents estimate whether
the probe differs from native UNSB primarily through update mean or covariance.
Only evidence of variance-dominated failure may route to an unbiased control
variate/stratified/antithetic estimator. Mean reversal routes to operator
reconstruction; co-state amplification routes to coupled dynamics.

## Routing output

The audit produces a machine-readable causal matrix. Each supported failure
class may create at most one Generation-1 derivation. Algorithm names and
formulas are not preregistered before this evidence is available.
