# G1-DT-HNEK-BCAVP: Brownian-Consistent Antithetic Variance Projection

## Evidence route

- DT's late correction is variance-dominated, while fixed AEB is negative through 1200 updates.
- Fixed HNEK changes sign repeatedly through 12000 updates, so its failure is not explained by one exit time.
- DCUM amplifies HNEK's 4000-step full-view loss and is excluded from this construction.

## Mathematical update

For the native endpoints

\[
P_+=G_\theta(X_t,z),\quad P_-=G_\theta(X_t,-z),\quad
M=\frac{P_++P_-}{2},\quad D=\frac{P_+-P_-}{2},
\]

let (h=1-t), (v=\operatorname{mean}(D^2)), and

\[
a=\min\left(1,\sqrt{\frac{\tau h}{v}}\right),\qquad Y=M+aD.
\]

When (v\leq\tau h), the implementation explicitly returns (P_+) rather than recomputing (M+D), making the self-null branch byte-exact.

## Why this is a new long-horizon operator

The restricted entropic bridge's Gaussian stationary variance is (	au h). BCAVP is the Euclidean projection of only the odd-latent component onto that physical second-moment ball. It leaves the antithetic endpoint mean—and therefore mean transport—unchanged. It is neither HNEK's fixed full-residual contraction nor AEB's unconditional removal of all latent variation.

Its intervention is decided by the current bridge state itself. The operator can engage, disengage, and re-engage without reading the optimizer step, an evaluation score, a paired target, or a plain branch.

## Falsification

Kill the construction if mean invariance, the variance bound, exact self-null, RNG preservation, or full-state resume fails. If engineering gates pass, its first empirical decision is a matched 400/800/1200 small-view trajectory.
