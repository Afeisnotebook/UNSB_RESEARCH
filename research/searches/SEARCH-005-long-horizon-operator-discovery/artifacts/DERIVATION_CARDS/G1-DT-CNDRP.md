# G1-DT-CNDRP: Confidence-Normalized Dispersion-Rate Preconditioner

## Evidence route

- `DT-FULL-3000-NEG::variance_fraction=0.739145`
- `DT-FULL-4000-NEG::variance_fraction=0.804189`
- `DT-FULL-4000-NEG::4/8 exact-zero correction batches`
- `FC-DT-INTERMITTENT-NULL`

## Mathematical update

For two independent antithetic U-statistic replicates s_r=log(eps+U_r), let a_r=grad_theta s_r and a_bar=(a_1+a_2)/2.  Per Adam coordinate j let v_j=(a_1j-a_2j)^2/2 and p_j=(v_j+eps_j)/(a_bar_j^2+v_j+eps_j), then use g_new,j=p_j*g0,j.

## Derivation

1. The old additive DT loss matches a frozen absolute log-U value and can remain biased after its early defect disappears.
2. The new object uses endpoint dispersion only as a local metric on the native gradient; it never supplies a target value.
3. Every p_j lies strictly in (0,1], so P=diag(p_j) is positive definite and commutes with Adam's positive diagonal preconditioner M.
4. Therefore g0^T M P g0=sum_j M_j*p_j*g0,j^2 > 0 for g0 != 0, and P*g0=0 iff g0=0: native descent and stationary points are preserved in the implemented optimizer geometry.
5. High estimator variance increases v_j and continuously returns that coordinate toward identity; no threshold or exit is used.

## Long-horizon property

positive-definite preconditioning preserves native UNSB stationary points and cannot turn the native gradient into an ascent direction.

## Falsification

kill if P is not SPD/identity-safe or if a 400/800-step micro run does not reduce gap amplification without a positive trajectory signal.
