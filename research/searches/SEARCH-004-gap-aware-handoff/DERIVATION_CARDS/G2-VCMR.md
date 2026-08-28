# G2-VCMR — Variance-Carried Moment Rebase

## Parent and revision evidence

At HNEK-3000, hard disable produced a -1.80 dB 32-step shock.  Clearing the G/F
optimizer state improved that point by +1.63 dB and the 200-step point by +0.16
dB.  Reconstructing native moments improved the first 32 steps but reversed by
200, while G1-LCNMP removed the measured native-opposing component exactly yet
remained below hard disable at 200.  The evidence therefore rejects both a
co-state explanation and the hypothesis that only one opposing component of
the inherited first moment is harmful.

## Operator

For player \(r\in\{G,F\}\), compute the current unpaired native mean gradient
\(\bar g_r\) and inherited effective first moment

\[
q_r=m_r/(\sqrt{v_r}+\epsilon).
\]

If \(q_r^\top\bar g_r\ge 0\), VCMR is identity.  If the inherited direction is
in conflict, VCMR performs

\[
m_r^*=0,\qquad v_r^*=v_r,\qquad t_r^*=t_r.
\]

All parameters, schedulers, D/E state, streams and RNG remain unchanged.

## Interpretation

The HNEK-frame first moment is discarded as a velocity because its coordinate
meaning changes when the HNEK bridge update is removed.  The second moment is
retained as an Adam-metric trust geometry.  Native gradients therefore rebuild
direction immediately, while the inherited variance scale adapts smoothly via
the unchanged canonical \(\beta_2\) recurrence.  This is not a scalar annealing
schedule and it does not move outputs toward plain.

## Properties

- target blind and plain-reference free;
- identity when the inherited tangent is native-compatible;
- exact preservation of parameters, second moments and optimizer age;
- no inference cost;
- four shadow G/F backward passes once at handoff.

## Falsifier

VCMR must outperform both complete-state hard disable and full optimizer reset
on the HNEK-3000 common-clock continuation at 200 updates, with the same
target-blind conflict reduced to zero.  If it only protects the 32-step point or
fails the 200/800 trajectory gate, this one permitted optimizer-transport
revision is closed.
