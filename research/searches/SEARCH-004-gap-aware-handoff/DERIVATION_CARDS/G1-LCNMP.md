# G1-LCNMP — Least-Change Native Moment Projection

## Evidence parent

SEARCH-005 found that positive HJ/HNEK perturbations can lose direction memory
under later native flow and that coupled-state transport is method-specific.
SEARCH-004 therefore tests optimizer tangent compatibility separately from
network-state quality.  Zeroing or fully rebuilding Adam moments is only a
causal ablation: both operations discard potentially useful history.

## Mathematical object

At handoff, for player \(r\in\{G,F\}\), let

\[
q_r = \frac{m_r}{\sqrt{v_r}+\epsilon}
\]

be the inherited effective Adam first moment and let \(\bar g_r\) be the mean
native UNSB gradient on four independent unpaired shadow batches at the current
method state.  Native Adam moves along \(-q_r\).  A first-order compatible state
therefore satisfies \(q_r^\top\bar g_r\geq 0\).

LCNMP applies the Euclidean projection

\[
q_r^* = q_r +
\max\left(0,-\frac{q_r^\top\bar g_r}{\|\bar g_r\|^2}\right)\bar g_r.
\]

It writes back only
\(m_r^*=q_r^*(\sqrt{v_r}+\epsilon)\).  Parameters, \(v_r\), optimizer age,
schedulers, D/E state, samplers, RNG and global clock are unchanged.

## Properties

- **Least change:** \(q_r^*\) is the closest Euclidean point in the native
  non-ascent half-space.
- **Identity/self-null:** if the inherited moment is compatible, LCNMP is
  exactly identity.
- **No plain imitation:** neither a plain state nor a plain output is an input.
- **Target blind:** only unpaired native UNSB gradients are read.  The shadow
  batches and RNG are restored, so continuation receives the original clock.
- **One-time transport:** this is not a controller or an exit criterion.

## Falsifying experiment

From the same positive checkpoint and common future clock, compare hard disable
with LCNMP followed by native UNSB.  LCNMP is rejected for that mechanism if it
does not reduce the measured gradient--moment constraint violation, or if the
reduced violation fails to improve the registered 200/800-step continuation.
An identity result is evidence that optimizer tangent mismatch was absent; it
is not counted as an algorithmic win.

## Cost

Four additional G/F backward passes once at handoff; no inference cost and no
persistent model memory beyond one accumulated native gradient.
