# SEARCH-004 goal contract

## Scientific target

Construct and test a transition operator

\[
T:\;S^{i}_{k}\longmapsto \widetilde S_{k}
\]

such that a positive state produced by an UNSB intervention can be continued by
the native UNSB update map without an avoidable loss caused by incompatible
optimizer or coupled-network state.  `T` may modify optimizer tangent state or
D/E/F co-state, but it must preserve the proposed benefit carrier unless a
component ablation shows that carrier is harmful.

This is a state-transport problem.  It is not a search for a better intervention
algorithm, a better fixed exit update, or a controller that predicts paired
quality.

## Non-negotiable exclusions

- No paired PSNR, SSIM, LPIPS, clean target, or confirmation20 sample may enter
  `T`, its stopping rule, or its diagnostics.
- No branch may be chosen online from paired quality.
- No objective may force the method output or parameters toward the plain
  branch.  A matched plain state is a causal control, never a training target.
- Fixed horizons may define an audit treatment and post-hoc evaluation point;
  they may not be presented as the mathematical mechanism of `T`.
- A successful short handoff may not be called sustained until it passes the
  registered long matched continuation.
- DT, HJ, HNEK and PCOA are evidence sources.  None receives a protected final
  slot.

## Allowed information

`T` may read the full current training state and target-blind native quantities:
G/F/D/E parameters, named optimizer and scheduler state, method co-state,
samplers/RNG/global clock, unpaired native gradients, native losses, bridge
coordinates, update norms, gradient--moment angles, and cross-batch tangent
agreement.

## Required causal separation

Every source is first compared under:

1. uninterrupted intervention;
2. hard disable with the complete method state;
3. component transport or repair;
4. matched common-clock plain.

An operator is generated only if a target-blind defect is measured and the
operator reduces that same defect.  A plain-relative distance may be reported
as a diagnostic, but reducing it is never a success condition.

## Candidate requirements

Each candidate must have a derivation card stating:

- which checkpoint evidence generated it;
- the exact state component it transports;
- a least-change, identity, self-null, or unbiased property;
- why it does not imitate plain;
- why paired targets are inaccessible;
- a falsifying experiment;
- compute and memory cost.

Promotion requires both target-blind defect reduction and post-branch matched
quality.  The latter can select a candidate for a longer experiment but cannot
change its formula.

## End state

SEARCH-004 ends with exactly one of:

- a route-2 candidate that passes the registered local long continuation and is
  frozen for 4090 validation; or
- a falsification identifying whether optimizer transport, coupled co-state
  transport, or state handoff itself failed to improve over hard disable.

In both cases confirmation20 remains sealed and the result must explicitly
separate transition loss from later native-flow convergence.
