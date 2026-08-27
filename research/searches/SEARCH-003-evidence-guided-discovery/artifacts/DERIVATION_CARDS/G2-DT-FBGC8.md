# G2-DT-FBGC8: Future-Batch Gradient Consensus for DT

## New failure evidence

G1-DT-RHGC8 changed from +0.104052 dB at 400 to -0.381266 dB at 800 despite committing 3/100 proposal blocks.

## Operator

Run B0=Phi_0^8(S) and Bi=Phi_i^8(S). Let delta be the G/F parameter correction theta_i-theta_0 and let g+ be the native UNSB generator gradient at B0 on the next independent unpaired batch. Commit Bi only when -1*(Delta G_GAN)>0 and <delta,g+><0.

## Exact self-null

rejected proposal and audit computations contribute no parameters, moments, scheduler, stream, RNG or method co-state.

## Falsification

from the same e0, kill after the 800-update micro run if final delta is non-positive or guardrails fail; no further revision is allowed.
