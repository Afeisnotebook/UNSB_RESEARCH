# DEC-20260828 — SEARCH-004 route-2 sustained local candidate

## Decision

Close SEARCH-004 with `HJ1200-NATIVE-HANDOFF` as the unique first candidate
for frozen full100 4090 verification. Classify it `route2_sustained_local`, not
as a confirmed paper method.

The candidate uses plain UNSB on updates `[0,240)`, canonical Layer-0 HJ on
`[240,1200)`, and native UNSB thereafter. The handoff disables only the HJ
correction and preserves the complete reached training state.

## Evidence

- Matched discovery70 PSNR deltas at total steps 2400/2800/3200 are
  `+0.536/+2.133/+0.871 dB`.
- The late-three mean is `+1.180 dB`; the final point is positive in 6/6
  domains and its worst domain is `+0.264 dB`.
- Final SSIM improves by `+0.01965`; LPIPS improves by `0.01665` in the lower-is-
  better direction; late peak-to-final rollback is zero.
- All 22 engineering checks pass and confirmation20 remains sealed.

## Causal interpretation

The positive HJ state does not require a bridge-to-plain imitation loss or a
new state transport. Complete-state native handoff is better than keeping HJ
active, and component resets/transplants generally destroy useful coupled
state. DT is also naturally inheritable. HNEK has a transient handoff shock but
recovers by h800.

PCOA supplies the necessary counterexample: G2-VCMR repairs its moment defect
by `+1.626 dB` over raw handoff at h400, yet remains `-0.467 dB` versus plain.
Transition mismatch is therefore source-dependent and is not a universal cause
of reversal.

## Scope boundary

This decision does not claim a learned or target-blind exit rule, smooth
monotone quality, cross-seed stability or confirmation performance. The
intervention interval comes from prior development evidence. The only next
authorized result-bearing run is a fresh, matched full100 seed-2026 4090 test
with exposure-normalized HJ `[960,4800)` and frozen 30k/60k/120k milestones.
No algorithm or interval change is allowed after observing intermediate
metrics. Additional seeds and confirmation20 remain downstream.

Primary evidence:

- `research/searches/SEARCH-004-gap-aware-handoff/RESULTS.md`
- `research/searches/SEARCH-004-gap-aware-handoff/artifacts/CANDIDATE.json`
- `research/searches/SEARCH-004-gap-aware-handoff/artifacts/HANDOFF_CAUSAL_MATRIX.json`
- `research/searches/SEARCH-004-gap-aware-handoff/artifacts/ROUTE2_ADJUDICATION.json`
