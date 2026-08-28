# Historical handoff boundary

SEARCH-004 does not treat every prior experiment containing the word
"handoff" as evidence for or against full-state transport.

## July/August 4090 partial-handoff rescue

The old `partial_handoff_floor025/050/075` experiments retained a minimum HJ
multiplier while DT was active in the old joint-training package.  The selected
`floor075` branch reported pooled PSNR +0.132 dB over HJ, but SSIM and LPIPS were
worse and the adjudication remained `JOINT_NONINFERIOR_MEAN_ONLY`.  This was a
loss-weight/factor handoff, not a transport of canonical G/F/D/E/Adam state.

## July/August 4090 gradient guard

The old global and parameter gradient guards projected DT/HJ corrections during
joint training.  Both were `JOINT_NEEDS_RESCUE`; the better global guard was
about -1.47 dB below HJ.  SEARCH-004 therefore excludes an ongoing generic
gradient-guard candidate.  A one-time least-change projection of inherited
optimizer state is a distinct treatment and must be compared directly with
hard disable.

## Consequence

These results close factor-floor tuning and persistent generic projection under
their historical protocols.  They do not establish whether the clean
canonical UNSB state at an intervention checkpoint has an optimizer-tangent or
coupled-co-state incompatibility.  That question is answered only by the
common-clock component audit in this search.
