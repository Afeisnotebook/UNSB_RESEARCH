# SEARCH-004 local gap-aware handoff report

SEARCH-004 separated four questions that had previously been conflated:
whether a positive state is inheritable, which components carry its benefit,
whether a target-blind transition defect exists, and whether repairing that
defect improves matched continuation.

The unique winner is finite HJ navigation followed by unmodified native UNSB.
Its total-step 2400/2800/3200 matched PSNR deltas are
`+0.536/+2.133/+0.871 dB`; the late-three mean is `+1.180 dB`, the final point
is positive in all six domains, and SSIM/LPIPS pass. LCNMP and VCMR remove their
declared optimizer defects but do not beat the untouched complete-state handoff.

The result is local and single-seed. It authorizes one frozen full100 4090
verification; it does not authorize interval tuning, confirmation20 access or
a paper-level robustness claim.
