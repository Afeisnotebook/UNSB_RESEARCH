# SEARCH-004 final route-2 report

## Outcome

Route 2 found a sustained local candidate, but not a new state-transport
operator. The winning procedure is **HJ1200-NATIVE-HANDOFF**: canonical UNSB
for updates `[0,240)`, Layer-0 HJ steering for `[240,1200)`, then native UNSB
forever from the complete reached state. At handoff, only the HJ correction is
disabled; G/F/D/E, Adam moments and ages, schedulers, sampler position and RNG
are preserved exactly.

This corrects an important premise from the route-2 entry discussion: positive
intervention states are not generally incompatible with native UNSB. DT and HJ
are naturally inheritable. HNEK has a short transition shock but later recovers.
PCOA is the counterexample: repairing its optimizer shock does not make its
continuation positive.

## Winning trajectory

All numbers below are matched against a common-clock plain branch on the sealed
discovery70 split, seed 2026, small25 training view.

| Total step | Native updates after HJ handoff | Plain PSNR | Candidate PSNR | Delta | Positive domains | Worst domain |
|---:|---:|---:|---:|---:|---:|---:|
| 2400 | 1200 | 12.039 | 12.574 | +0.536 | 6/6 | +0.147 |
| 2800 | 1600 | 12.244 | 14.377 | +2.133 | 5/6 | -0.060 |
| 3200 | 2000 | 13.664 | 14.536 | +0.871 | 6/6 | +0.264 |

The late-three mean is `+1.180 dB`. At total step 3200, SSIM improves from
`0.4485` to `0.4682`, LPIPS improves from `0.5951` to `0.5784`, all six domains
are positive, and late peak-to-final rollback is `0`. The registered local long
gate therefore passes.

This does **not** mean the absolute candidate trajectory is monotone: its PSNR
falls after the strong total-step-2000 state and later recovers. The supported
claim is matched benefit preservation through 2000 native continuation updates,
not monotonic image quality.

## What the causal audit changed

| Source | Direct native handoff | Uninterrupted method | Transition repair | Decision |
|---|---:|---:|---:|---|
| DT-400, h800 | +0.477 | +0.756 | no repair promoted | naturally inheritable; short-only backup |
| HJ-1200, h800 | +3.792 | +1.129 | LCNMP only +0.032 over hard at h200 | direct handoff wins |
| HNEK-3000, h800 | +1.217 | +0.260 | LCNMP/VCMR worse at h200 | shock is transient; SSIM guardrail fails |
| PCOA-1200, h400 | -2.093 | -0.487 | VCMR -0.467 | repair is real but cannot restore benefit |

The experiments reject a universal “algorithm/plain state gap” explanation.
For HJ, overwriting or equilibrating components damages the useful coupled
state. For PCOA, inherited Adam orientation does create a transition defect,
yet correcting it still leaves the branch below plain. State mismatch can be a
secondary failure mode; it is not the universal cause of reversal.

## New mathematical constructions

Two target-blind transports were derived and implemented:

- **G1-LCNMP** projects inherited effective Adam first moments onto the current
  native-gradient half-space with the least Euclidean change. It removes its
  stated defect exactly, but provides no sustained gain over untouched handoff.
- **G2-VCMR** is the one permitted evidence-driven revision. When the inherited
  effective moment opposes the native gradient, it zeros the first moment while
  retaining the second moment and optimizer age. It rescues PCOA by `+1.626 dB`
  over raw handoff at h400, but remains `-0.467 dB` versus plain.

Both are closed under the current protocol. This is not because their
mathematics failed: the targeted defect was removed. They are closed because
removing that defect did not produce the required matched outcome.

## Candidate status and boundary

`HJ1200-NATIVE-HANDOFF` is classified `route2_sustained_local`. It is the only
candidate that passed the registered local long gate and is therefore the only
method to send first to 4090.

It is not yet a paper-level result. The evidence is one seed, small25, and a
finite intervention interval inherited from prior discovery. It is a fixed
stage treatment, not a paired-PSNR controller and not a learned exit threshold.
A fresh full100 from-e0 matched run is mandatory; seed 2027/2028 and
confirmation20 remain downstream of that frozen test. Exact commands are in
`REPRODUCE.md`.

## Reproducibility and sealed data

The engineering gate passes all 22 registered checks, including exact twin and
resume behavior, nonpolluting counterfactual audits, component-safe transports,
full state restoration and target-blind schema rejection. `confirmation20` was
never opened.
