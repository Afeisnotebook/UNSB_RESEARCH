# SEARCH-003 local result

## Decision

SEARCH-003 closed the current route-1 protocol without a sustained candidate.
The unique frozen output is `G2-HJ-FBDFC8::proposal_only`, classified
`weak_fallback`; it is not promoted to the full 100-image/domain view, extra
seeds or 4090 validation.

This is not a no-signal result.  It distinguishes a repeatable early benefit
from a sustained benefit and preserves the former without overstating the
latter.

## Main evidence

All deltas below are six-domain macro PSNR against the matched plain state.

| lane | 400 | 800 | 1200 | 1600 | 2000 | 2400 |
|---|---:|---:|---:|---:|---:|---:|
| G1 DT selector | +0.104 | -0.381 | — | — | — | — |
| G1 HJ selector | +0.262 | -1.543 | — | — | — | — |
| G2 DT + future gradient | +0.104 | -0.381 | — | — | — | — |
| G2 HJ + future gradient | +0.502 | +0.224 | -0.298 | -1.539 | +0.852 | -0.607 |
| HJ proposal-only ablation | +0.339 | -0.041 | +0.673 | -0.727 | +2.518 | +1.045 |

The G2 HJ full selector is the cleanest causal result: adding the future native
gradient condition changed the parent's 800-step result from -1.543 dB to
+0.224 dB, so it delayed and reshaped the reversal.  It did not eliminate the
reversal: the late-three mean was -0.431 dB and final delta -0.607 dB.

The proposal-only lane has late-three mean +0.945 dB and final +1.045 dB, but
fails three preregistered conditions:

- domain coverage at the required late checkpoints;
- worst-domain guardrail;
- absolute retention (0.814 dB rolling rollback versus a 0.3 dB limit).

Its large +2.518 dB result at 2000 also coincides with matched plain falling to
11.861 dB, so relative improvement alone is not treated as sustained quality.

## Scientific interpretation

The experiments reject two current *protocols*, not the broad DT or HJ ideas.
An in-branch unpaired loss sign was insufficient, regardless of whether it
created sparse DT intervention or frequent HJ intervention.  Requiring the
proposal correction to be a descent direction for the next-batch native UNSB
gradient improved the HJ trajectory through 800, but native-objective descent
was still not equivalent to persistent paired restoration quality.

Therefore the honest stop category is:

`correction_valid_on_unpaired_native_objective_but_not_equivalent_to_psnr`.

Per the two-generation limit, no third controller, threshold, fixed window or
PSNR-driven exit was fitted.  A gap-aware handoff is a separate route-2 task,
not a hidden continuation of SEARCH-003.

## Reproducibility and access controls

- source anchor: `649b7ec2bd520ccb174b73c5b5187f7ce08ebb22`;
- manifest SHA-256: `1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b`;
- local seed: `2026`;
- small view: 25 training images/domain;
- evaluation: discovery10 only;
- `confirmation20`: sealed;
- candidate controller paired-target access: false;
- plain twin, exact resume, zero intervention and virtual-branch isolation:
  passed.

Machine-readable evidence is stored under `artifacts/`; full checkpoints and
raw local run state are intentionally not committed.
