# SEARCH-005 final route-1 report

## Outcome

No tested mathematical operator produced a sustained local win. Positive windows are real and reproducible, but every admissible mechanism either failed by 800 updates or reversed by the independent 2400-update trajectory. No full-view, second-seed or confirmation20 experiment was opened.

The frozen weak fallback is **G1-GAME-PCOA**, not because it passed, but because it was the only new route-1 operator positive at 400/800/1200 and the only one that completed 2400. It reversed at 1600 (-0.570 dB) and 2400 (-0.861 dB), so it cannot support a sustained-method claim.

## Ranking

| Rank | Candidate | Horizon | Late available mean ΔPSNR | Final ΔPSNR | Final positive domains | Promoted |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | G1-GAME-PCOA | 2400 | +0.124 | -0.861 | 2/6 | no |
| 2 | G1-DT-CNDRP | 800 | -0.011 | -0.005 | 4/6 | no |
| 3 | G1-HJ-ACMP | 800 | -0.040 | -0.640 | 1/6 | no |
| 4 | G2-HNEK-PHRSUP | 800 | -0.144 | -0.645 | 2/6 | no |
| 5 | G2-DT-BCNRP | 800 | -0.212 | -0.470 | 1/6 | no |
| 6 | G1-HNEK-PHCRP | 800 | -0.273 | -0.579 | 1/6 | no |
| 7 | G2-GAME-NPOOA | 800 | -0.367 | -0.966 | 1/6 | no |
| 8 | G2-HJ-FBCMP | 800 | -0.498 | -0.659 | 1/6 | no |
| 9 | G1-HNEK-ELIPRC | 800 | -1.093 | -2.053 | 1/6 | no |

## Causal conclusions

- DT-style sensitivity preconditioning can be nearly neutral but does not preserve structural quality; the block-safe revision worsened late PSNR.
- HJ-style projected correction produces a strong early window, but independent future-batch consensus removes rather than stabilizes the benefit.
- HNEK-style physical coordinate defects are not sufficient predictors: direct path correction damages content and the native gradient often already reduces the measured defect.
- Coupled-game optimism changes the phase and yields reproducible positive windows. Removing radial amplification strengthens 400-step quality but causes a larger 800-step reversal, so predictable angular motion alone is not a safe long-run correction.
- Historical HNEK repeatedly changes sign through 12k. This is evidence for an oscillatory endogenous training distribution, not evidence for a single correct exit point.

## Honest boundary

SEARCH-005 exhausted its registered six Generation-1 mechanisms and one causal revision per failure class. It did not drift into fixed-window, paired-PSNR, whole-branch selection or handoff optimization. A route-2 handoff study would need a separate plan and claim.
