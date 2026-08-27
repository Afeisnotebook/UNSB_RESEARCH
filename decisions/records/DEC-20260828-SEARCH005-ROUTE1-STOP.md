# DEC-20260828 — SEARCH-005 route-1 stop

## Decision

Close the registered route-1 mathematical-operator search without a sustained
candidate. Freeze `G1-GAME-PCOA` as the unique `weak_fallback`, not as a paper
method and not as an automatic 4090 promotion.

## Evidence

- Six Generation-1 failure mechanisms and four causal revisions were tested.
- PCOA was positive at 400/800/1200 (`+0.044/+0.075/+0.193 dB`) but negative at
  1600/2400 (`-0.570/-0.861 dB`), with about 2.10 dB absolute rollback.
- NPOOA exactly preserved each native G/D/E Adam displacement norm and improved
  400-step PSNR by `+0.231 dB`, but reversed to `-0.966 dB` at 800.
- No SEARCH-005 candidate passed the 2400-step promotion gate; full100,
  additional seeds and confirmation20 were not opened.

## Interpretation

The historical positive windows are not dismissed. The result is narrower:
the tested target-blind self-null/invariant operators did not convert those
windows into smooth long-horizon gain. The search did not fit an exit threshold
or use paired restoration metrics as controller inputs.

## Next authority boundary

A full100 PCOA command exists only for explicit weak-fallback falsification. A
gap-aware handoff is route 2 and requires a separate approved plan. SEARCH-005
must not be silently reopened as a window or threshold search.

Primary evidence:

- `research/searches/SEARCH-005-long-horizon-operator-discovery/RESULTS.md`
- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/CANDIDATE.json`
- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/ROUTE1_STOP.json`
- `research/searches/SEARCH-005-long-horizon-operator-discovery/artifacts/HYPOTHESIS_LEDGER.json`
