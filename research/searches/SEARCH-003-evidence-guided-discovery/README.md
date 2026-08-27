# SEARCH-003: evidence-guided long-horizon algorithm discovery

SEARCH-003 does not preregister DT-rate, audited-HJ, equilibrium-HNEK or MPVR
as algorithms to validate.  It preregisters the evidence contract that is
allowed to *create* an algorithm.

The first executable output is a reversal atlas built from the preserved
SEARCH-001/002 full-state checkpoints.  Historical methods are probes of
different state factors; candidates are admitted only after a target-blind
precursor, a mathematical identity/self-null condition and a falsification
test have been written to `HYPOTHESIS_LEDGER.json`.

Paired discovery metrics may label a completed counterfactual branch.  They
are structurally absent from `StateObservation`, `InterventionProposal` and
the online controller.  `confirmation20` remains sealed throughout the local
search.

## Frozen local outcome (2026-08-27)

The engineering gate passed, including exact zero-intervention/plain identity,
full controller resume and transactional virtual branches.  Generation 0
found no legal cross-method shared precursor.  Method-specific evidence
created two eight-update receding-horizon candidates:

- DT used the sign of the unpaired `G_GAN` branch difference;
- HJ used the sign of the unpaired `D_fake` branch difference.

Both Generation-1 selectors were positive at 400 updates and negative at 800.
Generation 2 therefore added a one-sided future-batch condition
`<theta_proposal-theta_plain, grad L_UNSB(next batch)> < 0`.  This changed HJ's
800-update result from -1.543 dB to +0.224 dB, but the full selector reversed
again after 800 and failed the frozen 2400-update promotion gate.

The best completed lane was the required `proposal_only` ablation of the HJ
parent: late-three mean +0.945 dB and final +1.045 dB.  It is frozen only as a
`weak_fallback`, because coverage, worst-domain and absolute-retention gates
failed (rolling rollback 0.814 dB).  SEARCH-003 therefore did not start the
full-view or extra-seed stages and does not approve a 4090 run.

See [RESULTS.md](RESULTS.md) and the generated files under `artifacts/`.  The
raw training checkpoints remain under
`E:\UNSB_Expl\runs\evidence_guided_discovery_20260827` and `confirmation20`
was never opened.

## Reproduction

```powershell
E:\conda\python.exe research\searches\SEARCH-003-evidence-guided-discovery\run_search.py --stage gate
E:\conda\python.exe research\searches\SEARCH-003-evidence-guided-discovery\run_search.py --stage candidate --candidate G2-HJ-FBDFC8 --candidate-mode proposal_only --candidate-steps 2400
E:\conda\python.exe research\searches\SEARCH-003-evidence-guided-discovery\run_search.py --stage report --candidate-steps 2400
```
