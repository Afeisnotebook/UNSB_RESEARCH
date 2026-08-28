# SEARCH-004: Gap-aware handoff (route 2)

SEARCH-004 asks where a positive intervention state lives and whether native
UNSB can inherit it through a target-blind state-transport operator.  The
carrier of benefit may be G/F parameters, optimizer tangent state, D/E/F
co-state, or a combination.  The transported state is judged by whether native
UNSB can continue from it, not by whether its output or parameters resemble a
matched plain branch.

The route is downstream of SEARCH-005.  SEARCH-005 established that no tested
continuous operator had smooth sustained benefit and that correction
propagation is heterogeneous: HJ and HNEK do not store a useful perturbation in
the same components.  SEARCH-004 therefore treats DT/HJ/HNEK/PCOA checkpoints
as causal sources, not as a protected candidate list.

The staged entry point is `run_search.py`. Full local run state is written
under `E:\UNSB_Expl\runs\gap_aware_handoff_20260827`; adjudicated,
machine-readable summaries are committed under `artifacts/`.
`GOAL_CONTRACT.md` is the authority if an implementation choice is ambiguous.

SEARCH-004 is complete. The local winner is `HJ1200-NATIVE-HANDOFF`: run HJ
only on updates `[240,1200)`, then continue native UNSB from the complete HJ
state without resetting parameters, Adam moments, schedulers, co-state, data
stream, or RNG. Across total steps 2400/2800/3200 its matched discovery70
PSNR deltas were `+0.536/+2.133/+0.871 dB`; the late-three mean was
`+1.180 dB`, the final point was positive in 6/6 domains, and SSIM/LPIPS
guardrails passed. See `RESULTS.md` for the evidence boundary and
`REPRODUCE.md` for exact commands.
