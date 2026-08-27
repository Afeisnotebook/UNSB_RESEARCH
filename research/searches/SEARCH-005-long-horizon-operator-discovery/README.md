# SEARCH-005: long-horizon operator discovery

SEARCH-005 continues route 1 from the valid SEARCH-003 evidence while correcting
its candidate-class drift. Its objective is to discover and reconstruct a new
UNSB update operator, estimator, coordinate system or coupled training dynamics
whose beneficial action remains mathematically valid over long training.

This search is not allowed to become an exit-policy search. Fixed windows,
annealing used only to postpone failure, paired-metric control, whole-state
plain/proposal branch selection and gap-aware handoff are outside its route.

The search begins by separating two causes that previous work conflated. For a
probe update

\[
u_i(S) = u_0(S) + c_i(S),
\]

it asks whether the correction field `c_i` has already become invalid at the
current state, or whether a locally beneficial correction is destroyed by the
subsequent native UNSB flow `Phi_0`. Existing DT/HJ/HNEK checkpoints are probes,
not preregistered candidates.

No long retraining is permitted until the causal audit produces a derivation
card and the candidate passes the automated claim contract. See
[GOAL_CONTRACT.md](GOAL_CONTRACT.md) and
[CAUSAL_AUDIT_PROTOCOL.md](CAUSAL_AUDIT_PROTOCOL.md).

## Final route-1 result (2026-08-28)

The registered search is complete. Six Generation-1 failure mechanisms and
four evidence-triggered revisions passed engineering gates and were tested.
None passed the 2400-update sustained-benefit promotion gate. PCOA is frozen as
the unique `weak_fallback`: it was reproducibly positive at 400/800/1200, but
reversed at 1600 and 2400. Its norm-preserving NPOOA revision was +0.231 dB at
400 and -0.966 dB at 800, closing the coupled-game mechanism after two
generations.

This result does not falsify every DT/HJ/HNEK idea and does not authorize an
exit-threshold search. It says that no tested target-blind, self-null or
invariant route-1 operator retained the local signal. See [RESULTS.md](RESULTS.md),
[`CANDIDATE.json`](artifacts/CANDIDATE.json), and
[`ROUTE1_STOP.json`](artifacts/ROUTE1_STOP.json).

```powershell
# exact local weak-fallback reproduction
python research/searches/SEARCH-005-long-horizon-operator-discovery/run_search.py --stage small

# valid but conditional full100 entrypoint; not an automatic promotion
python research/searches/SEARCH-005-long-horizon-operator-discovery/run_search.py --stage full --gpu 0 --seed 2026
```
