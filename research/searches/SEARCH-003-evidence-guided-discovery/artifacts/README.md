# SEARCH-003 artifact bundle

This directory is a checkpoint-free snapshot of the local SEARCH-003 run.

- `REVERSAL_ATLAS.jsonl` contains target-blind counterfactual observations and
  post-branch development labels.  Labels are explicitly unavailable to the
  controller.
- `REVERSAL_ANALYSIS*.json` and `GENERATION0_ADJUDICATION.*` contain signal
  screening and cause routing.
- `DERIVATION_CARDS/` and `HYPOTHESIS_LEDGER.json` preserve candidate lineage,
  revisions and closures.
- `ENGINEERING_GATE.json` records exact identity/resume/branch-isolation gates.
- `LOCAL_RANKING.json`, `PER_DOMAIN_TRAJECTORY.json` and
  `ABSOLUTE_RELATIVE_DECOMPOSITION.json` contain the frozen local comparison.
- `CANDIDATE.json`, `BACKUP_CANDIDATES.json` and `ROUTE1_STOP.json` are the
  final machine-readable decision.

The bundle intentionally excludes multi-gigabyte model checkpoints and the
raw materialized dataset.  Their identities and local paths are frozen in
`PROTOCOL_LOCK.json` and `CHECKPOINT_INVENTORY.json`.  `confirmation20` was not
opened.
