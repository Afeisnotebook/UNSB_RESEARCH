# SEARCH-004 reproduction

Run all commands from the repository root. The committed artifacts are compact
adjudications; full checkpoints and image-level evaluation records remain under
`E:\UNSB_Expl\runs\gap_aware_handoff_20260827`.

## Engineering gate

```powershell
python research/searches/SEARCH-004-gap-aware-handoff/run_search.py --stage gate
```

The gate must report `PASS` before any causal or long branch is interpreted.

## Reproduce the local long comparison

The command below resumes the already registered exact total-step-2000 states.
It adds 1200 common-clock updates, evaluating discovery70 every 400 updates.

```powershell
python research/searches/SEARCH-004-gap-aware-handoff/run_search.py `
  --stage long `
  --checkpoint-id HJ-HANDOFF-2000 `
  --long-horizon 1200 `
  --long-eval-interval 400 `
  --long-arms P_common_plain,A_hard_disable
```

This is an audit replay, not an independent from-e0 confirmation.

## Frozen 4090 full-view protocol

The existing SEARCH-002 runner implements exposure-normalized finite HJ:
full100 uses 600 updates per data epoch, so HJ is active on `[960,4800)` and
native UNSB is used thereafter. The candidate and plain are freshly initialized
and matched. Do not alter the candidate after observing 30k or 60k.

```powershell
python research/searches/SEARCH-002-dthj-rederivation/run_search.py `
  --stage verify4090 `
  --seed 2026 `
  --verify-milestones 30000 60000 120000 `
  --output runs/search004_verify4090_seed2026
```

If seed 2026 remains positive at the frozen late milestones, rerun the identical
command with `--seed 2027` and a distinct output directory. Add seed 2028 only
if signs disagree or the combined gain is below the preregistered threshold.
`confirmation20` remains sealed until the candidate and all seed decisions are
frozen.
