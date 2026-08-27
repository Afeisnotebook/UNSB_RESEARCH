# SEARCH-005 causal matrix

The eight-step pulse is a finite-difference diagnostic, not an activation window.

| checkpoint | historical | immediate pulse | final native | domains | worst | gap ratio | direction cosine | class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DT-FULL-3000-NEG | -0.806 | -0.011 | +0.905 | 3/6 | -3.155 | 7.05 | 0.128 | negative_impulse_rotated_by_native_flow |
| DT-FULL-4000-NEG | -1.519 | +0.000 | +0.000 | 0/6 | +0.000 | 0.00 | 0.000 | operator_exactly_null |
| HJ-SMALL-1200-POS | +0.805 | +0.550 | +0.064 | 5/6 | -0.106 | 12.99 | 0.036 | positive_impulse_strongly_attenuated |
| HJ-SMALL-800-NEG | -0.724 | +0.031 | +0.033 | 4/6 | -0.288 | 9.78 | 0.103 | positive_impulse_retained |
| HNEK-FULL-3000-POS | +0.954 | +0.870 | -0.072 | 2/6 | -0.444 | 10.45 | 0.042 | positive_impulse_reversed_by_native_flow |
| HNEK-FULL-4000-NEG | -0.712 | -0.058 | -0.337 | 2/6 | -1.329 | 13.30 | 0.041 | negative_impulse_remains_harmful |

## Evidence-routed failure classes

### FC-TRANSPORT-ROTATION-AMPLIFICATION — supported_cross_probe

native UNSB dynamics amplify the parameter-state gap while losing the initial beneficial correction direction.

Routing: `coupled_dynamics_or_stability_constrained_operator`.

### FC-POSITIVE-IMPULSE-NOT-INVARIANT — supported

an immediately beneficial correction is not an invariant or contractive direction of the later native flow.

Routing: `invariant_preserving_operator_not_exit_policy`.

### FC-COUPLED-COSTATE — heterogeneous_probe_specific

co-state propagation is mechanism-specific: HJ transfers become harmful, while HNEK G/F parameters plus moments preserve a strong short-horizon benefit.

Routing: `probe_specific_coupled_dynamics_not_unified_handoff`.

### FC-DT-INTERMITTENT-NULL — supported

the current DT statistic makes the correction exactly zero in some late states while remaining active in nearby states.

Routing: `audit_moving_statistic_before_derivation`.

### FC-STOCHASTIC-VARIANCE — supported_method_specific

HJ step1200 and DT step4000 are variance dominated across independent unpaired batches; HNEK and DT step3000 remain elevated but below the frozen threshold.

Routing: `unbiased_or_mean_preserving_variance_reduction_only_for_supported_probes`.

## Current gate

Candidate generation remains blocked until the listed causal audits are complete. No controller, schedule, handoff or fixed-window method is admitted by this matrix.
