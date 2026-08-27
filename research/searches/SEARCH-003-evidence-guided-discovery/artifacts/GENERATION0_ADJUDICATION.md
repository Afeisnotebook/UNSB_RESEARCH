# SEARCH-003 Generation 0 causal adjudication

Complete decisive grid: **True**.
Paired development labels were joined only after each branch and are not controller inputs.

## DT

Verdict: `reversal_observed_operator_rewrite_route`.

- pre full@2000: plain-state -0.7818 dB, own-state +0.0000 dB, `operator_locally_harmful`.
- near full@3000: plain-state +0.0567 dB, own-state +0.6879 dB, `operator_locally_sustainable`.
- post full@4000: plain-state -1.7666 dB, own-state -0.3120 dB, `operator_locally_harmful`.

## HJ

Verdict: `reversal_observed_state_feedback_route`.

- pre small@400: plain-state +0.5941 dB, own-state -0.2933 dB, `state_feedback_missing`.
- near small@800: plain-state +0.1922 dB, own-state +0.2392 dB, `operator_locally_sustainable`.
- post small@1200: plain-state +0.0225 dB, own-state -0.2103 dB, `state_feedback_missing`.

## HNEK

Verdict: `state_dependent_route`.

- pre full@2000: plain-state -1.1641 dB, own-state +0.7140 dB, `benefit_requires_method_state`.
- near full@3000: plain-state -0.8267 dB, own-state +0.1724 dB, `benefit_requires_method_state`.
- post full@4000: plain-state -1.2272 dB, own-state +1.0079 dB, `benefit_requires_method_state`.

## Signal gate

Shared signal passed: **False**.
Consequence: `shared_controller_route_closed_unless_decisive_audit_adds_evidence`.
