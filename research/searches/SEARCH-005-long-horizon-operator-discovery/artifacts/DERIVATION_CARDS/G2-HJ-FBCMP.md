# G2-HJ-FBCMP: Future-Batch Consensus Metric Projection

## Evidence route

- `G1-HJ-ACMP::step400=+0.560489::5/6::guardrail-pass`
- `G1-HJ-ACMP::step800=-0.640026::1/6`
- `G1-HJ-ACMP::all sampled native/Adv+SB metric alignments nonnegative`
- `G1-HJ-ACMP::antithetic energy ratio range 0.664-0.992`

## Mathematical update

Let c_k be the antithetic raw HJ correction on independent unpaired batch k and M_k the frozen Adam diagonal metric.  With the previous-batch field c_{k-1}, compute q_k=<c_{k-1},c_k>_M/(||c_{k-1}||_M||c_k||_M+eps), rho_k=[q_k]_+, and m_k=rho_k(c_{k-1}+c_k)/2.  Project m_k onto <g_UNSB,k,c>_M>=0 and <g_Adv+SB,k,c>_M>=0, apply it to G, then store c_k for the next independent batch.  The first update uses m_0=0.

## Derivation

1. G1 proves that same-batch half-space feasibility alone can coexist with a 400-to-800 PSNR reversal.
2. Consecutive shuffled unpaired batches are independent samples of the native training distribution, so positive metric cosine is a target-blind estimate of correction-field transportability.
3. The positive-part cosine is not an exit threshold: it acts per update and per correction field, continuously ranging from zero to one.
4. When two batches disagree, rho_k=0 and the HJ component self-nullifies exactly; when they agree, their mean reduces stochastic variance before the original ACMP safety projection.
5. No prior checkpoint, paired score, training age or fixed duration enters the operator.

## Long-horizon property

an HJ direction can accumulate only when it is both cross-batch transportable and locally feasible for the current native and Adv+SB objectives.

## Falsification

kill the HJ correction-field mechanism if the fixed 400/800 micro run is not positive at 800 or if consensus is almost always zero while no gain appears.
