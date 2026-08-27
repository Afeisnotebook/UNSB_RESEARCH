# G1-HJ-ACMP: Antithetic Constrained Metric Projection

## Evidence route

- `HJ-SMALL-1200-POS::pulse8=+0.550390`
- `HJ-SMALL-1200-POS::H200=+0.064069`
- `HJ-SMALL-1200-POS::gap_ratio=12.985::direction_cosine=0.036`
- `HJ-SMALL-1200-POS::variance_fraction=0.857968`
- `HJ component attribution::all arms negative at native horizon 32`

## Mathematical update

Let c_bar be the average HJ-minus-plain gradient correction from antithetic latent views z and -z.  Project c_bar continuously in the Adam metric onto C={c:<g_UNSB,c>>=0 and <g_Adv+SB,c>>=0}, then bound its metric norm by ||g_UNSB||.  Set g_new=g_UNSB+c_projected.

## Derivation

1. Because z and -z have the same marginal law, antithetic averaging preserves the mean raw HJ correction while cancelling odd latent variation.
2. Metric projection changes the correction geometrically instead of accepting or rejecting a whole branch.
3. The two half-space constraints make the correction a first-order descent contribution for both the full native objective and its adversarial/SB component.
4. The native-norm trust region prevents a high-variance PatchNCE correction from dominating the coupled G/D/E dynamics.
5. If the raw correction already satisfies the constraints it is retained; if not, it is rotated to the closest feasible correction rather than delayed by a schedule.

## Long-horizon property

every applied correction is continuously constrained to native and bridge/adversarial descent half-spaces and cannot exceed the native update scale.

## Falsification

kill if antithetic variance is not reduced, constraints fail numerically, or the 800-step micro run still shows strong rotation/amplification with no positive trajectory.
