# G1-HNEK-ELIPRC: Endpoint-Law-Invariant Physical Residual Coordinate

## Evidence route

- `HNEK-FULL-3000-POS::pulse8=+0.869870::H32=+1.213187::H200=-0.071773`
- `HNEK-FULL-4000-NEG::pulse8=-0.058180::H200=-0.336928`
- `HNEK-FULL-3000-POS::GF+moments attribution=+1.641274::6/6`
- `HNEK-FULL-3000-POS::full-state worst-domain=-2.154811`

## Mathematical update

Keep Y=G_theta(X_t,z) exactly unchanged.  For h=1-t>0 present the entropy critic with R_h=(Y-X_t)/sqrt(h), use the physical entropy coefficient h from the restricted UNSB objective, and leave GAN, PatchNCE, transport cost and rollout/inference endpoint transitions unchanged.

## Derivation

1. UNSB Theorem 1 restricts the bridge to [t,1], whose Brownian transition variance scales with the remaining horizon h.
2. sqrt(h) is therefore the physical fluctuation scale, while R_h is an invertible coordinate of Y given X_t and h.
3. Mutual information is invariant under this conditional bijection, so an ideal entropy critic may use R_h without changing the endpoint conditional law.
4. Unlike HNEK-all, the generator forward is never multiplied by h^gamma; training, rollout and inference share the exact same endpoint map.
5. The physical coefficient h replaces the code's uniform-index coefficient, correcting the identified nonuniform-time mismatch without a schedule.

## Long-horizon property

the endpoint law is exactly identical to plain at every state, while the continuously active entropy coordinate is an invertible physical reparameterization.

## Falsification

kill if forward/rollout endpoint identity fails or if a 400/800-step micro run reproduces HNEK-all reversal without improving coordinate conditioning.
