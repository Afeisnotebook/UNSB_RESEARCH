# G1-HJ-RHDFC8: Receding-Horizon Native-Discriminator Consensus for HJ

Parent probe: `hj`.

## Evidence

`loss_delta::D_fake` at H=8 predicts H=200: BA=0.750, Spearman=0.429, domains=4/6.

## Operator

From the same immutable full state S_k, compute B0=Phi_0^8(S_k) and Bi=Phi_hj^8(S_k). Let Delta=D_fake(Bi)-D_fake(B0). Commit Bi iff Delta>0; otherwise commit B0.

## Identity / self-null

Delta_D_fake <= 0 commits the exact plain branch.
a rejected proposal contributes zero parameters, moments, schedulers, streams or RNG state.

## Falsification

proposal-only, observable-only/plain, and full selector from the same e0; kill if the selector chooses harmful H200 branches or fails the 2400-step late gate.
