# SEARCH-003 scope correction

Date: 2026-08-27

SEARCH-003 produced valid engineering and counterfactual evidence, but its
executed candidate family was materially narrower than the registered research
objective. Both generations were state-conditional whole-branch selectors:

- Generation 1 selected an eight-update DT or HJ branch from an unpaired loss
  sign;
- Generation 2 added a next-independent-batch native-gradient condition before
  selecting the same branch.

These experiments answer whether target-blind evidence can select the current
DT/HJ intervention. They do **not** exhaust long-horizon algorithm discovery.
In particular, SEARCH-003 did not complete an evidence-routed search over:

- new UNSB update operators;
- unbiased time/domain/latent estimators;
- bridge-coordinate or physical-horizon reparameterizations;
- moving-reference rate or curvature constraints;
- adaptive rollout/teacher dynamics;
- coupled generator/discriminator/encoder/optimizer-state dynamics.

Therefore `ROUTE1_STOP.json` is a stop record for the controller subroute, not
for route-1 mathematical operator discovery. The evidence supports only:

> Two generations of target-blind DT/HJ whole-branch selection did not retain
> a smooth long-horizon benefit under the frozen local protocol.

It does not support any of the following claims:

- all algorithms fail to retain long-horizon gains;
- DT, HJ or HNEK mechanisms are falsified;
- route 1 has been exhausted;
- gap-aware handoff must be the next research route.

The original files remain in Git history. SEARCH-005 consumes the valid atlas
and engineering machinery while enforcing a separate operator-discovery
contract.
