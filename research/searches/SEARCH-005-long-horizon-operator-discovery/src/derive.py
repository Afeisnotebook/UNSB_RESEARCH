"""Evidence-routed Generation-1 mathematical derivations."""

from __future__ import annotations

import json
from pathlib import Path

from .claim_guard import validate_derivation_card


def derivation_cards(matrix: dict) -> list[dict]:
    if matrix.get("candidate_generation_allowed") is not True:
        raise RuntimeError("causal matrix has not opened candidate generation")
    cards = [
        {
            "candidate_id": "G1-DT-CNDRP",
            "name": "Confidence-Normalized Dispersion-Rate Preconditioner",
            "candidate_type": "operator",
            "generation": 1,
            "parent_probes": ["dt"],
            "parent_evidence": [
                "DT-FULL-3000-NEG::variance_fraction=0.739145",
                "DT-FULL-4000-NEG::variance_fraction=0.804189",
                "DT-FULL-4000-NEG::4/8 exact-zero correction batches",
                "FC-DT-INTERMITTENT-NULL",
            ],
            "causal_failure_class": "variance_dominated_intermittent_absolute_teacher_regularizer",
            "unsb_object": "native generator gradient under an endpoint-dispersion sensitivity metric",
            "mathematical_update": (
                "For two independent antithetic U-statistic replicates s_r=log(eps+U_r), "
                "let a_r=grad_theta s_r and a_bar=(a_1+a_2)/2.  Per Adam "
                "coordinate j let v_j=(a_1j-a_2j)^2/2 and "
                "p_j=(v_j+eps_j)/(a_bar_j^2+v_j+eps_j), then use "
                "g_new,j=p_j*g0,j."
            ),
            "derivation": [
                "The old additive DT loss matches a frozen absolute log-U value and can remain biased after its early defect disappears.",
                "The new object uses endpoint dispersion only as a local metric on the native gradient; it never supplies a target value.",
                "Every p_j lies strictly in (0,1], so P=diag(p_j) is positive definite and commutes with Adam's positive diagonal preconditioner M.",
                "Therefore g0^T M P g0=sum_j M_j*p_j*g0,j^2 > 0 for g0 != 0, and P*g0=0 iff g0=0: native descent and stationary points are preserved in the implemented optimizer geometry.",
                "High estimator variance increases v_j and continuously returns that coordinate toward identity; no threshold or exit is used.",
            ],
            "long_horizon_property": (
                "positive-definite preconditioning preserves native UNSB stationary points and "
                "cannot turn the native gradient into an ascent direction"
            ),
            "identity_condition": "a_bar=0 gives P=I exactly",
            "self_null_condition": "dispersion sensitivity absent or variance-dominated makes the correction zero or continuously negligible",
            "unbiased_or_invariance_property": "the native objective is unchanged; only its positive-definite metric is changed",
            "paired_target_access": False,
            "paired_target_proof": "U_r uses current unpaired bridge states, generator latents and no target image",
            "uses_fixed_window": False,
            "uses_whole_state_branch_selection": False,
            "changes_training_target": False,
            "changes_gradient_estimator": False,
            "changes_endpoint_law": False,
            "expected_valid_states": "states with a reproducible endpoint-dispersion sensitivity direction",
            "falsification_test": "kill if P is not SPD/identity-safe or if a 400/800-step micro run does not reduce gap amplification without a positive trajectory signal",
            "compute_cost": "two antithetic dispersion probes plus one native generator backward; diagonal parameter-space preconditioning",
            "memory_cost": "two dispersion gradients and the native gradient",
            "allowed_claim": "tests whether DT evidence is useful as a state metric without an absolute teacher anchor",
            "forbidden_claims": [
                "posterior covariance estimation",
                "calibrated uncertainty",
                "all variance-dominated methods are fixed",
                "route1 is exhausted if this candidate fails",
            ],
        },
        {
            "candidate_id": "G1-HJ-ACMP",
            "name": "Antithetic Constrained Metric Projection",
            "candidate_type": "operator",
            "generation": 1,
            "parent_probes": ["hj"],
            "parent_evidence": [
                "HJ-SMALL-1200-POS::pulse8=+0.550390",
                "HJ-SMALL-1200-POS::H200=+0.064069",
                "HJ-SMALL-1200-POS::gap_ratio=12.985::direction_cosine=0.036",
                "HJ-SMALL-1200-POS::variance_fraction=0.857968",
                "HJ component attribution::all arms negative at native horizon 32",
            ],
            "causal_failure_class": "high_variance_hj_correction_not_stable_under_native_adversarial_flow",
            "unsb_object": "generator update jointly constrained by native total and adversarial-plus-SB gradients",
            "mathematical_update": (
                "Let c_bar be the average HJ-minus-plain gradient correction from antithetic "
                "latent views z and -z.  Project c_bar continuously in the Adam metric onto "
                "C={c:<g_UNSB,c>>=0 and <g_Adv+SB,c>>=0}, then bound its metric norm by "
                "||g_UNSB||.  Set g_new=g_UNSB+c_projected."
            ),
            "derivation": [
                "Because z and -z have the same marginal law, antithetic averaging preserves the mean raw HJ correction while cancelling odd latent variation.",
                "Metric projection changes the correction geometrically instead of accepting or rejecting a whole branch.",
                "The two half-space constraints make the correction a first-order descent contribution for both the full native objective and its adversarial/SB component.",
                "The native-norm trust region prevents a high-variance PatchNCE correction from dominating the coupled G/D/E dynamics.",
                "If the raw correction already satisfies the constraints it is retained; if not, it is rotated to the closest feasible correction rather than delayed by a schedule.",
            ],
            "long_horizon_property": (
                "every applied correction is continuously constrained to native and bridge/adversarial descent half-spaces and cannot exceed the native update scale"
            ),
            "identity_condition": "zero raw HJ correction produces exact plain UNSB gradient",
            "self_null_condition": "when structure conflict vanishes, c_bar=0 and the projection is exactly inactive",
            "unbiased_or_invariance_property": "antithetic averaging preserves the raw HJ correction expectation under symmetric latent sampling",
            "paired_target_access": False,
            "paired_target_proof": "all gradients use unpaired UNSB batches, source structure and internal latent/patch randomness",
            "uses_fixed_window": False,
            "uses_whole_state_branch_selection": False,
            "changes_training_target": False,
            "changes_gradient_estimator": True,
            "changes_endpoint_law": False,
            "expected_valid_states": "states where HJ has nonzero structural conflict but its raw correction is noisy or fights adversarial/SB balance",
            "falsification_test": "kill if antithetic variance is not reduced, constraints fail numerically, or the 800-step micro run still shows strong rotation/amplification with no positive trajectory",
            "compute_cost": "multiple generator backward passes and two-metric half-space projection; no duplicated optimizer branch",
            "memory_cost": "plain, bridge/adversarial and two HJ correction gradients",
            "allowed_claim": "tests a continuously safe reconstruction of the HJ correction field",
            "forbidden_claims": [
                "future PSNR prediction",
                "exit detection",
                "unbiasedness with respect to plain UNSB",
                "all HJ failures share one cause",
            ],
        },
        {
            "candidate_id": "G1-HNEK-ELIPRC",
            "name": "Endpoint-Law-Invariant Physical Residual Coordinate",
            "candidate_type": "coordinate",
            "generation": 1,
            "parent_probes": ["hnek"],
            "parent_evidence": [
                "HNEK-FULL-3000-POS::pulse8=+0.869870::H32=+1.213187::H200=-0.071773",
                "HNEK-FULL-4000-NEG::pulse8=-0.058180::H200=-0.336928",
                "HNEK-FULL-3000-POS::GF+moments attribution=+1.641274::6/6",
                "HNEK-FULL-3000-POS::full-state worst-domain=-2.154811",
            ],
            "causal_failure_class": "useful_physical_coordinate_entangled_with_endpoint_law_and_adversarial_costate",
            "unsb_object": "restricted EROT entropy coordinate on [t,1] with the endpoint transition law held fixed",
            "mathematical_update": (
                "Keep Y=G_theta(X_t,z) exactly unchanged.  For h=1-t>0 present the entropy "
                "critic with R_h=(Y-X_t)/sqrt(h), use the physical entropy coefficient h from "
                "the restricted UNSB objective, and leave GAN, PatchNCE, transport cost and "
                "rollout/inference endpoint transitions unchanged."
            ),
            "derivation": [
                "UNSB Theorem 1 restricts the bridge to [t,1], whose Brownian transition variance scales with the remaining horizon h.",
                "sqrt(h) is therefore the physical fluctuation scale, while R_h is an invertible coordinate of Y given X_t and h.",
                "Mutual information is invariant under this conditional bijection, so an ideal entropy critic may use R_h without changing the endpoint conditional law.",
                "Unlike HNEK-all, the generator forward is never multiplied by h^gamma; training, rollout and inference share the exact same endpoint map.",
                "The physical coefficient h replaces the code's uniform-index coefficient, correcting the identified nonuniform-time mismatch without a schedule.",
            ],
            "long_horizon_property": (
                "the endpoint law is exactly identical to plain at every state, while the continuously active entropy coordinate is an invertible physical reparameterization"
            ),
            "identity_condition": "at h=1, R_h=Y-X_t; disabling the coordinate recovers byte-identical plain forward",
            "self_null_condition": "the method has no endpoint correction to withdraw; generator inference is always plain",
            "unbiased_or_invariance_property": "conditional endpoint law and inference map are exactly invariant",
            "paired_target_access": False,
            "paired_target_proof": "h, X_t and generated Y are native unpaired bridge variables",
            "uses_fixed_window": False,
            "uses_whole_state_branch_selection": False,
            "changes_training_target": True,
            "changes_gradient_estimator": False,
            "changes_endpoint_law": False,
            "expected_valid_states": "all bridge times h>0; no phase-specific activation",
            "falsification_test": "kill if forward/rollout endpoint identity fails or if a 400/800-step micro run reproduces HNEK-all reversal without improving coordinate conditioning",
            "compute_cost": "same generator cost as plain; residual-coordinate critic arithmetic only",
            "memory_cost": "same order as plain UNSB",
            "allowed_claim": "tests whether HNEK's physical-coordinate signal can be retained without changing the endpoint law",
            "forbidden_claims": [
                "exact finite-capacity critic invariance",
                "proof of PSNR improvement",
                "time conditioning repair",
                "route1 is exhausted if this coordinate fails",
            ],
        },
    ]
    for card in cards:
        validate_derivation_card(card)
    return cards


def write_cards(matrix_path: Path, output_dir: Path) -> dict:
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    cards = derivation_cards(matrix)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for card in cards:
        path = output_dir / f"{card['candidate_id']}.json"
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = [
            f"# {card['candidate_id']}: {card['name']}", "",
            "## Evidence route", "",
            *[f"- `{item}`" for item in card["parent_evidence"]], "",
            "## Mathematical update", "", card["mathematical_update"], "",
            "## Derivation", "",
            *[f"{index}. {item}" for index, item in enumerate(card["derivation"], 1)], "",
            "## Long-horizon property", "", card["long_horizon_property"] + ".", "",
            "## Falsification", "", card["falsification_test"] + ".", "",
        ]
        (output_dir / f"{card['candidate_id']}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    return {
        "schema": "clean-unsb-search005-generation1-derivation-v1",
        "candidate_count": len(cards),
        "candidate_ids": [card["candidate_id"] for card in cards],
        "shared_controller_generated": False,
        "fixed_window_generated": False,
        "paired_target_access": False,
        "confirmation20_opened": False,
    }
