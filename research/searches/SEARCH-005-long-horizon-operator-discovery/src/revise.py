"""Evidence-conditional Generation-2 revision cards."""

from __future__ import annotations

import json
from pathlib import Path

from .claim_guard import validate_derivation_card


def fbcmp_card(acmp_result_path: Path) -> dict:
    result = json.loads(Path(acmp_result_path).read_text(encoding="utf-8"))
    trajectory = {int(row["step"]): row for row in result["trajectory"]}
    if not (
        float(trajectory[400]["macro_psnr_delta"]) > 0
        and float(trajectory[800]["macro_psnr_delta"]) < 0
    ):
        raise RuntimeError("FBCMP revision requires the observed ACMP reversal")
    card = {
        "candidate_id": "G2-HJ-FBCMP",
        "name": "Future-Batch Consensus Metric Projection",
        "candidate_type": "operator",
        "generation": 2,
        "parent_probes": ["hj", "G1-HJ-ACMP"],
        "parent_evidence": [
            "G1-HJ-ACMP::step400=+0.560489::5/6::guardrail-pass",
            "G1-HJ-ACMP::step800=-0.640026::1/6",
            "G1-HJ-ACMP::all sampled native/Adv+SB metric alignments nonnegative",
            "G1-HJ-ACMP::antithetic energy ratio range 0.664-0.992",
        ],
        "causal_failure_class": "same_batch_first_order_safety_does_not_transfer_across_stochastic_unsb_updates",
        "unsb_object": "a temporally transportable HJ correction field under native Adam geometry",
        "mathematical_update": (
            "Let c_k be the antithetic raw HJ correction on independent unpaired batch k "
            "and M_k the frozen Adam diagonal metric.  With the previous-batch field "
            "c_{k-1}, compute q_k=<c_{k-1},c_k>_M/(||c_{k-1}||_M||c_k||_M+eps), "
            "rho_k=[q_k]_+, and m_k=rho_k(c_{k-1}+c_k)/2.  Project m_k onto "
            "<g_UNSB,k,c>_M>=0 and <g_Adv+SB,k,c>_M>=0, apply it to G, then store "
            "c_k for the next independent batch.  The first update uses m_0=0."
        ),
        "derivation": [
            "G1 proves that same-batch half-space feasibility alone can coexist with a 400-to-800 PSNR reversal.",
            "Consecutive shuffled unpaired batches are independent samples of the native training distribution, so positive metric cosine is a target-blind estimate of correction-field transportability.",
            "The positive-part cosine is not an exit threshold: it acts per update and per correction field, continuously ranging from zero to one.",
            "When two batches disagree, rho_k=0 and the HJ component self-nullifies exactly; when they agree, their mean reduces stochastic variance before the original ACMP safety projection.",
            "No prior checkpoint, paired score, training age or fixed duration enters the operator.",
        ],
        "long_horizon_property": (
            "an HJ direction can accumulate only when it is both cross-batch transportable and "
            "locally feasible for the current native and Adv+SB objectives"
        ),
        "identity_condition": "no previous field or nonpositive cross-batch cosine gives the exact plain update",
        "self_null_condition": "stochastic or state-specific HJ fields with nonpositive next-batch agreement contribute exactly zero",
        "unbiased_or_invariance_property": "endpoint law and native objective are unchanged; consensus is a biased but target-blind safety operator on the auxiliary HJ field",
        "paired_target_access": False,
        "paired_target_proof": "both correction fields and both native normals come only from consecutive unpaired training batches",
        "uses_fixed_window": False,
        "uses_whole_state_branch_selection": False,
        "changes_training_target": False,
        "changes_gradient_estimator": True,
        "changes_endpoint_law": False,
        "expected_valid_states": "states where HJ structure corrections persist across independent unpaired updates",
        "falsification_test": "kill the HJ correction-field mechanism if the fixed 400/800 micro run is not positive at 800 or if consensus is almost always zero while no gain appears",
        "compute_cost": "G1-ACMP cost plus one previous correction field and metric consensus reductions",
        "memory_cost": "one CPU-resident generator correction field in addition to G1-ACMP",
        "allowed_claim": "tests whether cross-batch transportability repairs G1-HJ's locally-safe but globally-reversing correction field",
        "forbidden_claims": [
            "future PSNR prediction",
            "learned exit timing",
            "unbiasedness with respect to plain UNSB",
            "route1 is exhausted if this revision fails",
        ],
    }
    validate_derivation_card(card)
    return card


def markdown(card: dict) -> str:
    return "\n".join([
        f"# {card['candidate_id']}: {card['name']}", "",
        "## Evidence route", "",
        *[f"- `{item}`" for item in card["parent_evidence"]], "",
        "## Mathematical update", "", card["mathematical_update"], "",
        "## Derivation", "",
        *[f"{index}. {item}" for index, item in enumerate(card["derivation"], 1)], "",
        "## Long-horizon property", "", card["long_horizon_property"] + ".", "",
        "## Falsification", "", card["falsification_test"] + ".", "",
    ])
