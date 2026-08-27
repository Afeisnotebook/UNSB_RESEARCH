"""Synthesize propagation audits into an evidence-routing matrix."""

from __future__ import annotations

import json
from pathlib import Path


def _delta(row: dict) -> dict:
    return row.get("native_view_delta", row.get("pulse_native_view_delta"))


def retention_class(initial: float, final: float) -> str:
    if initial == 0.0 and final == 0.0:
        return "operator_exactly_null"
    if initial > 0.0 and final < 0.0:
        return "positive_impulse_reversed_by_native_flow"
    if initial < 0.0 and final > 0.0:
        return "negative_impulse_rotated_by_native_flow"
    if initial > 0.0 and final >= 0.0 and final < 0.25 * initial:
        return "positive_impulse_strongly_attenuated"
    if initial > 0.0 and final >= 0.25 * initial:
        return "positive_impulse_retained"
    if initial < 0.0 and final <= 0.0:
        return "negative_impulse_remains_harmful"
    return "boundary"


def build_causal_matrix(
    propagation_dir: Path,
    evidence_seed_path: Path,
    variance_dir: Path | None = None,
    attribution_dir: Path | None = None,
) -> dict:
    seed = json.loads(Path(evidence_seed_path).read_text(encoding="utf-8"))
    rows = []
    attribution_rows = []
    for path in sorted(Path(propagation_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = payload["checkpoint"]
        first = payload["trajectory"][0]
        last = payload["trajectory"][-1]
        immediate = _delta(first)
        final = _delta(last)
        geometry = last["network_gap"]["global"]
        row = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "probe": checkpoint["probe"],
            "stage": checkpoint["stage"],
            "step": int(checkpoint["step"]),
            "historical_accumulated_delta": float(checkpoint["historical_delta"]),
            "pulse_steps": int(payload["pulse_steps"]),
            "immediate_native_view_delta": float(immediate["macro_psnr_delta"]),
            "final_horizon": int(last["horizon"]),
            "final_native_view_delta": float(final["macro_psnr_delta"]),
            "final_positive_domains": int(final["positive_domains"]),
            "final_worst_domain_delta": float(final["worst_domain_delta"]),
            "final_guardrails_pass": bool(final["guardrails_pass"]),
            "gap_retention_ratio": float(geometry["retention_ratio"]),
            "initial_direction_cosine": float(geometry["initial_direction_cosine"]),
            "direction_memory_lost": bool(
                abs(float(geometry["initial_direction_cosine"])) < 0.2
            ),
            "gap_amplified": bool(float(geometry["retention_ratio"]) > 2.0),
            "retention_class": retention_class(
                float(immediate["macro_psnr_delta"]),
                float(final["macro_psnr_delta"]),
            ),
            "paired_metrics_available_to_algorithm": False,
        }
        rows.append(row)
        for arm in payload.get("component_attribution", []):
            attribution_rows.append({
                "checkpoint_id": checkpoint["checkpoint_id"],
                "probe": checkpoint["probe"],
                "arm": arm["arm"],
                "native_steps": int(arm["native_steps"]),
                "macro_psnr_delta": float(arm["delta"]["macro_psnr_delta"]),
                "positive_domains": int(arm["delta"]["positive_domains"]),
                "worst_domain_delta": float(arm["delta"]["worst_domain_delta"]),
            })

    if attribution_dir is not None:
        for path in sorted(Path(attribution_dir).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = payload["checkpoint"]
            for arm in payload["arms"]:
                attribution_rows.append({
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "probe": checkpoint["probe"],
                    "arm": arm["arm"],
                    "native_steps": int(payload["native_steps"]),
                    "macro_psnr_delta": float(arm["delta"]["macro_psnr_delta"]),
                    "positive_domains": int(arm["delta"]["positive_domains"]),
                    "worst_domain_delta": float(arm["delta"]["worst_domain_delta"]),
                })

    variance_rows = []
    if variance_dir is not None:
        for path in sorted(Path(variance_dir).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = payload["checkpoint"]
            global_row = payload["global"]
            variance_rows.append({
                "checkpoint_id": checkpoint["checkpoint_id"],
                "probe": checkpoint["probe"],
                "step": int(checkpoint["step"]),
                "replicates": int(payload["replicates"]),
                "variance_fraction": float(global_row["variance_fraction"]),
                "variance_dominated": bool(global_row["variance_dominated"]),
                "mean_same_batch_native_cosine": float(
                    global_row["mean_same_batch_native_cosine"]
                ),
                "mean_next_batch_native_cosine": float(
                    global_row["mean_next_batch_native_cosine"]
                ),
                "exact_zero_samples": sum(
                    float(sample["correction_norm"]) == 0.0
                    for sample in payload["samples"]
                ),
                "network_variance_fraction": {
                    name: float(value["variance_fraction"])
                    for name, value in payload["networks"].items()
                },
            })

    nonzero = [row for row in rows if row["retention_class"] != "operator_exactly_null"]
    memory_loss = [row for row in nonzero if row["direction_memory_lost"] and row["gap_amplified"]]
    reversals = [row for row in nonzero if row["retention_class"] == (
        "positive_impulse_reversed_by_native_flow"
    )]
    attenuated = [row for row in nonzero if row["retention_class"] == (
        "positive_impulse_strongly_attenuated"
    )]
    dt_null = [row for row in rows if row["probe"] == "dt" and row["retention_class"] == "operator_exactly_null"]
    attribution_probes = {row["probe"] for row in attribution_rows}
    variance_dominated = [row for row in variance_rows if row["variance_dominated"]]
    failure_classes = [
        {
            "id": "FC-TRANSPORT-ROTATION-AMPLIFICATION",
            "status": "supported_cross_probe",
            "evidence": [row["checkpoint_id"] for row in memory_loss],
            "statement": "native UNSB dynamics amplify the parameter-state gap while losing the initial beneficial correction direction",
            "candidate_routing": "coupled_dynamics_or_stability_constrained_operator",
        },
        {
            "id": "FC-POSITIVE-IMPULSE-NOT-INVARIANT",
            "status": "supported" if reversals or attenuated else "not_supported",
            "evidence": [row["checkpoint_id"] for row in reversals + attenuated],
            "statement": "an immediately beneficial correction is not an invariant or contractive direction of the later native flow",
            "candidate_routing": "invariant_preserving_operator_not_exit_policy",
        },
        {
            "id": "FC-COUPLED-COSTATE",
            "status": (
                "heterogeneous_probe_specific" if {"hj", "hnek"} <= attribution_probes
                else "supported_single_probe" if attribution_rows else "pending"
            ),
            "evidence": [row["checkpoint_id"] + "::" + row["arm"] for row in attribution_rows],
            "statement": "co-state propagation is mechanism-specific: HJ transfers become harmful, while HNEK G/F parameters plus moments preserve a strong short-horizon benefit",
            "candidate_routing": "probe_specific_coupled_dynamics_not_unified_handoff",
        },
        {
            "id": "FC-DT-INTERMITTENT-NULL",
            "status": "supported" if dt_null else "not_supported",
            "evidence": [row["checkpoint_id"] for row in dt_null],
            "statement": "the current DT statistic makes the correction exactly zero in some late states while remaining active in nearby states",
            "candidate_routing": "audit_moving_statistic_before_derivation",
        },
        {
            "id": "FC-STOCHASTIC-VARIANCE",
            "status": "supported_method_specific" if variance_dominated else "not_supported",
            "evidence": [row["checkpoint_id"] for row in variance_dominated],
            "statement": "HJ step1200 and DT step4000 are variance dominated across independent unpaired batches; HNEK and DT step3000 remain elevated but below the frozen threshold",
            "candidate_routing": "unbiased_or_mean_preserving_variance_reduction_only_for_supported_probes",
        },
    ]
    required_variance = {"DT-FULL-3000-NEG", "DT-FULL-4000-NEG", "HJ-SMALL-1200-POS", "HNEK-FULL-3000-POS"}
    completed_variance = {row["checkpoint_id"] for row in variance_rows}
    candidate_generation_allowed = bool(
        required_variance <= completed_variance and {"hj", "hnek"} <= attribution_probes
    )
    return {
        "schema": "clean-unsb-search005-causal-matrix-v1",
        "propagation_rows": rows,
        "component_attribution": attribution_rows,
        "bias_variance": variance_rows,
        "failure_classes": failure_classes,
        "search003_retrospective_contradictions": seed["contradictions"],
        "candidate_generation_allowed": candidate_generation_allowed,
        "candidate_generation_blockers": [] if candidate_generation_allowed else [
            "representative bias-variance or cross-probe component attribution is incomplete"
        ],
        "generation_rule": "derive at most one minimal operator for each probe-specific supported failure mechanism; do not force a shared controller",
        "forbidden_inference": "the eight-step pulse is a finite-difference diagnostic and not a fixed-window candidate",
        "paired_metrics_available_to_algorithm": False,
        "confirmation20_opened": False,
    }


def markdown(matrix: dict) -> str:
    lines = [
        "# SEARCH-005 causal matrix", "",
        "The eight-step pulse is a finite-difference diagnostic, not an activation window.", "",
        "| checkpoint | historical | immediate pulse | final native | domains | worst | gap ratio | direction cosine | class |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in matrix["propagation_rows"]:
        lines.append(
            f"| {row['checkpoint_id']} | {row['historical_accumulated_delta']:+.3f} | "
            f"{row['immediate_native_view_delta']:+.3f} | "
            f"{row['final_native_view_delta']:+.3f} | "
            f"{row['final_positive_domains']}/6 | {row['final_worst_domain_delta']:+.3f} | "
            f"{row['gap_retention_ratio']:.2f} | {row['initial_direction_cosine']:.3f} | "
            f"{row['retention_class']} |"
        )
    lines.extend(["", "## Evidence-routed failure classes", ""])
    for failure in matrix["failure_classes"]:
        lines.extend([
            f"### {failure['id']} — {failure['status']}", "",
            failure["statement"] + ".", "",
            f"Routing: `{failure['candidate_routing']}`.", "",
        ])
    lines.extend([
        "## Current gate", "",
        "Candidate generation remains blocked until the listed causal audits are complete. "
        "No controller, schedule, handoff or fixed-window method is admitted by this matrix.", "",
    ])
    return "\n".join(lines)
