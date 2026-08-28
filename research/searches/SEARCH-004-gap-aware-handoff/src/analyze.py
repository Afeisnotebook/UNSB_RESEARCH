"""Matched continuation metrics, promotion gates and mechanism adjudication."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .catalog import HandoffCheckpoint
from .protocol import Search004Protocol


CANDIDATE_ARMS = (
    "B_gf_zero_moment",
    "C_local_native_moment",
    "D_costate_equilibration",
    "E_combined",
    "F_g_only_transplant",
    "G_gf_transplant",
    "H_native_moment_projection",
    "K_gf_state_transplant",
)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_at(result: dict, horizon: int) -> dict | None:
    for value in result.get("evaluations", []):
        if int(value["horizon"]) == int(horizon):
            return value
    return None


def source_metric_path(checkpoint: HandoffCheckpoint, *, plain: bool) -> Path:
    path = checkpoint.plain if plain else checkpoint.method
    return path.parent / f"metrics_step_{checkpoint.step}.json"


def source_metrics(checkpoint: HandoffCheckpoint) -> tuple[dict, dict]:
    method_path = source_metric_path(checkpoint, plain=False)
    plain_path = source_metric_path(checkpoint, plain=True)
    if not method_path.is_file() or not plain_path.is_file():
        raise FileNotFoundError((method_path, plain_path))
    return read_json(method_path), read_json(plain_path)


def trapezoid(points: list[tuple[int, float]], horizon: int) -> float:
    selected = sorted((x, y) for x, y in points if x <= horizon)
    if not selected or selected[0][0] != 0 or selected[-1][0] != horizon:
        return float("nan")
    area = 0.0
    for (left_x, left_y), (right_x, right_y) in zip(selected, selected[1:]):
        area += (right_x - left_x) * (left_y + right_y) / 2.0
    return area / float(horizon)


def domain_delta(method: dict, reference: dict) -> dict[str, float]:
    return {
        domain: float(method["domains"][domain]["psnr"])
        - float(reference["domains"][domain]["psnr"])
        for domain in method["domains"]
    }


def plain_relative_component_distance(value: dict, plain: dict, names: tuple[str, ...]) -> float:
    gaps = []
    for name in names:
        key = f"{name}_relative_step_norm"
        left = max(abs(float(value.get(key, 0.0))), 1e-30)
        right = max(abs(float(plain.get(key, 0.0))), 1e-30)
        gaps.append(abs(math.log(left / right)))
    return sum(gaps) / len(gaps)


def analyze_checkpoint(
    checkpoint: HandoffCheckpoint,
    results: dict[str, dict],
    protocol: Search004Protocol,
) -> dict:
    if "P_common_plain" not in results or "A_hard_disable" not in results:
        raise RuntimeError(f"{checkpoint.checkpoint_id}: P/A results are required")
    p = results["P_common_plain"]
    a = results["A_hard_disable"]
    source_method, source_plain = source_metrics(checkpoint)
    initial_advantage = float(source_method["macro_psnr"] - source_plain["macro_psnr"])
    p_components = p["final_component_diagnostics"]
    a_components = a["final_component_diagnostics"]
    rows = []
    for arm, result in sorted(results.items()):
        if int(result["horizon"]) != 200:
            continue
        m32, m200 = metric_at(result, 32), metric_at(result, 200)
        p32, p200 = metric_at(p, 32), metric_at(p, 200)
        a32, a200 = metric_at(a, 32), metric_at(a, 200)
        if any(value is None for value in (m32, m200, p32, p200, a32, a200)):
            raise RuntimeError(f"{checkpoint.checkpoint_id}/{arm}: incomplete evaluation")
        delta32 = float(m32["macro_psnr"] - p32["macro_psnr"])
        delta200 = float(m200["macro_psnr"] - p200["macro_psnr"])
        lift32 = float(m32["macro_psnr"] - a32["macro_psnr"])
        lift200 = float(m200["macro_psnr"] - a200["macro_psnr"])
        lifts = domain_delta(m200, a200)
        if arm in {"B_gf_zero_moment", "C_local_native_moment"}:
            base_defect = plain_relative_component_distance(a_components, p_components, ("G", "F"))
            arm_defect = plain_relative_component_distance(result["final_component_diagnostics"], p_components, ("G", "F"))
            defect_basis = "plain_relative_diagnostic_only"
        elif arm == "D_costate_equilibration" and "D0_hold_only" in results:
            d0 = results["D0_hold_only"]["final_component_diagnostics"]
            base_defect = plain_relative_component_distance(d0, p_components, ("D", "E", "F"))
            arm_defect = plain_relative_component_distance(result["final_component_diagnostics"], p_components, ("D", "E", "F"))
            defect_basis = "plain_relative_diagnostic_only"
        elif arm == "E_combined":
            base_defect = plain_relative_component_distance(a_components, p_components, ("G", "F", "D", "E"))
            arm_defect = plain_relative_component_distance(result["final_component_diagnostics"], p_components, ("G", "F", "D", "E"))
            defect_basis = "plain_relative_diagnostic_only"
        elif arm == "H_native_moment_projection":
            record = result.get("transport_record") or {}
            base_defect = float(record.get("target_blind_defect_before", 0.0))
            arm_defect = float(record.get("target_blind_defect_after", base_defect))
            defect_basis = "native_gradient_moment_constraint"
        elif arm in {
            "F_g_only_transplant", "G_gf_transplant", "K_gf_state_transplant",
        }:
            base_defect, arm_defect = 1.0, 0.0
            defect_basis = "component_ablation_mask"
        else:
            base_defect = arm_defect = 0.0
            defect_basis = "none"
        defect_reduction = (
            (base_defect - arm_defect) / base_defect if base_defect > 1e-12 else 0.0
        )
        auc_lift = trapezoid([(0, 0.0), (32, lift32), (200, lift200)], 200)
        auc_advantage = trapezoid(
            [(0, initial_advantage), (32, delta32), (200, delta200)], 200
        )
        d0_lift32 = d0_lift200 = d0_auc_lift = None
        d0_domain_lift = None
        if arm == "D_costate_equilibration" and "D0_hold_only" in results:
            d0_32 = metric_at(results["D0_hold_only"], 32)
            d0_200 = metric_at(results["D0_hold_only"], 200)
            d0_lift32 = float(m32["macro_psnr"] - d0_32["macro_psnr"])
            d0_lift200 = float(m200["macro_psnr"] - d0_200["macro_psnr"])
            d0_auc_lift = trapezoid(
                [(0, 0.0), (32, d0_lift32), (200, d0_lift200)], 200
            )
            d0_domain_lift = domain_delta(m200, d0_200)
        retention = (
            delta200 / initial_advantage if initial_advantage >= 0.10 else None
        )
        generic_conditions = {
            "lift": lift200 >= protocol.lift_min,
            "auc_lift": auc_lift >= protocol.auc_lift_min,
            "domain_coverage": sum(value >= 0.0 for value in lifts.values())
            >= protocol.domain_nonnegative_min,
            "worst_domain": min(lifts.values()) > protocol.worst_domain_lift_min,
            "defect_reduction": (
                defect_basis != "plain_relative_diagnostic_only"
                and defect_basis != "none"
                and defect_reduction >= protocol.defect_reduction_min
            ),
        }
        transferable_conditions = None
        if arm in {
            "F_g_only_transplant", "G_gf_transplant", "K_gf_state_transplant",
        } and initial_advantage >= 0.10:
            u200 = metric_at(results.get("U_uninterrupted", {}), 200)
            absolute_floor = min(
                float(source_method["macro_psnr"]),
                float(u200["macro_psnr"]) if u200 else float(source_method["macro_psnr"]),
            ) - 0.30
            transferable_conditions = {
                "retention": retention is not None and retention >= 0.50 and delta200 >= 0.10,
                "domain_coverage": sum(value > 0.0 for value in domain_delta(m200, p200).values()) >= 4,
                "absolute_floor": float(m200["macro_psnr"]) >= absolute_floor,
            }
        promoted = arm in CANDIDATE_ARMS and all(generic_conditions.values())
        if transferable_conditions is not None:
            promoted = promoted and all(transferable_conditions.values())
        rows.append({
            "checkpoint_id": checkpoint.checkpoint_id,
            "family": checkpoint.family,
            "source_positive": checkpoint.positive_source,
            "arm": arm,
            "initial_advantage": initial_advantage,
            "macro_psnr_32": float(m32["macro_psnr"]),
            "macro_psnr_200": float(m200["macro_psnr"]),
            "delta_plain_32": delta32,
            "delta_plain_200": delta200,
            "lift_hard_32": lift32,
            "lift_hard_200": lift200,
            "auc_handoff_lift": auc_lift,
            "auc_advantage": auc_advantage,
            "advantage_retention_200": retention,
            "absolute_change_200": float(m200["macro_psnr"] - source_method["macro_psnr"]),
            "positive_domains_vs_hard": sum(value >= 0.0 for value in lifts.values()),
            "worst_domain_vs_hard": min(lifts.values()),
            "domain_lift_vs_hard": lifts,
            "costate_lift_vs_hold_32": d0_lift32,
            "costate_lift_vs_hold_200": d0_lift200,
            "costate_auc_lift_vs_hold": d0_auc_lift,
            "costate_domain_lift_vs_hold": d0_domain_lift,
            "base_defect": base_defect,
            "arm_defect": arm_defect,
            "defect_reduction": defect_reduction,
            "defect_basis": defect_basis,
            "generic_conditions": generic_conditions,
            "transferable_conditions": transferable_conditions,
            "promote_800": promoted,
        })
    eligible = [row for row in rows if row["promote_800"]]
    eligible.sort(
        key=lambda row: (
            row["lift_hard_200"], row["auc_handoff_lift"],
            row["positive_domains_vs_hard"], -row["arm_defect"],
        ),
        reverse=True,
    )
    promoted = [row["arm"] for row in eligible[:2]]
    return {
        "schema": "clean-unsb-search004-checkpoint-analysis-v1",
        "checkpoint": checkpoint.to_dict(),
        "initial_advantage": initial_advantage,
        "source_method_macro_psnr": float(source_method["macro_psnr"]),
        "source_plain_macro_psnr": float(source_plain["macro_psnr"]),
        "rows": rows,
        "promoted_arms": promoted,
        "confirmation20_opened": False,
    }


def extension_pass(
    checkpoint_analysis: dict, result: dict, p: dict, a: dict, u: dict
) -> dict:
    arm = result["arm"]
    m800, p800, a800 = metric_at(result, 800), metric_at(p, 800), metric_at(a, 800)
    if any(value is None for value in (m800, p800, a800)):
        raise RuntimeError(f"{checkpoint_analysis['checkpoint']['checkpoint_id']}/{arm}: no 800 metric")
    delta = float(m800["macro_psnr"] - p800["macro_psnr"])
    lift = float(m800["macro_psnr"] - a800["macro_psnr"])
    domain_plain = domain_delta(m800, p800)
    conditions = {
        "positive_delta": delta >= 0.10,
        "hard_lift": lift >= 0.15,
        "coverage": sum(value > 0 for value in domain_plain.values()) >= 4,
        "worst_domain": min(domain_plain.values()) > -1.0,
    }
    initial_advantage = float(checkpoint_analysis["initial_advantage"])
    retention = delta / initial_advantage if initial_advantage >= 0.10 else None
    source_quality = float(checkpoint_analysis["source_method_macro_psnr"])
    absolute_floor = min(
        source_quality, float(metric_at(u, 800)["macro_psnr"])
    ) - 0.30
    transfer_conditions = None
    if arm in {
        "F_g_only_transplant", "G_gf_transplant", "K_gf_state_transplant",
    }:
        transfer_conditions = {
            "retention": retention is not None and retention >= 0.50 and delta >= 0.10,
            "coverage": sum(value > 0 for value in domain_plain.values()) >= 4,
            "absolute_floor": float(m800["macro_psnr"]) >= absolute_floor,
        }
    return {
        "checkpoint_id": checkpoint_analysis["checkpoint"]["checkpoint_id"],
        "arm": arm,
        "delta_plain_800": delta,
        "lift_hard_800": lift,
        "positive_domains_800": sum(value > 0 for value in domain_plain.values()),
        "worst_domain_800": min(domain_plain.values()),
        "advantage_retention_800": retention,
        "macro_psnr_800": float(m800["macro_psnr"]),
        "macro_ssim_800": float(m800["macro_ssim"]),
        "macro_lpips_800": m800["macro_lpips"],
        "conditions": conditions,
        "transfer_conditions": transfer_conditions,
        "replicated_800": all(conditions.values()) and (
            transfer_conditions is None or all(transfer_conditions.values())
        ),
    }


def adjudicate_mechanisms(
    analyses: list[dict], extensions: list[dict], compatibilities: list[dict] | None = None
) -> dict:
    rows = [row for analysis in analyses for row in analysis["rows"]]
    extension_lookup = {(row["checkpoint_id"], row["arm"]): row for row in extensions}

    def supported(arms: set[str]) -> tuple[bool, list[dict]]:
        evidence = [
            row for row in rows
            if row["source_positive"] and row["arm"] in arms and row["promote_800"]
        ]
        replicated = [
            row for row in evidence
            if extension_lookup.get((row["checkpoint_id"], row["arm"]), {}).get("replicated_800")
        ]
        arm_families = {(row["arm"], row["family"]) for row in evidence}
        same_arm_family_repeat = any(
            sum(
                row["arm"] == arm and row["family"] == family
                for row in evidence
            ) >= 2
            for arm, family in arm_families
        )
        return bool(replicated or same_arm_family_repeat), evidence

    optimizer_supported, optimizer_evidence = supported({"B_gf_zero_moment", "C_local_native_moment"})
    _, costate_evidence_raw = supported({"D_costate_equilibration"})
    costate_evidence = [
        row for row in costate_evidence_raw
        if row["costate_lift_vs_hold_200"] is not None
        and row["costate_lift_vs_hold_200"] >= 0.15
        and row["costate_auc_lift_vs_hold"] >= 0.10
        and sum(value >= 0.0 for value in row["costate_domain_lift_vs_hold"].values()) >= 4
        and min(row["costate_domain_lift_vs_hold"].values()) > -0.5
        and row["defect_reduction"] >= 0.25
    ]
    costate_replicated = any(
        extension_lookup.get((row["checkpoint_id"], row["arm"]), {}).get("replicated_800")
        for row in costate_evidence
    ) or any(
        sum(
            other["arm"] == row["arm"] and other["family"] == row["family"]
            for other in costate_evidence
        ) >= 2
        for row in costate_evidence
    )
    costate_supported = bool(costate_evidence and costate_replicated)
    transfer_supported, transfer_evidence = supported({
        "F_g_only_transplant", "G_gf_transplant", "K_gf_state_transplant",
    })
    positive_sources = {analysis["checkpoint"]["checkpoint_id"] for analysis in analyses if analysis["checkpoint"]["positive_source"]}
    positive_families = {analysis["checkpoint"]["family"] for analysis in analyses if analysis["checkpoint"]["positive_source"]}
    any_component_lift = any(
        row["source_positive"] and row["arm"] in CANDIDATE_ARMS
        and row["lift_hard_200"] >= 0.10
        for row in rows
    )
    u_rows = [row for row in rows if row["source_positive"] and row["arm"] == "U_uninterrupted"]
    u_sustained = any(row["lift_hard_200"] > 0.10 for row in u_rows)
    falsified = (
        len(positive_sources) >= 3 and len(positive_families) >= 2
        and not any_component_lift and not u_sustained
    )
    trajectory_only = (
        len(positive_sources) >= 3 and len(positive_families) >= 2
        and not any_component_lift and u_sustained
    )
    compatibility_evidence = [
        row for row in (compatibilities or [])
        if row.get("confidence_sequence", {}).get("valid")
        and row.get("persistently_incompatible")
    ]
    safe_component_repair = any(
        row["source_positive"] and row["arm"] in {
            "B_gf_zero_moment", "C_local_native_moment",
            "D_costate_equilibration", "E_combined",
        } and row["lift_hard_200"] >= 0.10
        for row in rows
    )
    return {
        "schema": "clean-unsb-search004-mechanism-adjudication-v1",
        "optimizer_state_mismatch_supported": optimizer_supported,
        "optimizer_evidence": optimizer_evidence,
        "DEF_costate_mismatch_supported": costate_supported,
        "costate_evidence": costate_evidence,
        "transferable_G_state_supported": transfer_supported,
        "transfer_evidence": transfer_evidence,
        "native_gradient_incompatibility_supported": False,
        "native_gradient_precursor_supported": bool(
            compatibility_evidence and not safe_component_repair
        ),
        "native_gradient_compatibility_evidence": compatibility_evidence,
        "native_gradient_status": (
            "requires_generation1_one_sided_or_proximal_counterfactual"
            if compatibility_evidence and not safe_component_repair
            else "confidence_sequence_does_not_support_persistent_incompatibility"
        ),
        "trajectory_dependent_nontransferable": trajectory_only,
        "handoff_primary_hypothesis_falsified": falsified,
        "confirmation20_opened": False,
    }
