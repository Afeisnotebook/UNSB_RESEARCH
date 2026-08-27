"""Turn Generation-0 observations into falsifiable routing evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .catalog import MatchedCheckpoint
from .protocol import Search003Protocol
from .search001_compat import modules


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def historical_label(cell: MatchedCheckpoint) -> dict | None:
    plain_path = cell.plain.parent / f"metrics_step_{cell.step}.json"
    method_path = cell.method.parent / f"metrics_step_{cell.step}.json"
    if not plain_path.is_file() or not method_path.is_file():
        return None
    plain = json.loads(plain_path.read_text(encoding="utf-8"))
    method = json.loads(method_path.read_text(encoding="utf-8"))
    return modules()[2].compare(method, plain, step=cell.step)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(abs(float(denominator)), 1e-20)


def feature_row(row: dict) -> dict[str, float]:
    geometry = row["update_geometry"]
    reference_losses = row["reference"].get("losses", {})
    proposal_losses = row["proposal"].get("losses", {})
    features = {
        "reference_proposal_cosine": float(geometry["reference_proposal_cosine"]),
        "correction_reference_cosine": float(geometry["correction_reference_cosine"]),
        "correction_to_reference_norm": _safe_ratio(
            geometry["correction_norm"], geometry["reference_norm"]
        ),
        "proposal_to_reference_norm": _safe_ratio(
            geometry["proposal_norm"], geometry["reference_norm"]
        ),
    }
    for key in sorted(set(reference_losses) & set(proposal_losses)):
        features[f"loss_delta::{key}"] = float(proposal_losses[key]) - float(
            reference_losses[key]
        )
    invalid_hj = (
        row.get("diagnostic_validity", {}).get("hj_accumulators")
        == "invalid_cumulative_legacy"
    )
    for key, value in row["proposal"].get("diagnostics", {}).items():
        if not key.startswith(("domain_count::", "time_count::")):
            if invalid_hj and key.startswith("hj_") and key.endswith("_sum"):
                continue
            features[f"diagnostic::{key}"] = float(value)
    return features


def _balanced_accuracy(labels: list[bool], predictions: list[bool]) -> float:
    positives = [index for index, value in enumerate(labels) if value]
    negatives = [index for index, value in enumerate(labels) if not value]
    if not positives or not negatives:
        return float("nan")
    tpr = sum(predictions[index] for index in positives) / len(positives)
    tnr = sum(not predictions[index] for index in negatives) / len(negatives)
    return 0.5 * (tpr + tnr)


def evaluate_sign_signal(rows: list[dict], feature: str, direction: int) -> dict | None:
    eligible = []
    for row in rows:
        label = row.get("post_branch_development_label")
        features = row.get("features", {})
        if label is None or feature not in features:
            continue
        delta = float(label["macro_psnr_delta"])
        eligible.append((row["probe"], features[feature], delta, label))
    if len(eligible) < 6:
        return None
    labels = [delta > 0.0 for _, _, delta, _ in eligible]
    predictions = [(direction * value) > 0.0 for _, value, _, _ in eligible]
    ba_raw = _balanced_accuracy(labels, predictions)
    signal_values = [direction * value for _, value, _, _ in eligible]
    outcome_values = [delta for _, _, delta, _ in eligible]
    rho_raw = (
        float("nan")
        if len(set(signal_values)) < 2 or len(set(outcome_values)) < 2
        else float(spearmanr(signal_values, outcome_values).statistic)
    )
    ba = ba_raw if math.isfinite(ba_raw) else None
    rho = rho_raw if math.isfinite(rho_raw) else None
    probes = sorted({probe for probe, _, _, _ in eligible})
    folds = {}
    for heldout in probes:
        indices = [index for index, item in enumerate(eligible) if item[0] == heldout]
        fold_labels = [labels[index] for index in indices]
        fold_predictions = [predictions[index] for index in indices]
        fold = _balanced_accuracy(fold_labels, fold_predictions)
        folds[heldout] = fold if math.isfinite(fold) else None
    finite_folds = [value for value in folds.values() if value is not None]

    # A macro label can hide a domain-specific sign failure.  Treat each
    # domain as a separate target-blind-audit label and require the signal to
    # be decidable there as well; a domain with only one observed class gives
    # no evidence and therefore cannot count as support.
    domain_folds: dict[str, float | None] = {}
    domains = sorted({
        domain
        for _, _, _, label in eligible
        for domain in label.get("domain_delta", {})
    })
    for domain in domains:
        domain_indices = [
            index for index, (_, _, _, label) in enumerate(eligible)
            if domain in label.get("domain_delta", {})
        ]
        domain_labels = [
            float(eligible[index][3]["domain_delta"][domain]["psnr"]) > 0.0
            for index in domain_indices
        ]
        domain_predictions = [predictions[index] for index in domain_indices]
        score = _balanced_accuracy(domain_labels, domain_predictions)
        domain_folds[domain] = score if math.isfinite(score) else None
    supporting_domains = [
        domain for domain, score in domain_folds.items()
        if score is not None and score >= 0.65
    ]
    return {
        "feature": feature,
        "direction": int(direction),
        "n": len(eligible),
        "balanced_accuracy": ba,
        "spearman": rho,
        "leave_one_method_out": folds,
        "finite_fold_mean": (
            float(np.mean(finite_folds)) if finite_folds else None
        ),
        "finite_fold_min": min(finite_folds) if finite_folds else None,
        "finite_fold_count": len(finite_folds),
        "required_method_fold_count": len(probes),
        "domain_balanced_accuracy": domain_folds,
        "supporting_domains": supporting_domains,
        "supporting_domain_count": len(supporting_domains),
    }


def analyze_atlas(
    atlas_path: Path,
    catalog: list[MatchedCheckpoint],
    protocol: Search003Protocol,
) -> dict:
    raw_rows = read_jsonl(atlas_path)
    excluded_rows = [
        row for row in raw_rows
        if row.get("diagnostic_validity", {}).get("causal_operator_state")
        == "invalid_teacher_transplant"
    ]
    rows = [row for row in raw_rows if row not in excluded_rows]
    label_by_key = {}
    for cell in catalog:
        label_by_key[(cell.probe, cell.stage, cell.step)] = historical_label(cell)
    enriched = []
    for row in rows:
        value = dict(row)
        value["features"] = feature_row(row)
        value["historical_development_label"] = label_by_key.get(
            (row["probe"], row["stage"], int(row["step"]))
        )
        enriched.append(value)

    # A precursor must be measured before the outcome horizon.  Never
    # correlate losses accumulated during a 200-step branch with that same
    # branch's paired-after-completion label.  For every causal state choose
    # the longest available short audit (1/8/32) and attach only the later
    # H=200 outcome.
    future_200 = {}
    for row in enriched:
        if int(row.get("horizon", 1)) != 200:
            continue
        label = row.get("post_branch_development_label")
        if label is None:
            continue
        key = (
            row["probe"], row["stage"], int(row["step"]), row["source_state"]
        )
        future_200[key] = label
    precursor_by_key = {}
    for row in enriched:
        horizon = int(row.get("horizon", 1))
        if horizon >= 200:
            continue
        key = (
            row["probe"], row["stage"], int(row["step"]), row["source_state"]
        )
        if key not in future_200:
            continue
        current = precursor_by_key.get(key)
        if current is None or horizon > int(current.get("horizon", 1)):
            value = dict(row)
            value["precursor_horizon"] = horizon
            value["post_branch_development_label"] = future_200[key]
            precursor_by_key[key] = value
    precursor_rows = list(precursor_by_key.values())

    feature_names = sorted({
        name for row in precursor_rows for name in row.get("features", {})
        if not name.startswith("diagnostic::bridge_time_index")
    })
    signals = []
    for feature in feature_names:
        for direction in (1, -1):
            result = evaluate_sign_signal(precursor_rows, feature, direction)
            if result is not None:
                signals.append(result)
    signals.sort(
        key=lambda item: (
            float("-inf") if item["balanced_accuracy"] is None
            else float(item["balanced_accuracy"]),
            float("-inf") if item["spearman"] is None
            else float(item["spearman"]),
        ),
        reverse=True,
    )
    def passes(item: dict) -> bool:
        return (
            item["balanced_accuracy"] is not None
            and float(item["balanced_accuracy"])
            >= protocol.signal_balanced_accuracy_min
            and item["spearman"] is not None
            and float(item["spearman"]) >= protocol.signal_spearman_min
            # "Shared" means that every observed probe is independently
            # decidable.  Ignoring an undefined held-out fold caused an NCE
            # loss delta to be incorrectly promoted when HNEK supplied no
            # two-class evidence.
            and int(item["finite_fold_count"])
            == int(item["required_method_fold_count"])
            and item["finite_fold_min"] is not None
            and float(item["finite_fold_min"])
            >= protocol.signal_balanced_accuracy_min
            and int(item["supporting_domain_count"])
            >= protocol.signal_domain_agreement_min
        )

    eligible_signals = [item for item in signals if passes(item)]

    # A failed shared gate does not erase a real method-specific precursor.
    # Evaluate the same frozen thresholds independently, without training a
    # black-box predictor or borrowing labels from another probe.
    method_signal_ranking = {}
    eligible_method_signals = {}
    for probe in sorted({row["probe"] for row in precursor_rows}):
        probe_rows = [row for row in precursor_rows if row["probe"] == probe]
        probe_results = []
        for feature in feature_names:
            for direction in (1, -1):
                result = evaluate_sign_signal(probe_rows, feature, direction)
                if result is not None:
                    probe_results.append(result)
        probe_results.sort(
            key=lambda item: (
                float("-inf") if item["balanced_accuracy"] is None
                else float(item["balanced_accuracy"]),
                float("-inf") if item["spearman"] is None
                else float(item["spearman"]),
            ),
            reverse=True,
        )
        method_signal_ranking[probe] = probe_results
        eligible_method_signals[probe] = [
            item for item in probe_results if passes(item)
        ]

    eligible_method_signals_by_horizon = {}
    for precursor_horizon in (1, 8, 32):
        horizon_rows = []
        for row in enriched:
            if int(row.get("horizon", 1)) != precursor_horizon:
                continue
            key = (
                row["probe"], row["stage"], int(row["step"]), row["source_state"]
            )
            if key not in future_200:
                continue
            value = dict(row)
            value["post_branch_development_label"] = future_200[key]
            horizon_rows.append(value)
        by_probe_horizon = {}
        for probe in sorted({row["probe"] for row in horizon_rows}):
            probe_rows = [row for row in horizon_rows if row["probe"] == probe]
            probe_features = sorted({
                feature for row in probe_rows for feature in row.get("features", {})
                if not feature.startswith("diagnostic::bridge_time_index")
            })
            candidates = []
            for feature in probe_features:
                for direction in (1, -1):
                    result = evaluate_sign_signal(probe_rows, feature, direction)
                    if result is not None and passes(result):
                        candidates.append(result)
            candidates.sort(
                key=lambda item: (
                    float(item["balanced_accuracy"]), float(item["spearman"])
                ),
                reverse=True,
            )
            by_probe_horizon[probe] = candidates
        eligible_method_signals_by_horizon[str(precursor_horizon)] = by_probe_horizon

    by_probe = defaultdict(list)
    for row in enriched:
        by_probe[row["probe"]].append(row)
    probe_summaries = {}
    for probe, values in sorted(by_probe.items()):
        values.sort(key=lambda row: (row["stage"], int(row["step"]), int(row.get("horizon", 1))))
        historical = [
            row["historical_development_label"] for row in values
            if row["historical_development_label"] is not None
        ]
        unique_historical = {
            int(item["step"]): float(item["macro_psnr_delta"]) for item in historical
        }
        ordered_delta = [unique_historical[key] for key in sorted(unique_historical)]
        probe_summaries[probe] = {
            "atlas_rows": len(values),
            "historical_positive_observed": any(value > 0 for value in ordered_delta),
            "historical_negative_observed": any(value < 0 for value in ordered_delta),
            "reversal_observed": (
                any(value > 0 for value in ordered_delta)
                and any(value < 0 for value in ordered_delta)
            ),
            "historical_delta_by_step": unique_historical,
        }

    return {
        "schema": "clean-unsb-search003-reversal-analysis-v1",
        "raw_atlas_rows": len(raw_rows),
        "excluded_invalid_causal_rows": len(excluded_rows),
        "precursor_rows": len(precursor_rows),
        "precursor_contract": "longest_available_H_lt_200_predicts_H200",
        "rows": enriched,
        "probe_summaries": probe_summaries,
        "signal_ranking": signals,
        "eligible_shared_signals": eligible_signals,
        "shared_signal_gate_passed": bool(eligible_signals),
        "method_signal_ranking": method_signal_ranking,
        "eligible_method_signals": eligible_method_signals,
        "eligible_method_signals_by_horizon": eligible_method_signals_by_horizon,
        "signal_gate": {
            "balanced_accuracy_min": protocol.signal_balanced_accuracy_min,
            "spearman_min": protocol.signal_spearman_min,
            "domain_agreement_min": protocol.signal_domain_agreement_min,
        },
        "paired_labels_available_to_controller": False,
        "confirmation20_opened": False,
    }
