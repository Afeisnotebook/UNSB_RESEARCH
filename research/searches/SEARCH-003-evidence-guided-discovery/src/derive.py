"""Generate Generation-1 algorithms only from adjudicated SEARCH-003 evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .ledger import HypothesisLedger


PROBE_CONSTRUCTIONS = {
    "dt": {
        "id": "G1-DT-RHGC8",
        "name": "Receding-Horizon GAN Consensus for DT",
        "preferred_observable": "loss_delta::G_GAN",
        "unsb_object": "full optimizer-state transition under the native GAN component",
        "parent_failure": "DT locally helps at 3k/H200 but decays by H1000 and becomes harmful at 4k",
    },
    "hj": {
        "id": "G1-HJ-RHDFC8",
        "name": "Receding-Horizon Native-Discriminator Consensus for HJ",
        "preferred_observable": "loss_delta::D_fake",
        "unsb_object": "full optimizer-state transition under the native fake-discriminator component",
        "parent_failure": "HJ's own gate/risk cannot distinguish beneficial and harmful states",
    },
}


def _signal(
    analysis: dict, probe: str, preferred: str, *, horizon: int = 8
) -> dict | None:
    eligible = (
        analysis.get("eligible_method_signals_by_horizon", {})
        .get(str(horizon), {})
        .get(probe, [])
    )
    return next((item for item in eligible if item["feature"] == preferred), None)


def _hnek_time_variance(atlas_rows: list[dict]) -> dict:
    fractions = []
    details = []
    for row in atlas_rows:
        if not (
            row.get("probe") == "hnek"
            and row.get("stage") == "full"
            and int(row.get("horizon", 1)) == 200
            and row.get("source_state") == "hnek"
        ):
            continue
        diagnostics = row.get("proposal", {}).get("diagnostics", {})
        means = []
        variances = []
        for time_index in range(5):
            mean_key = f"time_moment::{time_index}::generator_grad_norm::mean"
            variance_key = f"time_moment::{time_index}::generator_grad_norm::variance"
            if mean_key in diagnostics and variance_key in diagnostics:
                means.append(float(diagnostics[mean_key]))
                variances.append(float(diagnostics[variance_key]))
        if len(means) != 5:
            continue
        between = statistics.pvariance(means)
        within = statistics.mean(variances)
        fraction = between / max(between + within, 1e-20)
        fractions.append(fraction)
        details.append({
            "step": int(row["step"]),
            "between_time_variance": between,
            "mean_within_time_variance": within,
            "between_fraction": fraction,
        })
    return {
        "cells": details,
        "median_between_fraction": (
            statistics.median(fractions) if fractions else None
        ),
        "time_variance_dominant": bool(
            fractions and statistics.median(fractions) > 0.5
        ),
    }


def _card(probe: str, signal: dict, *, horizon: int = 8) -> dict:
    spec = PROBE_CONSTRUCTIONS[probe]
    observable = signal["feature"].split("::", 1)[1]
    comparator = ">0" if int(signal["direction"]) > 0 else "<0"
    formula = (
        f"From the same immutable full state S_k, compute B0=Phi_0^{horizon}(S_k) "
        f"and Bi=Phi_{probe}^{horizon}(S_k). Let Delta={observable}(Bi)-"
        f"{observable}(B0). Commit Bi iff Delta{comparator}; otherwise commit B0."
    )
    return {
        "id": spec["id"],
        "generation": 1,
        "name": spec["name"],
        "parents": [probe],
        "parent_probe": probe,
        "observed_failure": spec["parent_failure"],
        "evidence": {
            "feature": signal["feature"],
            "direction": signal["direction"],
            "balanced_accuracy": signal["balanced_accuracy"],
            "spearman": signal["spearman"],
            "supporting_domains": signal["supporting_domains"],
            "precursor_horizon": horizon,
            "outcome_horizon": 200,
        },
        "unsb_object": spec["unsb_object"],
        "operator": "target-blind receding-horizon full-state branch selection",
        "derivation": formula,
        "identity_condition": (
            f"Delta_{observable} {'<= 0' if int(signal['direction']) > 0 else '>= 0'} "
            "commits the exact plain branch"
        ),
        "self_null_condition": "a rejected proposal contributes zero parameters, moments, schedulers, streams or RNG state",
        "changes_training_target": False,
        "changes_gradient_estimator": False,
        "changes_endpoint_law": probe == "dt" and False,
        "paired_target_access": False,
        "paired_target_proof": (
            f"the decision reads only branch-averaged unpaired {observable}; "
            "PSNR/SSIM/LPIPS and confirmation identities are absent from the controller interface"
        ),
        "expected_valid_states": (
            f"states whose {horizon}-step observable retains its audited lead relationship"
        ),
        "falsification_test": (
            "proposal-only, observable-only/plain, and full selector from the same e0; "
            "kill if the selector chooses harmful H200 branches or fails the 2400-step late gate"
        ),
        "compute_cost": (
            f"two {horizon}-update branches per committed {horizon} updates; "
            "proposal branch inherits parent-probe cost"
        ),
        "memory_cost": "one live branch at a time plus one CPU full-state snapshot",
        "pseudocode": [
            "snapshot = full_state()",
            f"plain = run(snapshot, operator=UNSB, H={horizon})",
            f"proposal = run(snapshot, operator={probe}, H={horizon})",
            f"commit proposal if direction * (mean({observable})_proposal - "
            f"mean({observable})_plain) > 0 else plain",
            "repeat from committed full state",
        ],
        "status": "proposed",
    }


def derive_generation1(
    analysis_path: Path,
    adjudication_path: Path,
    atlas_path: Path,
    output_dir: Path,
) -> dict:
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    adjudication = json.loads(Path(adjudication_path).read_text(encoding="utf-8"))
    with Path(atlas_path).open("r", encoding="utf-8") as handle:
        atlas_rows = [json.loads(line) for line in handle if line.strip()]
    if not adjudication.get("complete"):
        raise RuntimeError("Generation 0 decisive grid is incomplete")

    output_dir = Path(output_dir)
    cards_dir = output_dir / "DERIVATION_CARDS"
    cards_dir.mkdir(parents=True, exist_ok=True)
    ledger = HypothesisLedger(output_dir / "HYPOTHESIS_LEDGER.json")
    superseded = {"G1-DT-RHGC", "G1-HJ-RHNC"}
    for entry in list(ledger.entries):
        if entry["id"] in superseded and entry["status"] == "proposed":
            ledger.transition(
                entry["id"],
                status="superseded_protocol_alignment",
                evidence={
                    "reason": (
                        "the 32-update precursor does not divide the frozen 400-update "
                        "evaluation interval; a separately audited 8-update signal is used"
                    ),
                    "paired_target_access": False,
                },
            )
    cards = []
    for probe, spec in PROBE_CONSTRUCTIONS.items():
        signal = _signal(analysis, probe, spec["preferred_observable"], horizon=8)
        if signal is None:
            continue
        card = _card(probe, signal, horizon=8)
        cards.append(card)
        json_path = cards_dir / f"{card['id']}.json"
        json_path.write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown = [
            f"# {card['id']}: {card['name']}", "",
            f"Parent probe: `{probe}`.", "",
            "## Evidence", "",
            f"`{signal['feature']}` at H=8 predicts H=200: BA={signal['balanced_accuracy']:.3f}, "
            f"Spearman={signal['spearman']:.3f}, domains={len(signal['supporting_domains'])}/6.", "",
            "## Operator", "", card["derivation"], "",
            "## Identity / self-null", "",
            card["identity_condition"] + ".", card["self_null_condition"] + ".", "",
            "## Falsification", "", card["falsification_test"] + ".", "",
        ]
        (cards_dir / f"{card['id']}.md").write_text(
            "\n".join(markdown), encoding="utf-8"
        )
        if not any(entry["id"] == card["id"] for entry in ledger.entries):
            ledger.append(card)

    hnek_variance = _hnek_time_variance(atlas_rows)
    hnek_signal = analysis.get("eligible_method_signals", {}).get("hnek", [])
    route_closures = []
    if not hnek_signal and not hnek_variance["time_variance_dominant"]:
        route_closures.append({
            "probe": "hnek",
            "status": "no_generation1_candidate",
            "reason": (
                "no legal lead signal and between-time generator-gradient variance "
                "does not dominate within-time variance; an adaptive controller or "
                "time-importance sampler is not evidence-routed"
            ),
            "variance_evidence": hnek_variance,
        })

    report = {
        "schema": "clean-unsb-search003-generation1-v1",
        "candidate_count": len(cards),
        "cards": cards,
        "route_closures": route_closures,
        "shared_controller_used": False,
        "paired_target_access": False,
        "confirmation20_opened": False,
    }
    (output_dir / "GENERATION1_DERIVATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
