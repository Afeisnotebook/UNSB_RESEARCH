"""Reinterpret SEARCH-003 evidence without turning it into an algorithm."""

from __future__ import annotations

import json
from pathlib import Path


def _branch_delta(row: dict) -> float | None:
    label = row.get("post_branch_development_label")
    return None if not label else float(label["macro_psnr_delta"])


def _history_delta(row: dict) -> float | None:
    label = row.get("historical_development_label")
    return None if not label else float(label["macro_psnr_delta"])


def contradiction_class(historical: float, branch: float) -> str:
    if historical > 0.0 and branch < 0.0:
        return "beneficial_state_with_current_operator_harm"
    if historical < 0.0 and branch > 0.0:
        return "harmful_accumulated_state_with_current_operator_benefit"
    if historical > 0.0 and branch > 0.0:
        return "state_and_current_operator_both_beneficial"
    if historical < 0.0 and branch < 0.0:
        return "state_and_current_operator_both_harmful"
    return "zero_boundary"


def seed_from_search003(analysis_path: Path, catalog: list) -> dict:
    payload = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    wanted = {(cell.probe, cell.stage.replace("100", "").replace("25", ""), cell.step)
              for cell in catalog}
    # SEARCH-003 calls the stages simply ``small`` and ``full``.
    wanted = {(probe, "small" if stage.startswith("small") else "full", step)
              for probe, stage, step in wanted}
    selected = []
    for row in payload["rows"]:
        identity = (row.get("probe"), row.get("stage"), int(row.get("step", -1)))
        if identity not in wanted or row.get("source_state") != row.get("probe"):
            continue
        horizon = int(row.get("horizon", 1))
        if horizon not in {1, 8, 32, 200, 1000}:
            continue
        historical = _history_delta(row)
        branch = _branch_delta(row)
        geometry = row.get("update_geometry", {})
        consensus = row.get("next_independent_native_consensus") or {}
        record = {
            "probe": row["probe"],
            "stage": row["stage"],
            "step": int(row["step"]),
            "source_state": row["source_state"],
            "horizon": horizon,
            "historical_macro_psnr_delta": historical,
            "continuous_probe_vs_native_branch_delta": branch,
            "contradiction_class": (
                contradiction_class(historical, branch)
                if historical is not None and branch is not None else None
            ),
            "correction_norm": float(geometry.get("correction_norm", 0.0)),
            "correction_reference_cosine": float(
                geometry.get("correction_reference_cosine", 0.0)
            ),
            "next_native_cosine": (
                float(consensus["cosine"]) if "cosine" in consensus else None
            ),
            "paired_label_available_only_after_branch": branch is not None,
        }
        selected.append(record)
    contradictions = [row for row in selected if row["contradiction_class"] in {
        "beneficial_state_with_current_operator_harm",
        "harmful_accumulated_state_with_current_operator_benefit",
    }]
    return {
        "schema": "clean-unsb-search005-evidence-seed-v1",
        "source": str(analysis_path),
        "source_rows": int(payload.get("raw_atlas_rows", len(payload["rows"]))),
        "selected_rows": selected,
        "contradictions": contradictions,
        "interpretation": {
            "historical_delta": "quality of the accumulated method state versus matched plain",
            "branch_delta": "future continuous method versus native continuation from the same method state",
            "why_not_algorithm": "these paired labels are retrospective causal evidence and are inaccessible to training",
            "required_next_test": "pulse the correction then continue both states with synchronized native UNSB to separate field validity from flow propagation",
        },
        "candidate_generated": False,
        "confirmation20_opened": False,
    }
