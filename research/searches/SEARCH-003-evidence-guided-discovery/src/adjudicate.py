"""Adjudicate Generation-0 probes without inventing a candidate in advance."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


DECISIVE_GRID = {
    "dt": (("full", 2000, "pre"), ("full", 3000, "near"), ("full", 4000, "post")),
    "hj": (("small", 400, "pre"), ("small", 800, "near"), ("small", 1200, "post")),
    "hnek": (("full", 2000, "pre"), ("full", 3000, "near"), ("full", 4000, "post")),
}


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _future_delta(row: dict) -> float | None:
    label = row.get("post_branch_development_label")
    return None if label is None else float(label["macro_psnr_delta"])


def four_cell_case(plain_delta: float, own_delta: float) -> str:
    """Route the four-cell causal matrix using only post-branch labels."""
    plain_good = plain_delta > 0.0
    own_good = own_delta > 0.0
    if plain_good and own_good:
        return "operator_locally_sustainable"
    if plain_good and not own_good:
        return "state_feedback_missing"
    if not plain_good and not own_good:
        return "operator_locally_harmful"
    return "benefit_requires_method_state"


def _cell_record(rows: list[dict], probe: str, stage: str, step: int) -> dict | None:
    selected = [
        row for row in rows
        if row["probe"] == probe
        and row["stage"] == stage
        and int(row["step"]) == int(step)
        and int(row.get("horizon", 1)) == 200
        and row.get("diagnostic_validity", {}).get("causal_operator_state")
        != "invalid_teacher_transplant"
    ]
    # Prefer the explicit reinitialized/matched schema if a normalized legacy
    # row and its corrected replacement coexist.
    priority = {
        "reinitialized_from_source_state": 3,
        "matched_historical_costate": 3,
        "stateless_equivalent": 2,
        "legacy_transplanted_method_costate": 1,
    }
    by_state = {}
    for row in selected:
        state = row["source_state"]
        if state not in by_state or priority.get(row.get("operator_costate"), 0) > priority.get(
            by_state[state].get("operator_costate"), 0
        ):
            by_state[state] = row
    if "plain" not in by_state or probe not in by_state:
        return None
    plain = by_state["plain"]
    own = by_state[probe]
    plain_delta = _future_delta(plain)
    own_delta = _future_delta(own)
    if plain_delta is None or own_delta is None:
        return None
    return {
        "probe": probe,
        "stage": stage,
        "step": int(step),
        "horizon": 200,
        "plain_state_proposal_delta": plain_delta,
        "own_state_proposal_delta": own_delta,
        "plain_state_positive_domains": int(
            plain["post_branch_development_label"]["positive_domains"]
        ),
        "own_state_positive_domains": int(
            own["post_branch_development_label"]["positive_domains"]
        ),
        "plain_state_correction_cosine": float(
            plain["update_geometry"]["correction_reference_cosine"]
        ),
        "own_state_correction_cosine": float(
            own["update_geometry"]["correction_reference_cosine"]
        ),
        "plain_state_correction_ratio": float(
            plain["update_geometry"]["correction_norm"]
            / max(float(plain["update_geometry"]["reference_norm"]), 1e-20)
        ),
        "own_state_correction_ratio": float(
            own["update_geometry"]["correction_norm"]
            / max(float(own["update_geometry"]["reference_norm"]), 1e-20)
        ),
        "causal_case": four_cell_case(plain_delta, own_delta),
    }


def adjudicate_generation0(atlas_path: Path, analysis_path: Path) -> dict:
    rows = _read_jsonl(atlas_path)
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    probes = {}
    missing = []
    for probe, grid in DECISIVE_GRID.items():
        cells = []
        for stage, step, phase in grid:
            cell = _cell_record(rows, probe, stage, step)
            if cell is None:
                missing.append({"probe": probe, "stage": stage, "step": step})
                continue
            cell["phase"] = phase
            cells.append(cell)
        cases = [cell["causal_case"] for cell in cells]
        if not cells:
            verdict = "awaiting_decisive_audit"
        elif any(case == "state_feedback_missing" for case in cases):
            verdict = "reversal_observed_state_feedback_route"
        elif any(case == "operator_locally_harmful" for case in cases):
            verdict = "reversal_observed_operator_rewrite_route"
        elif all(case == "operator_locally_sustainable" for case in cases):
            verdict = "operator_sustained_in_audit_implementation_or_longer_state_route"
        else:
            verdict = "state_dependent_route"
        probes[probe] = {
            "decisive_cells": cells,
            "verdict": verdict,
            "mechanism_falsified": False,
            "reversal_observed": any(
                cell["plain_state_proposal_delta"] <= 0.0
                or cell["own_state_proposal_delta"] <= 0.0
                for cell in cells
            ),
            "closed_current_protocol": False,
            "variance_claim": "not_identifiable_from_single_seed_counterfactuals",
        }

    by_probe_horizon = defaultdict(list)
    for row in rows:
        if row.get("post_branch_development_label") is not None:
            by_probe_horizon[(row["probe"], int(row.get("horizon", 1)))].append(
                float(row["post_branch_development_label"]["macro_psnr_delta"])
            )

    return {
        "schema": "clean-unsb-search003-generation0-adjudication-v1",
        "complete": not missing,
        "missing_decisive_cells": missing,
        "probes": probes,
        "target_blind_signal_gate": {
            "shared_signal_gate_passed": bool(analysis["shared_signal_gate_passed"]),
            "eligible_shared_signals": analysis["eligible_shared_signals"],
            "consequence": (
                "shared_controller_route_permitted"
                if analysis["shared_signal_gate_passed"]
                else "shared_controller_route_closed_unless_decisive_audit_adds_evidence"
            ),
            "eligible_method_signals": analysis.get("eligible_method_signals", {}),
        },
        "observed_branch_delta_ranges": {
            f"{probe}::H{horizon}": [min(values), max(values)]
            for (probe, horizon), values in sorted(by_probe_horizon.items())
        },
        "paired_metrics_available_to_controller": False,
        "confirmation20_opened": False,
    }


def generation0_markdown(report: dict) -> str:
    lines = [
        "# SEARCH-003 Generation 0 causal adjudication",
        "",
        f"Complete decisive grid: **{report['complete']}**.",
        "Paired development labels were joined only after each branch and are not controller inputs.",
        "",
    ]
    for probe, value in report["probes"].items():
        lines += [f"## {probe.upper()}", "", f"Verdict: `{value['verdict']}`.", ""]
        for cell in value["decisive_cells"]:
            lines.append(
                "- {phase} {stage}@{step}: plain-state {plain:+.4f} dB, "
                "own-state {own:+.4f} dB, `{case}`.".format(
                    phase=cell["phase"], stage=cell["stage"], step=cell["step"],
                    plain=cell["plain_state_proposal_delta"],
                    own=cell["own_state_proposal_delta"], case=cell["causal_case"],
                )
            )
        lines.append("")
    signal = report["target_blind_signal_gate"]
    lines += [
        "## Signal gate",
        "",
        f"Shared signal passed: **{signal['shared_signal_gate_passed']}**.",
        f"Consequence: `{signal['consequence']}`.",
        "",
    ]
    return "\n".join(lines)
