"""Evidence-routed Generation-2 revision after the Generation-1 micro run."""

from __future__ import annotations

import json
from pathlib import Path

from .ledger import HypothesisLedger
from .search001_compat import modules


REVISIONS = {
    "G1-DT-RHGC8": {
        "id": "G2-DT-FBGC8",
        "name": "Future-Batch Gradient Consensus for DT",
        "probe": "dt",
        "observable": "G_GAN",
        "direction": -1,
    },
    "G1-HJ-RHDFC8": {
        "id": "G2-HJ-FBDFC8",
        "name": "Future-Batch Gradient Consensus for HJ",
        "probe": "hj",
        "observable": "D_fake",
        "direction": 1,
    },
}


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def revise_generation2(output_dir: Path, small_dir: Path) -> dict:
    output_dir = Path(output_dir)
    small_dir = Path(small_dir)
    baseline_dir = small_dir / "G1-DT-RHGC8__observable_only"
    compare = modules()[2].compare
    ledger = HypothesisLedger(output_dir / "HYPOTHESIS_LEDGER.json")
    cards_dir = output_dir / "DERIVATION_CARDS"
    cards_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for parent, revision in REVISIONS.items():
        lane = small_dir / f"{parent}__full"
        parent_400 = _read(lane / "metrics_step_400.json")
        parent_800 = _read(lane / "metrics_step_800.json")
        plain_400 = _read(baseline_dir / "metrics_step_400.json")
        plain_800 = _read(baseline_dir / "metrics_step_800.json")
        delta_400 = compare(parent_400, plain_400, step=400)
        delta_800 = compare(parent_800, plain_800, step=800)
        if not (
            float(delta_400["macro_psnr_delta"]) > 0.0
            and float(delta_800["macro_psnr_delta"]) < 0.0
        ):
            raise RuntimeError(f"{parent} lacks the required positive-to-negative micro reversal")
        run_state = _read(lane / "RUN_STATE.json")
        card = {
            "id": revision["id"],
            "generation": 2,
            "name": revision["name"],
            "parents": [parent],
            "observed_failure": (
                f"{parent} changed from {delta_400['macro_psnr_delta']:+.6f} dB at 400 "
                f"to {delta_800['macro_psnr_delta']:+.6f} dB at 800 despite committing "
                f"{run_state['proposal_blocks']}/{run_state['decision_blocks']} proposal blocks"
            ),
            "unsb_object": (
                "one-sided alignment of the proposal correction with the next-batch "
                "native UNSB generator gradient"
            ),
            "operator": "two-gate target-blind receding-horizon full-state selection",
            "derivation": (
                "Run B0=Phi_0^8(S) and Bi=Phi_i^8(S). Let delta be the G/F "
                "parameter correction theta_i-theta_0 and let g+ be the native UNSB "
                "generator gradient at B0 on the next independent unpaired batch. "
                f"Commit Bi only when {revision['direction']}*(Delta "
                f"{revision['observable']})>0 and <delta,g+><0."
            ),
            "identity_condition": (
                "failure of either the audited lead gate or future-gradient gate commits "
                "the exact plain full state"
            ),
            "self_null_condition": (
                "rejected proposal and audit computations contribute no parameters, moments, "
                "scheduler, stream, RNG or method co-state"
            ),
            "changes_training_target": False,
            "changes_gradient_estimator": False,
            "changes_endpoint_law": False,
            "paired_target_access": False,
            "paired_target_proof": (
                "both gates read only unpaired training losses, G/F parameter displacement "
                "and a native gradient on the next unpaired batch"
            ),
            "expected_valid_states": (
                "states where the historical short lead and an independent future native "
                "descent direction agree"
            ),
            "falsification_test": (
                "from the same e0, kill after the 800-update micro run if final delta is "
                "non-positive or guardrails fail; no further revision is allowed"
            ),
            "compute_cost": (
                "plain and proposal 8-update branches plus one native next-batch "
                "generator forward/backward per block"
            ),
            "memory_cost": "one live branch plus CPU full-state snapshots",
            "pseudocode": [
                "plain = Phi_0^8(S); proposal = Phi_i^8(S)",
                "lead = direction * (observable(proposal)-observable(plain)) > 0",
                "g_future = grad L_UNSB(plain; next_unpaired_batch)",
                "consensus = dot(theta_proposal-theta_plain, g_future) < 0",
                "commit proposal iff lead and consensus else exact plain",
            ],
            "status": "proposed",
        }
        if any(entry["id"] == parent and entry["status"] == "proposed" for entry in ledger.entries):
            ledger.transition(
                parent,
                status="failed_micro_reversal",
                evidence={"step_400": delta_400, "step_800": delta_800},
            )
        if not any(entry["id"] == card["id"] for entry in ledger.entries):
            ledger.append(card)
        (cards_dir / f"{card['id']}.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (cards_dir / f"{card['id']}.md").write_text(
            "\n".join([
                f"# {card['id']}: {card['name']}", "",
                "## New failure evidence", "", card["observed_failure"] + ".", "",
                "## Operator", "", card["derivation"], "",
                "## Exact self-null", "", card["self_null_condition"] + ".", "",
                "## Falsification", "", card["falsification_test"] + ".", "",
            ]),
            encoding="utf-8",
        )
        reports.append({
            "parent": parent,
            "revision": card,
            "parent_step_400": delta_400,
            "parent_step_800": delta_800,
        })
    result = {
        "schema": "clean-unsb-search003-generation2-v1",
        "revision_count": len(reports),
        "revisions": reports,
        "new_failure_class": "in_branch_observable_does_not_ensure_future_native_descent",
        "hyperparameter_search_used": False,
        "paired_target_access_by_controller": False,
        "confirmation20_opened": False,
    }
    (output_dir / "GENERATION2_REVISION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
