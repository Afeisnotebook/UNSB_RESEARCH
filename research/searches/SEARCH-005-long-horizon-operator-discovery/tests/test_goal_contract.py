from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search005_claim_guard", ROOT / "src" / "claim_guard.py"
)
claim_guard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = claim_guard
SPEC.loader.exec_module(claim_guard)


def valid_card() -> dict:
    return {
        "candidate_id": "example",
        "candidate_type": "operator",
        "parent_evidence": ["AUDIT-EXAMPLE"],
        "causal_failure_class": "mean_direction_reversal",
        "unsb_object": "generator update field",
        "mathematical_update": "u_new = u0 + A(S)c",
        "long_horizon_property": "A is identity-safe and self-null at zero defect",
        "paired_target_access": False,
        "uses_fixed_window": False,
        "uses_whole_state_branch_selection": False,
        "falsification_test": "same-state and 200-step propagation audit",
        "allowed_claim": "tests one correction-field reconstruction",
        "forbidden_claims": ["all algorithms are exhausted"],
    }


def test_valid_operator_card_passes():
    claim_guard.validate_derivation_card(valid_card())


@pytest.mark.parametrize("candidate_type", ["controller", "schedule", "handoff"])
def test_controller_schedule_and_handoff_are_rejected(candidate_type):
    card = valid_card()
    card["candidate_type"] = candidate_type
    with pytest.raises(claim_guard.GoalDriftError):
        claim_guard.validate_derivation_card(card)


def test_whole_state_selector_cannot_be_renamed_operator():
    card = valid_card()
    card["uses_whole_state_branch_selection"] = True
    with pytest.raises(claim_guard.GoalDriftError, match="whole-state"):
        claim_guard.validate_derivation_card(card)


def test_long_run_requires_formula_and_counterfactual_review():
    run = {
        "route": "route1_mathematical_operator_discovery",
        "candidate_type": "operator",
        "paired_control": False,
        "whole_state_branch_selection": False,
        "max_updates": 2400,
        "formula_review_passed": False,
        "counterfactual_passed": True,
        "derivation_card": "DERIVATION_CARDS/example.json",
    }
    with pytest.raises(claim_guard.GoalDriftError, match="formula review"):
        claim_guard.validate_run_declaration(run)


def test_route_cannot_close_with_untested_supported_failure_class():
    record = {
        "route": "route1_mathematical_operator_discovery",
        "supported_failure_classes": ["mean_reversal", "costate_amplification"],
        "failure_classes_with_admissible_attempt": ["mean_reversal"],
        "failure_classes_completed": ["mean_reversal"],
        "controller_failure_used_as_route_evidence": False,
    }
    with pytest.raises(claim_guard.GoalDriftError, match="no admissible algorithm"):
        claim_guard.validate_route_exhaustion(record)
