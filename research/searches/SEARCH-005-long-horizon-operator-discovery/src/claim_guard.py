"""Machine-checkable protection against SEARCH-005 objective drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_TYPES = {"operator", "estimator", "coordinate", "coupled_dynamics"}
FORBIDDEN_TYPES = {
    "controller",
    "schedule",
    "whole_state_branch_selector",
    "branch_selector",
    "handoff",
}
REQUIRED_CARD_FIELDS = {
    "candidate_id",
    "candidate_type",
    "parent_evidence",
    "causal_failure_class",
    "unsb_object",
    "mathematical_update",
    "long_horizon_property",
    "paired_target_access",
    "uses_fixed_window",
    "uses_whole_state_branch_selection",
    "falsification_test",
    "allowed_claim",
    "forbidden_claims",
}


class GoalDriftError(ValueError):
    """Raised when an artifact silently changes the registered research task."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoalDriftError(message)


def validate_derivation_card(card: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CARD_FIELDS - set(card))
    _require(not missing, f"derivation card missing fields: {missing}")
    candidate_type = str(card["candidate_type"])
    _require(candidate_type in ALLOWED_TYPES, f"inadmissible candidate type: {candidate_type}")
    _require(candidate_type not in FORBIDDEN_TYPES, f"forbidden candidate type: {candidate_type}")
    _require(card["paired_target_access"] is False, "paired targets must be inaccessible")
    _require(card["uses_fixed_window"] is False, "fixed windows are outside SEARCH-005")
    _require(
        card["uses_whole_state_branch_selection"] is False,
        "whole-state branch selection is outside SEARCH-005",
    )
    _require(bool(str(card["mathematical_update"]).strip()), "mathematical update is empty")
    _require(bool(str(card["long_horizon_property"]).strip()), "long-horizon property is empty")
    _require(bool(card["parent_evidence"]), "candidate has no causal parent evidence")
    _require(bool(card["forbidden_claims"]), "candidate must declare forbidden claims")


def validate_run_declaration(run: dict[str, Any]) -> None:
    _require(run.get("route") == "route1_mathematical_operator_discovery", "wrong route")
    _require(run.get("candidate_type") in ALLOWED_TYPES, "run has inadmissible candidate type")
    _require(run.get("paired_control") is False, "paired online control is forbidden")
    _require(run.get("whole_state_branch_selection") is False, "branch selection is forbidden")
    max_updates = int(run.get("max_updates", 0))
    if max_updates > 800:
        _require(run.get("formula_review_passed") is True, "long run lacks formula review")
        _require(run.get("counterfactual_passed") is True, "long run lacks counterfactual gate")
        _require(bool(run.get("derivation_card")), "long run lacks derivation card")


def validate_route_exhaustion(record: dict[str, Any]) -> None:
    _require(record.get("route") == "route1_mathematical_operator_discovery", "wrong route")
    supported = set(record.get("supported_failure_classes", []))
    attempted = set(record.get("failure_classes_with_admissible_attempt", []))
    completed = set(record.get("failure_classes_completed", []))
    _require(bool(supported), "route exhaustion requires supported failure classes")
    _require(supported <= attempted, "a supported failure class has no admissible algorithm")
    _require(supported <= completed, "a supported failure class has not completed its protocol")
    _require(
        record.get("controller_failure_used_as_route_evidence") is False,
        "controller failure cannot close operator discovery",
    )


def validate_file(path: Path, kind: str) -> None:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if kind == "card":
        validate_derivation_card(value)
    elif kind == "run":
        validate_run_declaration(value)
    elif kind == "exhaustion":
        validate_route_exhaustion(value)
    else:
        raise ValueError(f"unknown validation kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("card", "run", "exhaustion"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    validate_file(args.path, args.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
