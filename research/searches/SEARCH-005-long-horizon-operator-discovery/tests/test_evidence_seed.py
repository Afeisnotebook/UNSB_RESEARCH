from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search005_evidence_seed", ROOT / "src" / "evidence_seed.py"
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_contradiction_classes_separate_state_from_operator():
    assert module.contradiction_class(0.5, -0.2) == (
        "beneficial_state_with_current_operator_harm"
    )
    assert module.contradiction_class(-0.5, 0.2) == (
        "harmful_accumulated_state_with_current_operator_benefit"
    )
    assert module.contradiction_class(0.5, 0.2) == (
        "state_and_current_operator_both_beneficial"
    )
