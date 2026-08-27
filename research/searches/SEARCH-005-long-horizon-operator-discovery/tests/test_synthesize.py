from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search005_synthesize", ROOT / "src" / "synthesize.py"
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_retention_class_is_about_flow_not_window_selection():
    assert module.retention_class(0.5, -0.1) == (
        "positive_impulse_reversed_by_native_flow"
    )
    assert module.retention_class(0.5, 0.05) == (
        "positive_impulse_strongly_attenuated"
    )
    assert module.retention_class(0.0, 0.0) == "operator_exactly_null"
