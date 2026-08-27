from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "search005_candidate_runtime_tests"
if PACKAGE not in sys.modules:
    package = type(sys)(PACKAGE)
    package.__path__ = [str(ROOT / "src")]
    package.__package__ = PACKAGE
    sys.modules[PACKAGE] = package
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE}.candidate_runtime", ROOT / "src" / "candidate_runtime.py"
)
candidate_runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = candidate_runtime
SPEC.loader.exec_module(candidate_runtime)


def test_eliprc_configuration_is_endpoint_law_invariant_variant():
    assert dict(candidate_runtime.ELIPRC.option_overrides) == {
        "--hnek_gamma": "0.5",
        "--hnek_coord": "residual",
        "--hnek_horizon_mode": "physical",
        "--hnek_partial": "entropy_only",
    }


def test_cndrp_uses_plain_endpoint_model_and_search_scoped_installer():
    assert candidate_runtime.CNDRP.model == "sb"
    assert candidate_runtime.CNDRP.option_overrides == ()
    assert candidate_runtime.CNDRP.installer == "cndrp"


def test_acmp_uses_plain_model_with_search_scoped_hj_probe():
    assert candidate_runtime.ACMP.model == "sb"
    assert candidate_runtime.ACMP.option_overrides == ()
    assert candidate_runtime.ACMP.installer == "acmp"


def test_fbcmp_is_a_second_generation_operator_not_a_window():
    assert candidate_runtime.FBCMP.model == "sb"
    assert candidate_runtime.FBCMP.option_overrides == ()
    assert candidate_runtime.FBCMP.installer == "fbcmp"


def test_bcavp_uses_plain_model_and_search_scoped_physical_projection():
    assert candidate_runtime.BCAVP.model == "sb"
    assert candidate_runtime.BCAVP.option_overrides == ()
    assert candidate_runtime.BCAVP.installer == "bcavp"


def test_phcrp_is_a_pathwise_coordinate_without_schedule_options():
    assert candidate_runtime.PHCRP.model == "sb"
    assert candidate_runtime.PHCRP.option_overrides == ()
    assert candidate_runtime.PHCRP.installer == "phcrp"


def test_phrsup_is_endpoint_native_second_generation_projection():
    assert candidate_runtime.PHRSUP.model == "sb"
    assert candidate_runtime.PHRSUP.option_overrides == ()
    assert candidate_runtime.PHRSUP.installer == "phrsup"


def test_bcnrp_is_dt_second_generation_block_metric():
    assert candidate_runtime.BCNRP.model == "sb"
    assert candidate_runtime.BCNRP.option_overrides == ()
    assert candidate_runtime.BCNRP.installer == "bcnrp"


def test_pcoa_is_coupled_dynamics_not_endpoint_or_schedule():
    assert candidate_runtime.PCOA.model == "sb"
    assert candidate_runtime.PCOA.option_overrides == ()
    assert candidate_runtime.PCOA.installer == "pcoa"


def test_npooa_is_norm_preserving_coupled_dynamics():
    assert candidate_runtime.NPOOA.model == "sb"
    assert candidate_runtime.NPOOA.option_overrides == ()
    assert candidate_runtime.NPOOA.installer == "npooa"


def test_replace_option_rejects_missing_or_duplicate_values():
    arguments = ["--x", "old"]
    candidate_runtime.replace_option(arguments, "--x", "new")
    assert arguments == ["--x", "new"]
    with pytest.raises(RuntimeError):
        candidate_runtime.replace_option(arguments, "--missing", "new")
    with pytest.raises(RuntimeError):
        candidate_runtime.replace_option(["--x", "1", "--x", "2"], "--x", "3")
