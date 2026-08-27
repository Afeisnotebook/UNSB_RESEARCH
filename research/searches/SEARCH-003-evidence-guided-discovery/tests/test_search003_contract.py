from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"search003_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_from_package(name: str):
    package_name = "search003_test_package"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "src")]
        package.__package__ = package_name
        sys.modules[package_name] = package
    qualified = f"{package_name}.{name}"
    path = ROOT / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_freezes_evidence_not_candidates():
    protocol = load("protocol").Search003Protocol()
    assert protocol.confirmation20_opened is False
    assert protocol.max_generation1_candidates == 6
    assert not any(probe.trainable_candidate for probe in protocol.probes)
    assert {probe.name for probe in protocol.probes} >= {"dt", "hj", "hnek", "ptq"}


def test_receding_selector_uses_audited_sign_and_aligned_period():
    receding = load_from_package("receding")
    assert receding.proposal_selected(-1, 0.9, 1.0)
    assert not receding.proposal_selected(-1, 1.1, 1.0)
    assert receding.proposal_selected(1, 1.1, 1.0)
    assert not receding.proposal_selected(1, 0.9, 1.0)
    eval_steps = (400, 800, 1200, 1600, 2000, 2400)
    for spec in receding.CANDIDATES.values():
        assert all(step % spec.horizon == 0 for step in eval_steps)


def test_state_observation_blocks_paired_metrics():
    observations = load("observations")
    with pytest.raises(ValueError, match="paired metrics"):
        observations.StateObservation(
            source_probe="plain", source_state="plain", step=0,
            paired_metrics_accessed=True,
        )


def test_update_geometry_identifies_sign_reversal():
    observations = load("observations")
    before = {"block.weight": torch.tensor([0.0, 0.0])}
    reference = {"block.weight": torch.tensor([1.0, 0.0])}
    proposal = {"block.weight": torch.tensor([-1.0, 0.0])}
    global_geometry, blocks = observations.state_dict_update_geometry(
        before, reference, proposal
    )
    assert global_geometry["reference_proposal_cosine"] == pytest.approx(-1.0)
    assert global_geometry["correction_reference_cosine"] == pytest.approx(-1.0)
    assert "block" in blocks
    consensus = observations.state_dict_delta_cosine(
        before, proposal, before, reference
    )
    assert consensus["cosine"] == pytest.approx(-1.0)


def test_ledger_requires_target_blind_derivation(tmp_path):
    ledger_module = load("ledger")
    ledger = ledger_module.HypothesisLedger(tmp_path / "HYPOTHESIS_LEDGER.json")
    entry = {
        "id": "HYP-001",
        "generation": 1,
        "parents": ["dt"],
        "observed_failure": "drift",
        "unsb_object": "endpoint law",
        "operator": "rate barrier",
        "identity_condition": "rate inside noise band",
        "self_null_condition": "correction=0",
        "paired_target_access": False,
        "falsification_test": "200-step branch",
        "status": "proposed",
    }
    ledger.append(entry)
    assert ledger.entries[0]["id"] == "HYP-001"
    bad = dict(entry, id="HYP-002", paired_target_access=True)
    with pytest.raises(ValueError, match="paired targets"):
        ledger.append(bad)


def test_small_promotion_gate():
    protocol = load("protocol")
    trajectory = []
    for index, step in enumerate((400, 800, 1200, 1600, 2000, 2400)):
        trajectory.append({
            "step": step,
            "macro_psnr": 14.0 + 0.05 * index,
            "macro_psnr_delta": 0.1 if step >= 1600 else 0.0,
            "positive_domains": 5,
            "worst_domain_delta": -0.2,
        })
    assert protocol.promotion_decision(trajectory)["promote"] is True


def test_preserved_catalog_has_matched_decisive_states():
    catalog = load("catalog")
    rows = catalog.preserved_catalog(Path(r"E:\UNSB_Expl\runs"))
    assert any(row.probe == "dt" and row.step == 2000 for row in rows)
    assert any(row.probe == "hj" and row.step == 1200 for row in rows)
    assert any(row.probe == "hnek" and row.step == 12000 for row in rows)
    assert all(row.plain.is_file() and row.method.is_file() for row in rows)


def test_shared_signal_requires_every_heldout_method_to_be_decidable():
    analyze = load_from_package("analyze")
    rows = []
    for probe, deltas in (("a", (1.0, 2.0)), ("b", (-1.0, -2.0))):
        for delta in deltas:
            rows.append({
                "probe": probe,
                "features": {"x": delta},
                "post_branch_development_label": {
                    "macro_psnr_delta": delta,
                    "positive_domains": 6 if delta > 0 else 0,
                },
            })
    # Each held-out method contains only one class, so its balanced accuracy
    # is undefined and cannot support a cross-method claim.
    rows *= 2
    result = analyze.evaluate_sign_signal(rows, "x", 1)
    assert result is not None
    assert result["finite_fold_count"] == 0
    assert result["required_method_fold_count"] == 2


def test_constraint_projector_rotates_or_zeros_without_target_fields():
    interfaces = load_from_package("interfaces")
    projector = interfaces.ConstraintProjector()
    correction = torch.tensor([1.0, 1.0])
    result = projector.project(
        correction,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0.0]),
        radius=2.0,
    )
    assert result.feasible
    assert result.correction[0].item() == pytest.approx(0.0)
    assert result.correction[1].item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="target-aware"):
        interfaces.CandidateUpdate(
            hypothesis_id="x",
            correction=torch.zeros(2),
            defect_before=1.0,
            predicted_defect_direction=-1.0,
            diagnostics={"psnr": 1.0},
        )


def test_four_cell_adjudication_routes_state_feedback():
    adjudicate = load("adjudicate")
    assert adjudicate.four_cell_case(0.1, -0.1) == "state_feedback_missing"
    assert adjudicate.four_cell_case(0.1, 0.2) == "operator_locally_sustainable"
