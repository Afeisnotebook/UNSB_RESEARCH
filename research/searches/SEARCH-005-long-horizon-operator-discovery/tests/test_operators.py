from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search005_operators", ROOT / "src" / "operators.py"
)
operators = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = operators
SPEC.loader.exec_module(operators)


def test_phcrp_is_exact_native_at_first_bridge_point():
    state = torch.zeros((1, 1, 1, 2))
    endpoint = torch.tensor([[[[1.0, -2.0]]]])
    result, energy, diagnostics = operators.pathwise_horizon_residual_projection(
        state, endpoint, torch.tensor([1.0]), None, None
    )
    assert torch.equal(result, endpoint)
    assert torch.equal(energy, torch.tensor([2.5]))
    assert diagnostics.active_fraction == 0.0


def test_phcrp_enforces_pathwise_linear_horizon_scaling():
    state = torch.zeros((1, 1, 1, 2), dtype=torch.float64)
    endpoint = torch.tensor([[[[2.0, -2.0]]]], dtype=torch.float64, requires_grad=True)
    result, energy, diagnostics = operators.pathwise_horizon_residual_projection(
        state,
        endpoint,
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    assert torch.allclose(energy, torch.tensor([0.5], dtype=torch.float64))
    assert diagnostics.maximum_cap_ratio <= 1.0 + 1e-12
    assert diagnostics.active_fraction == 1.0
    result.sum().backward()
    assert torch.isfinite(endpoint.grad).all()


def test_phcrp_self_nulls_when_native_path_scales_fast_enough():
    state = torch.zeros((1, 1, 1, 1))
    endpoint = torch.tensor([[[[0.25]]]])
    result, energy, diagnostics = operators.pathwise_horizon_residual_projection(
        state, endpoint, torch.tensor([0.25]), torch.tensor([0.5]), torch.tensor([1.0])
    )
    assert torch.equal(result, endpoint)
    assert torch.equal(energy, torch.tensor([0.0625]))
    assert diagnostics.active_fraction == 0.0


def test_phrsup_is_exact_identity_when_native_update_is_defect_safe():
    native = torch.tensor([1.0, 2.0])
    defect = torch.tensor([2.0, 1.0])
    metric = torch.tensor([0.5, 3.0])
    result, diagnostics = operators.rate_safe_native_gradient(native, defect, metric)
    assert torch.equal(result, native)
    assert diagnostics.active is False
    assert diagnostics.projected_defect_alignment > 0


def test_phrsup_projects_unsafe_update_and_preserves_native_descent():
    native = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    defect = torch.tensor([3.0, 0.0], dtype=torch.float64)
    metric = torch.tensor([2.0, 0.5], dtype=torch.float64)
    result, diagnostics = operators.rate_safe_native_gradient(native, defect, metric)
    assert diagnostics.active is True
    assert diagnostics.projected_defect_alignment >= -1e-12
    assert diagnostics.native_descent_alignment >= -1e-12
    assert torch.dot(native * metric, result) >= -1e-12


def test_phrsup_preserves_native_stationary_point():
    zero = torch.zeros(4)
    result, diagnostics = operators.rate_safe_native_gradient(
        zero, torch.arange(4, dtype=torch.float32), torch.ones(4)
    )
    assert torch.equal(result, zero)
    assert diagnostics.active is False


def test_pcoa_first_step_and_unpredictable_field_are_native():
    current = torch.tensor([1.0, -2.0])
    first, first_diag = operators.predictability_gated_optimistic_displacement(
        current, None
    )
    assert torch.equal(first, current)
    assert first_diag.predictability == 0.0
    anti, anti_diag = operators.predictability_gated_optimistic_displacement(
        current, -current
    )
    assert torch.equal(anti, current)
    assert anti_diag.predictability == 0.0


def test_pcoa_recovers_omd_under_perfect_predictability():
    current = torch.tensor([2.0, 1.0])
    previous = torch.tensor([1.0, 1.0])
    result, diagnostics = operators.predictability_gated_optimistic_displacement(
        current, previous
    )
    assert diagnostics.predictability == 1.0
    assert torch.equal(result, 2 * current - previous)


def test_pcoa_self_nulls_on_unchanged_native_field():
    current = torch.tensor([0.5, -0.25])
    result, diagnostics = operators.predictability_gated_optimistic_displacement(
        current, current.clone()
    )
    assert torch.equal(result, current)
    assert diagnostics.predictability == 1.0
    assert diagnostics.correction_norm == 0.0


def test_npooa_preserves_native_norm_and_removes_radial_acceleration():
    current = torch.tensor([4.0, 0.0], dtype=torch.float64)
    previous = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result, diagnostics = operators.norm_preserving_orthogonal_optimistic_displacement(
        current, previous
    )
    assert torch.allclose(result.norm(), current.norm(), atol=1e-12, rtol=1e-12)
    assert diagnostics.norm_ratio == pytest.approx(1.0, abs=1e-12)
    assert diagnostics.orthogonal_innovation_norm > 0
    assert not torch.equal(result, current)


def test_npooa_is_native_for_first_nonpredictive_and_collinear_fields():
    current = torch.tensor([2.0, -1.0], dtype=torch.float64)
    first, _ = operators.norm_preserving_orthogonal_optimistic_displacement(current, None)
    anti, anti_diag = operators.norm_preserving_orthogonal_optimistic_displacement(
        current, -current
    )
    collinear, collinear_diag = operators.norm_preserving_orthogonal_optimistic_displacement(
        current, 0.25 * current
    )
    assert torch.equal(first, current)
    assert torch.equal(anti, current)
    assert anti_diag.predictability == 0.0
    assert torch.allclose(collinear, current, atol=1e-12, rtol=1e-12)
    assert collinear_diag.orthogonal_innovation_norm == pytest.approx(0.0, abs=1e-12)


def test_bcavp_is_exact_identity_when_native_variance_is_feasible():
    positive = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    negative = positive - 0.01
    result, diagnostics = operators.brownian_antithetic_variance_projection(
        positive, negative, torch.tensor([0.5]), tau=0.01
    )
    assert torch.equal(result, positive)
    assert diagnostics.active_fraction == 0.0
    assert diagnostics.mean_scale == 1.0


def test_bcavp_preserves_antithetic_mean_and_enforces_brownian_cap():
    positive = torch.tensor([[[[3.0, 1.0]]]], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([[[[-1.0, 1.0]]]], dtype=torch.float64, requires_grad=True)
    horizon = torch.tensor([0.25], dtype=torch.float64)
    result, diagnostics = operators.brownian_antithetic_variance_projection(
        positive, negative, horizon, tau=0.5
    )
    reflected, _ = operators.brownian_antithetic_variance_projection(
        negative, positive, horizon, tau=0.5
    )
    midpoint = 0.5 * (positive + negative)
    assert torch.allclose(0.5 * (result + reflected), midpoint, atol=1e-12, rtol=0)
    projected_half_difference = 0.5 * (result - reflected)
    assert projected_half_difference.square().mean() <= 0.5 * 0.25 + 1e-12
    assert diagnostics.active_fraction == 1.0
    assert diagnostics.maximum_projected_ratio <= 1.0 + 1e-12
    result.sum().backward()
    assert torch.isfinite(positive.grad).all()
    assert torch.isfinite(negative.grad).all()


def test_bcavp_zero_horizon_removes_only_odd_latent_component():
    positive = torch.tensor([[[[2.0]]]])
    negative = torch.tensor([[[[0.0]]]])
    result, diagnostics = operators.brownian_antithetic_variance_projection(
        positive, negative, torch.tensor([0.0]), tau=0.01
    )
    assert torch.equal(result, torch.tensor([[[[1.0]]]]))
    assert diagnostics.mean_scale == 0.0
    assert diagnostics.maximum_projected_ratio == 0.0


def test_cndrp_is_identity_without_sensitivity():
    gradient = torch.tensor([1.0, -2.0])
    zero = torch.zeros_like(gradient)
    result, _ = operators.cndrp_precondition(gradient, zero, zero)
    assert torch.equal(result, gradient)


def test_cndrp_is_positive_definite_and_preserves_stationary_points():
    gradient = torch.tensor([1.0, 2.0])
    first = torch.tensor([3.0, 0.0])
    second = torch.tensor([3.0, 0.0])
    result, _ = operators.cndrp_precondition(gradient, first, second)
    assert torch.dot(gradient, result) > 0
    assert result[0].item() > 0
    zero, _ = operators.cndrp_precondition(torch.zeros(2), first, second)
    assert torch.equal(zero, torch.zeros(2))


def test_cndrp_preserves_descent_in_positive_diagonal_adam_metric():
    gradient = torch.tensor([1.0, -2.0, 0.5])
    first = torch.tensor([3.0, -4.0, 2.0])
    second = torch.tensor([2.5, -3.0, -1.0])
    adam_metric = torch.tensor([1e-3, 100.0, 0.2])
    result, diagnostics = operators.cndrp_precondition(gradient, first, second)
    assert torch.sum(gradient * adam_metric * result) > 0
    assert diagnostics["minimum_scale"] > 0
    assert diagnostics["maximum_scale"] <= 1


def test_cndrp_high_variation_returns_toward_native():
    gradient = torch.tensor([1.0, 0.0])
    low, _ = operators.cndrp_precondition(
        gradient, torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
    )
    high, _ = operators.cndrp_precondition(
        gradient, torch.tensor([10.0, 0.0]), torch.tensor([-8.0, 0.0])
    )
    assert abs(high[0] - gradient[0]) < abs(low[0] - gradient[0])


def test_bcnrp_preserves_direction_inside_each_parameter_block():
    gradient = torch.tensor([1.0, -2.0, 3.0])
    first = torch.tensor([3.0, 1.0, -2.0])
    second = torch.tensor([2.0, 0.0, -1.0])
    result, diagnostics = operators.block_confidence_precondition(
        gradient, first, second
    )
    ratios = result / gradient
    assert torch.allclose(ratios, torch.full_like(ratios, ratios[0]))
    assert 0 < diagnostics["mean_scale"] <= 1


def test_bcnrp_is_spd_in_any_positive_diagonal_adam_metric():
    gradient = torch.tensor([1.0, -2.0, 0.5])
    first = torch.tensor([10.0, -3.0, 1.0])
    second = torch.tensor([9.0, -1.0, -2.0])
    metric = torch.tensor([0.001, 30.0, 2.0])
    result, _ = operators.block_confidence_precondition(gradient, first, second)
    assert torch.sum(gradient * metric * result) > 0
    zero, _ = operators.block_confidence_precondition(
        torch.zeros_like(gradient), first, second
    )
    assert torch.equal(zero, torch.zeros_like(gradient))


def test_acmp_rotates_into_both_descent_halfspaces_and_respects_trust_region():
    raw = torch.tensor([-2.0, -1.0])
    native = torch.tensor([1.0, 0.0])
    bridge = torch.tensor([0.0, 1.0])
    metric = torch.ones(2)
    projected, diag = operators.acmp_project(raw, native, bridge, metric)
    assert torch.dot(native, projected) >= -1e-7
    assert torch.dot(bridge, projected) >= -1e-7
    assert torch.linalg.vector_norm(projected) <= torch.linalg.vector_norm(native) + 1e-7
    assert diag.native_alignment >= -1e-7
    assert diag.bridge_alignment >= -1e-7


def test_acmp_is_identity_for_feasible_bounded_correction():
    raw = torch.tensor([0.2, 0.3])
    native = torch.tensor([1.0, 0.0])
    bridge = torch.tensor([0.0, 1.0])
    projected, diag = operators.acmp_project(raw, native, bridge, torch.ones(2))
    assert torch.allclose(projected, raw)
    assert diag.active_constraints == ()
