from __future__ import annotations

import torch

from lttr.core import LTTRConfig, lttr_loss, response_statistics


def _stats(x, plus, minus):
    return response_statistics(
        x_t=x, endpoint_plus=plus, endpoint_minus=minus, region_patch=2
    )


def test_identical_response_has_zero_loss():
    x = torch.zeros(1, 3, 4, 4)
    plus = torch.ones_like(x)
    minus = -torch.ones_like(x)
    stats = _stats(x, plus, minus)
    loss, diag = lttr_loss(
        current=stats, teacher=stats, source=x, mode="safe",
        config=LTTRConfig(region_patch=2),
    )
    assert loss.item() == 0.0
    assert diag["tangent"].item() == 0.0


def test_tangent_energy_drift_is_detected_and_differentiable():
    x = torch.zeros(1, 3, 4, 4)
    amplitude = torch.tensor(2.0, requires_grad=True)
    current = _stats(x, amplitude * torch.ones_like(x), -amplitude * torch.ones_like(x))
    teacher = _stats(x, torch.ones_like(x), -torch.ones_like(x))
    loss, _ = lttr_loss(
        current=current, teacher=teacher, source=x, mode="tangent",
        config=LTTRConfig(region_patch=2),
    )
    loss.backward()
    assert loss.item() > 0.0
    assert amplitude.grad is not None and amplitude.grad.abs().item() > 0.0


def test_safe_mode_penalizes_mean_direction_reversal():
    x = torch.zeros(1, 3, 4, 4)
    tangent = 0.1 * torch.ones_like(x)
    current = _stats(x, -torch.ones_like(x) + tangent, -torch.ones_like(x) - tangent)
    teacher = _stats(x, torch.ones_like(x) + tangent, torch.ones_like(x) - tangent)
    tangent_only, _ = lttr_loss(
        current=current, teacher=teacher, source=x, mode="tangent",
        config=LTTRConfig(region_patch=2),
    )
    safe, diag = lttr_loss(
        current=current, teacher=teacher, source=x, mode="safe",
        config=LTTRConfig(region_patch=2),
    )
    assert diag["direction"].item() > 0.0
    assert safe.item() > tangent_only.item()
