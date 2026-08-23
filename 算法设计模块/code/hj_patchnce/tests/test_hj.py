"""CPU tests for the clean HJ-PatchNCE core."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hj.core import StructureProjectConfig, structure_project_nce_step  # noqa: E402
from hj.projection import (  # noqa: E402
    apply_absolute_evidence_gate,
    apply_factorial_structure_control,
    conservative_central_delta,
    positive_quantile_gate,
    project_conflicting_gradient,
)
from hj.structure import source_structure_direction  # noqa: E402


def test_project_conflicting_gradient_forward_identity():
    feat = torch.randn(4, 8, requires_grad=True)
    direction = torch.randn(4, 8)
    out = project_conflicting_gradient(feat, direction, strength=0.5)
    assert torch.equal(out, feat)


def test_project_conflicting_gradient_backward_removes_positive_alignment():
    feat = torch.randn(2, 3, requires_grad=True)
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    strength = torch.tensor(1.0)
    out = project_conflicting_gradient(feat, direction, strength=strength, update_mode="remove")
    out.sum().backward()
    grad = feat.grad
    # conflict along direction dim is removed, perpendicular dims untouched
    assert abs(grad[0, 0].item()) < 1e-5
    assert abs(grad[0, 1].item() - 1.0) < 1e-5
    assert abs(grad[1, 1].item()) < 1e-5
    assert abs(grad[1, 0].item() - 1.0) < 1e-5


def test_conservative_central_delta():
    center = torch.tensor([1.0, 1.0])
    descent = torch.tensor([1.2, 0.9])
    ascent = torch.tensor([0.8, 0.8])
    delta, one, central = conservative_central_delta(center, descent, ascent, 0.1)
    assert torch.allclose(delta, torch.minimum(one, central))


def test_positive_quantile_gate():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    gate = positive_quantile_gate(values, batch_size=1, quantile=0.5)
    assert gate.shape == (1, 4)
    assert int(gate.sum().item()) >= 1


def test_factorial_structure_control_true_preserves_gated_direction():
    direction = torch.randn(2, 5, 7)
    gate = torch.zeros(2, 5)
    gate[:, 0] = 1.0
    controlled, effective_gate = apply_factorial_structure_control(direction, gate, "true")
    assert controlled.shape == direction.shape
    assert torch.equal(effective_gate, gate)


def test_absolute_evidence_gate():
    gate = torch.ones(2, 4)
    scale = torch.tensor([0.1, 1.0])
    out = apply_absolute_evidence_gate(gate, scale, minimum=0.5)
    assert out[0].sum().item() == 0.0
    assert out[1].sum().item() == 4.0


def test_source_structure_direction_shape_and_finite():
    target = torch.rand(1, 3, 16, 16, requires_grad=True)
    source = torch.rand(1, 3, 16, 16)
    direction = source_structure_direction(target, source, direction="joint", scales="1,2,4")
    assert direction.shape == target.shape
    assert torch.isfinite(direction).all()
    assert not direction.requires_grad


def _fake_criterion(f_q, f_k):
    return (f_q * f_k.detach()).sum(dim=1)


def _run_step(schedule_weight):
    torch.manual_seed(0)
    B, N, D = 2, 4, 6
    feat_q = torch.randn(B * N, D, requires_grad=True)
    feat_k = torch.randn(B * N, D)
    source = torch.rand(B, 3, 16, 16)
    tgt_nce = torch.rand(B, 3, 16, 16, requires_grad=True)
    cfg = StructureProjectConfig()

    def probe_fn(tgt):
        scale = tgt.flatten(1).mean(dim=1)  # [B]
        scale = scale.repeat_interleave(feat_q.shape[0] // B, dim=0).unsqueeze(1)
        return feat_q.detach() * (1.0 + scale)

    projected, diag = structure_project_nce_step(
        feat_q=feat_q,
        feat_k=feat_k,
        criterion=_fake_criterion,
        source=source,
        tgt_nce=tgt_nce,
        probe_fn=probe_fn,
        batch_size=B,
        cfg=cfg,
        schedule_weight=schedule_weight,
    )
    raw = _fake_criterion(feat_q, feat_k).mean()
    return projected, raw, diag


def test_eval_off_returns_raw_loss():
    # At schedule_weight=0 the projection is a no-op, so projected loss == raw loss.
    projected, raw, diag = _run_step(0.0)
    assert abs(projected.item() - raw.item()) < 1e-4


def test_active_step_runs_and_is_finite():
    projected, raw, diag = _run_step(1.0)
    assert torch.isfinite(projected)
    assert diag["probe_agreement"] >= 0.0
    assert diag["risk_mean"] >= 0.0
