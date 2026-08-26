"""Deterministic DT/HJ re-derivation in endpoint response space.

LTTR replaces noisy batch/domain covariance calibration with a paired latent
tangent measured inside each image.  A frozen first-use generator defines the
local response chart.  The tangent lane preserves only the relative latent
tangent energy.  The safe lane additionally penalizes one-sided reversals of
the mean endpoint direction in structurally uncertain regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LTTRConfig:
    region_patch: int = 32
    direction_margin: float = 0.5
    direction_weight: float = 0.25
    eps: float = 1e-6


@dataclass
class LTTRStatistics:
    mean_residual: torch.Tensor
    tangent: torch.Tensor
    mean_energy: torch.Tensor
    tangent_energy: torch.Tensor
    log_tangent_ratio: torch.Tensor


def _region_pool(x: torch.Tensor, patch: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("region pooling expects BCHW")
    patch = max(1, int(patch))
    height, width = x.shape[-2:]
    if height % patch or width % patch:
        raise ValueError("LTTR region_patch must divide the spatial dimensions")
    return F.avg_pool2d(x, kernel_size=patch, stride=patch)


def response_statistics(
    *,
    x_t: torch.Tensor,
    endpoint_plus: torch.Tensor,
    endpoint_minus: torch.Tensor,
    region_patch: int,
    eps: float = 1e-6,
) -> LTTRStatistics:
    """Return an antithetic first-order chart of the endpoint response."""
    if endpoint_plus.shape != x_t.shape or endpoint_minus.shape != x_t.shape:
        raise ValueError("x_t and both endpoint tensors must have identical BCHW shapes")
    mean_residual = 0.5 * (endpoint_plus + endpoint_minus) - x_t
    tangent = 0.5 * (endpoint_plus - endpoint_minus)
    mean_energy = _region_pool(mean_residual.square().mean(dim=1, keepdim=True), region_patch)
    tangent_energy = _region_pool(tangent.square().mean(dim=1, keepdim=True), region_patch)
    log_ratio = torch.log(tangent_energy + eps) - torch.log(mean_energy + eps)
    return LTTRStatistics(
        mean_residual=mean_residual,
        tangent=tangent,
        mean_energy=mean_energy,
        tangent_energy=tangent_energy,
        log_tangent_ratio=log_ratio,
    )


def _per_image_median(x: torch.Tensor) -> torch.Tensor:
    values = x.flatten(1).sort(dim=1).values
    return values[:, (values.shape[1] - 1) // 2].reshape(-1, 1, 1, 1)


def _teacher_scale(log_ratio: torch.Tensor, eps: float) -> torch.Tensor:
    median = _per_image_median(log_ratio)
    mad = _per_image_median((log_ratio - median).abs())
    # A finite floor makes the chart well-defined even when all regions have
    # the same teacher response.  Unlike the old z-score clip, it never makes
    # current and teacher identically saturated.
    return (1.4826 * mad).clamp_min(0.25 + eps)


def _structure_weight(source: torch.Tensor, target_shape: tuple[int, int], eps: float) -> torch.Tensor:
    gray = source.mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = F.adaptive_avg_pool2d(dx + dy, target_shape)
    return (edge / edge.mean(dim=(2, 3), keepdim=True).clamp_min(eps)).clamp(0.0, 4.0)


def _risk_weight(
    teacher: LTTRStatistics,
    source: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    q = teacher.log_tangent_ratio.detach()
    median = _per_image_median(q)
    scale = _teacher_scale(q, eps)
    latent_risk = torch.sigmoid((q - median) / scale)
    structure = _structure_weight(source.detach(), q.shape[-2:], eps)
    weight = 1.0 + latent_risk * structure
    return (weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(eps)).detach()


def lttr_loss(
    *,
    current: LTTRStatistics,
    teacher: LTTRStatistics,
    source: torch.Tensor,
    mode: str,
    config: LTTRConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute scale-safe tangent preservation and optional reversal barrier."""
    mode = str(mode).lower()
    if mode not in {"tangent", "safe"}:
        raise ValueError(f"unknown LTTR mode: {mode}")
    eps = float(config.eps)
    weight = _risk_weight(teacher, source, eps)
    scale = _teacher_scale(teacher.log_tangent_ratio.detach(), eps)
    normalized_delta = (
        current.log_tangent_ratio - teacher.log_tangent_ratio.detach()
    ) / scale
    tangent_map = F.smooth_l1_loss(
        normalized_delta, torch.zeros_like(normalized_delta), reduction="none"
    )
    tangent_loss = (weight * tangent_map).mean()

    direction_loss = tangent_loss.new_zeros(())
    cosine_mean = tangent_loss.new_ones(())
    reversal_rate = tangent_loss.new_zeros(())
    if mode == "safe":
        patch = int(config.region_patch)
        current_mean = _region_pool(current.mean_residual, patch)
        teacher_mean = _region_pool(teacher.mean_residual.detach(), patch)
        dot = (current_mean * teacher_mean).sum(dim=1, keepdim=True)
        norm = current_mean.square().sum(dim=1, keepdim=True).sqrt()
        norm = norm * teacher_mean.square().sum(dim=1, keepdim=True).sqrt()
        cosine = dot / norm.clamp_min(eps)
        reliable = (teacher.mean_energy.detach() > eps).to(cosine.dtype)
        barrier = F.relu(float(config.direction_margin) - cosine).square()
        direction_loss = (weight * reliable * barrier).sum() / (
            (weight * reliable).sum().clamp_min(eps)
        )
        cosine_mean = (cosine * reliable).sum() / reliable.sum().clamp_min(1.0)
        reversal_rate = ((cosine < 0.0).to(cosine.dtype) * reliable).sum() / (
            reliable.sum().clamp_min(1.0)
        )

    total = tangent_loss + float(config.direction_weight) * direction_loss
    diagnostics = {
        "total": total.detach(),
        "tangent": tangent_loss.detach(),
        "direction": direction_loss.detach(),
        "cosine_mean": cosine_mean.detach(),
        "reversal_rate": reversal_rate.detach(),
        "risk_weight_max": weight.max().detach(),
    }
    return total, diagnostics
