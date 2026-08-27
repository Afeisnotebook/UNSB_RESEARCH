"""Pure mathematical operators for SEARCH-005 Generation 1."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BrownianProjectionDiagnostics:
    active_fraction: float
    mean_scale: float
    minimum_scale: float
    maximum_scale: float
    mean_variance: float
    mean_cap: float
    maximum_projected_ratio: float


@dataclass(frozen=True)
class PathwiseResidualDiagnostics:
    active_fraction: float
    mean_scale: float
    minimum_scale: float
    maximum_scale: float
    mean_raw_energy: float
    mean_accepted_energy: float
    mean_physical_cap: float
    maximum_cap_ratio: float


@dataclass(frozen=True)
class RateSafeProjectionDiagnostics:
    active: bool
    native_defect_alignment: float
    projected_defect_alignment: float
    native_descent_alignment: float
    projection_coefficient: float


@dataclass(frozen=True)
class OptimisticDisplacementDiagnostics:
    predictability: float
    current_norm: float
    previous_norm: float
    innovation_norm: float
    correction_norm: float


@dataclass(frozen=True)
class NormPreservingOptimisticDiagnostics:
    predictability: float
    current_norm: float
    previous_norm: float
    innovation_norm: float
    orthogonal_innovation_norm: float
    correction_norm: float
    applied_norm: float
    norm_ratio: float


def predictability_gated_optimistic_displacement(
    current_native_displacement: torch.Tensor,
    previous_native_displacement: torch.Tensor | None,
) -> tuple[torch.Tensor, OptimisticDisplacementDiagnostics]:
    """Return a target-blind optimistic displacement with regression gating.

    The previous native Adam displacement predicts the current one with the
    clipped least-squares coefficient ``rho``.  The returned displacement is
    ``u_t + rho*(u_t-u_{t-1})``.  It is native on the first step, under a
    non-predictive/anti-aligned field, and whenever the field is unchanged.
    """
    current = current_native_displacement
    if previous_native_displacement is None:
        zero = current.new_zeros(())
        return current, OptimisticDisplacementDiagnostics(
            predictability=0.0,
            current_norm=float(current.detach().norm().item()),
            previous_norm=0.0,
            innovation_norm=0.0,
            correction_norm=0.0,
        )
    previous = previous_native_displacement
    if current.shape != previous.shape:
        raise ValueError("optimistic displacement shapes differ")
    denominator = previous.square().sum()
    if denominator <= 0:
        rho = current.new_zeros(())
    else:
        rho = torch.clamp((current * previous).sum() / denominator, 0.0, 1.0)
    innovation = current - previous
    correction = rho * innovation
    result = current + correction
    return result, OptimisticDisplacementDiagnostics(
        predictability=float(rho.detach().item()),
        current_norm=float(current.detach().norm().item()),
        previous_norm=float(previous.detach().norm().item()),
        innovation_norm=float(innovation.detach().norm().item()),
        correction_norm=float(correction.detach().norm().item()),
    )


def norm_preserving_orthogonal_optimistic_displacement(
    current_native_displacement: torch.Tensor,
    previous_native_displacement: torch.Tensor | None,
) -> tuple[torch.Tensor, NormPreservingOptimisticDiagnostics]:
    """Rotate a native Adam displacement optimistically without changing its norm.

    PCOA's full innovation mixes game-field rotation with changes in update
    magnitude.  This revision removes the innovation component parallel to the
    current native displacement, adds only the orthogonal (phase) component,
    then projects the proposal back to the native Euclidean sphere.  It is
    target blind, preserves every native step norm and is exact native on the
    first step, under non-predictive fields, and under collinear fields.
    """
    current = current_native_displacement
    current_sq = current.square().sum()
    current_norm = current_sq.clamp_min(0).sqrt()
    if previous_native_displacement is None:
        zero = current.new_zeros(())
        return current, NormPreservingOptimisticDiagnostics(
            predictability=0.0,
            current_norm=float(current_norm.detach().item()),
            previous_norm=0.0,
            innovation_norm=0.0,
            orthogonal_innovation_norm=0.0,
            correction_norm=0.0,
            applied_norm=float(current_norm.detach().item()),
            norm_ratio=1.0,
        )
    previous = previous_native_displacement
    if current.shape != previous.shape:
        raise ValueError("norm-preserving optimistic displacement shapes differ")
    previous_sq = previous.square().sum()
    if previous_sq <= 0 or current_sq <= 0:
        rho = current.new_zeros(())
    else:
        rho = torch.clamp((current * previous).sum() / previous_sq, 0.0, 1.0)
    innovation = current - previous
    if current_sq <= 0 or rho <= 0:
        orthogonal = torch.zeros_like(current)
        result = current
    else:
        parallel_coefficient = (innovation * current).sum() / current_sq
        orthogonal = innovation - parallel_coefficient * current
        proposal = current + rho * orthogonal
        proposal_norm = proposal.square().sum().clamp_min(0).sqrt()
        result = (
            current
            if proposal_norm <= 0
            else proposal * (current_norm / proposal_norm)
        )
    correction = result - current
    applied_norm = result.square().sum().clamp_min(0).sqrt()
    ratio = (
        current.new_ones(())
        if current_norm <= 0
        else applied_norm / current_norm
    )
    return result, NormPreservingOptimisticDiagnostics(
        predictability=float(rho.detach().item()),
        current_norm=float(current_norm.detach().item()),
        previous_norm=float(previous_sq.detach().clamp_min(0).sqrt().item()),
        innovation_norm=float(innovation.detach().norm().item()),
        orthogonal_innovation_norm=float(orthogonal.detach().norm().item()),
        correction_norm=float(correction.detach().norm().item()),
        applied_norm=float(applied_norm.detach().item()),
        norm_ratio=float(ratio.detach().item()),
    )


def rate_safe_native_gradient(
    native_gradient: torch.Tensor,
    defect_gradient: torch.Tensor,
    adam_metric: torch.Tensor,
) -> tuple[torch.Tensor, RateSafeProjectionDiagnostics]:
    """Project a native gradient so its Adam step cannot increase a defect.

    The optimizer step is ``-M g``.  First-order defect safety is therefore
    ``<a,g>_M >= 0`` for defect gradient ``a``.  This is the exact metric
    projection of ``g`` onto that halfspace and is identity when already safe.
    """
    if not (
        native_gradient.shape == defect_gradient.shape == adam_metric.shape
    ):
        raise ValueError("rate-safe projection tensor shapes differ")
    if bool((adam_metric <= 0).any()) or not bool(torch.isfinite(adam_metric).all()):
        raise ValueError("Adam metric must be finite and strictly positive")
    alignment = metric_dot(defect_gradient, native_gradient, adam_metric)
    defect_norm = metric_dot(defect_gradient, defect_gradient, adam_metric)
    active = bool(alignment < 0 and defect_norm > 0)
    if active:
        coefficient = alignment / defect_norm
        projected = native_gradient - coefficient * defect_gradient
    else:
        coefficient = torch.zeros_like(alignment)
        projected = native_gradient
    projected_alignment = metric_dot(defect_gradient, projected, adam_metric)
    descent_alignment = metric_dot(native_gradient, projected, adam_metric)
    return projected, RateSafeProjectionDiagnostics(
        active=active,
        native_defect_alignment=float(alignment.detach().item()),
        projected_defect_alignment=float(projected_alignment.detach().item()),
        native_descent_alignment=float(descent_alignment.detach().item()),
        projection_coefficient=float(coefficient.detach().item()),
    )


def pathwise_horizon_residual_projection(
    state: torch.Tensor,
    native_endpoint: torch.Tensor,
    horizon: torch.Tensor,
    previous_horizon: torch.Tensor | None,
    previous_accepted_energy: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, PathwiseResidualDiagnostics]:
    """Project endpoint residual energy onto a pathwise physical horizon cone.

    Starting with the exact native proposal at the first bridge point, the
    next accepted energy obeys ``q_i <= q_{i-1} h_i/h_{i-1}``.  A native
    proposal already inside that ball is returned byte-for-byte.
    """
    if state.shape != native_endpoint.shape or state.ndim < 2:
        raise ValueError("state and endpoint shapes must match and include a batch")
    batch = state.shape[0]
    h = torch.as_tensor(horizon, device=state.device, dtype=state.dtype).reshape(-1)
    if h.numel() == 1:
        h = h.expand(batch)
    if h.numel() != batch:
        raise ValueError("horizon must be scalar or per-sample")
    if not bool(torch.isfinite(h).all()) or bool(((h <= 0) | (h > 1)).any()):
        raise ValueError("generator-call horizon must lie in (0, 1]")
    reduction_dims = tuple(range(1, state.ndim))
    residual = native_endpoint - state
    raw_energy = residual.square().mean(dim=reduction_dims)

    if previous_horizon is None or previous_accepted_energy is None:
        scale = torch.ones_like(raw_energy)
        accepted_energy = raw_energy
        cap = raw_energy
        active = torch.zeros_like(raw_energy, dtype=torch.bool)
    else:
        previous_h = torch.as_tensor(
            previous_horizon, device=state.device, dtype=state.dtype
        ).reshape(-1)
        previous_q = torch.as_tensor(
            previous_accepted_energy, device=state.device, dtype=state.dtype
        ).reshape(-1)
        if previous_h.numel() == 1:
            previous_h = previous_h.expand(batch)
        if previous_q.numel() == 1:
            previous_q = previous_q.expand(batch)
        if previous_h.numel() != batch or previous_q.numel() != batch:
            raise ValueError("previous horizon and energy must be scalar or per-sample")
        if bool((h >= previous_h).any()):
            raise ValueError("pathwise horizons must strictly decrease")
        if bool((previous_q < 0).any()) or not bool(torch.isfinite(previous_q).all()):
            raise ValueError("previous accepted energy must be finite and nonnegative")
        cap = previous_q * (h / previous_h)
        tiny = torch.finfo(state.dtype).tiny
        scale = torch.minimum(
            torch.ones_like(raw_energy),
            torch.sqrt(cap / raw_energy.clamp_min(tiny)),
        )
        active = raw_energy > cap
        accepted_energy = raw_energy * scale.square()

    scale_view = scale.reshape((batch,) + (1,) * (state.ndim - 1))
    projected = state + scale_view * residual
    active_view = active.reshape((batch,) + (1,) * (state.ndim - 1))
    result = torch.where(active_view, projected, native_endpoint)
    tiny = torch.finfo(state.dtype).tiny
    ratios = torch.where(
        cap > 0,
        accepted_energy / cap.clamp_min(tiny),
        torch.where(accepted_energy == 0, torch.zeros_like(cap), torch.full_like(cap, float("inf"))),
    )
    return result, accepted_energy.detach(), PathwiseResidualDiagnostics(
        active_fraction=float(active.float().mean().detach().item()),
        mean_scale=float(scale.mean().detach().item()),
        minimum_scale=float(scale.min().detach().item()),
        maximum_scale=float(scale.max().detach().item()),
        mean_raw_energy=float(raw_energy.mean().detach().item()),
        mean_accepted_energy=float(accepted_energy.mean().detach().item()),
        mean_physical_cap=float(cap.mean().detach().item()),
        maximum_cap_ratio=float(ratios.max().detach().item()),
    )


def brownian_antithetic_variance_projection(
    positive_endpoint: torch.Tensor,
    negative_endpoint: torch.Tensor,
    horizon: torch.Tensor,
    *,
    tau: float,
) -> tuple[torch.Tensor, BrownianProjectionDiagnostics]:
    """Project excessive odd-latent endpoint variance onto ``tau*h``.

    For ``M=(P(z)+P(-z))/2`` and ``D=(P(z)-P(-z))/2``, return ``M+aD``
    with ``a=min(1, sqrt(tau*h/E[D^2]))`` per sample.  The antithetic mean is
    unchanged and a feasible native endpoint is returned byte-for-byte.
    """
    if positive_endpoint.shape != negative_endpoint.shape:
        raise ValueError("antithetic endpoints must have identical shapes")
    if positive_endpoint.ndim < 2:
        raise ValueError("endpoints must include batch and feature dimensions")
    if not positive_endpoint.dtype.is_floating_point:
        raise ValueError("endpoints must be floating point")
    if not (float(tau) > 0.0):
        raise ValueError("tau must be positive")

    batch = positive_endpoint.shape[0]
    h = torch.as_tensor(
        horizon,
        device=positive_endpoint.device,
        dtype=positive_endpoint.dtype,
    ).reshape(-1)
    if h.numel() == 1:
        h = h.expand(batch)
    if h.numel() != batch:
        raise ValueError("horizon must be scalar or have one value per sample")
    if not bool(torch.isfinite(h).all()) or bool(((h < 0) | (h > 1)).any()):
        raise ValueError("horizon must be finite and lie in [0, 1]")

    reduction_dims = tuple(range(1, positive_endpoint.ndim))
    midpoint = 0.5 * (positive_endpoint + negative_endpoint)
    half_difference = 0.5 * (positive_endpoint - negative_endpoint)
    variance = half_difference.square().mean(dim=reduction_dims)
    cap = float(tau) * h
    tiny = torch.finfo(positive_endpoint.dtype).tiny
    scale = torch.minimum(
        torch.ones_like(variance),
        torch.sqrt(cap / variance.clamp_min(tiny)),
    )
    scale_view = scale.reshape((batch,) + (1,) * (positive_endpoint.ndim - 1))
    projected = midpoint + scale_view * half_difference
    active = variance > cap
    active_view = active.reshape((batch,) + (1,) * (positive_endpoint.ndim - 1))
    # The explicit branch is the operator's exact self-null condition.
    result = torch.where(active_view, projected, positive_endpoint)

    projected_variance = variance * scale.square()
    positive_cap = cap > 0
    ratios = torch.where(
        positive_cap,
        projected_variance / cap.clamp_min(tiny),
        torch.where(
            projected_variance == 0,
            torch.zeros_like(cap),
            torch.full_like(cap, float("inf")),
        ),
    )
    return result, BrownianProjectionDiagnostics(
        active_fraction=float(active.float().mean().detach().item()),
        mean_scale=float(scale.mean().detach().item()),
        minimum_scale=float(scale.min().detach().item()),
        maximum_scale=float(scale.max().detach().item()),
        mean_variance=float(variance.mean().detach().item()),
        mean_cap=float(cap.mean().detach().item()),
        maximum_projected_ratio=float(ratios.max().detach().item()),
    )


def cndrp_precondition(
    native_gradient: torch.Tensor,
    sensitivity_first: torch.Tensor,
    sensitivity_second: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the confidence-normalized diagonal DT metric.

    Each coordinate uses ``0.5*(a1-a2)^2`` as its independent-replicate
    variation estimate.  The diagonal form commutes with every positive
    diagonal optimizer metric, including frozen Adam geometry.
    """
    if not (
        native_gradient.shape == sensitivity_first.shape == sensitivity_second.shape
    ):
        raise ValueError("gradient and sensitivity shapes differ")
    mean = 0.5 * (sensitivity_first + sensitivity_second)
    variation = 0.5 * (sensitivity_first - sensitivity_second) ** 2
    mean_sq = mean ** 2
    # Keep the strict-positive term representable in float32.  Without this
    # floor, the implemented map can silently become a hard projector.
    dtype_floor = torch.finfo(native_gradient.dtype).eps * 8.0
    effective_eps = torch.maximum(
        torch.full_like(mean_sq, float(eps)),
        (mean_sq + variation).detach().clamp_min(1.0) * dtype_floor,
    )
    scale = (variation + effective_eps) / (mean_sq + variation + effective_eps)
    result = scale * native_gradient
    return result, {
        "mean_sensitivity_norm": float(mean_sq.detach().sum().sqrt().item()),
        "variation_trace": float(variation.detach().sum().item()),
        "minimum_scale": float(scale.detach().min().item()),
        "mean_scale": float(scale.detach().mean().item()),
        "maximum_scale": float(scale.detach().max().item()),
    }


def block_confidence_precondition(
    native_gradient: torch.Tensor,
    sensitivity_first: torch.Tensor,
    sensitivity_second: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use one confidence scalar for an entire parameter tensor block."""
    if not (
        native_gradient.shape == sensitivity_first.shape == sensitivity_second.shape
    ):
        raise ValueError("gradient and sensitivity shapes differ")
    mean = 0.5 * (sensitivity_first + sensitivity_second)
    variation = 0.5 * (sensitivity_first - sensitivity_second).square().sum()
    signal = mean.square().sum()
    dtype_floor = torch.finfo(native_gradient.dtype).eps * 8.0
    effective_eps = torch.maximum(
        signal.new_tensor(float(eps)),
        (signal + variation).detach().clamp_min(1.0) * dtype_floor,
    )
    scale = (variation + effective_eps) / (signal + variation + effective_eps)
    result = scale * native_gradient
    scalar = float(scale.detach().item())
    return result, {
        "mean_sensitivity_norm": float(signal.detach().sqrt().item()),
        "variation_trace": float(variation.detach().item()),
        "minimum_scale": scalar,
        "mean_scale": scalar,
        "maximum_scale": scalar,
    }


def metric_dot(first: torch.Tensor, second: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    if not (first.shape == second.shape == metric.shape):
        raise ValueError("metric dot shapes differ")
    if torch.any(metric <= 0):
        raise ValueError("metric must be strictly positive")
    return torch.sum(first * metric * second)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    raw_norm: float
    projected_norm: float
    native_alignment: float
    bridge_alignment: float
    trust_scale: float
    active_constraints: tuple[str, ...]


def _single_halfspace(
    correction: torch.Tensor,
    normal: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    value = metric_dot(normal, correction, metric)
    norm_sq = metric_dot(normal, normal, metric)
    if value >= 0 or norm_sq <= 0:
        return correction
    return correction - value / norm_sq * normal


def acmp_project(
    raw_correction: torch.Tensor,
    native_gradient: torch.Tensor,
    bridge_adversarial_gradient: torch.Tensor,
    adam_metric: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, ProjectionDiagnostics]:
    """Exact two-halfspace projection followed by a native-scale trust bound.

    The metric is the positive Adam gradient-to-update diagonal.  Feasibility
    means both correction contributions are descent directions after applying
    that same preconditioner.
    """
    tensors = (
        raw_correction,
        native_gradient,
        bridge_adversarial_gradient,
        adam_metric,
    )
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("ACMP tensor shapes differ")
    if torch.any(adam_metric <= 0):
        raise ValueError("Adam metric must be strictly positive")

    normals = (native_gradient, bridge_adversarial_gradient)
    labels = ("native", "bridge_adversarial")
    b = torch.stack([metric_dot(normal, raw_correction, adam_metric) for normal in normals])
    gram = torch.stack([
        torch.stack([metric_dot(left, right, adam_metric) for right in normals])
        for left in normals
    ])
    projected = raw_correction
    active: tuple[str, ...] = ()
    if not bool(torch.all(b >= 0)):
        candidates: list[tuple[torch.Tensor, tuple[str, ...], torch.Tensor]] = []
        for index in range(2):
            if gram[index, index] <= eps:
                continue
            multiplier = -b[index] / gram[index, index]
            if multiplier < 0:
                continue
            value = raw_correction + multiplier * normals[index]
            feasibility = torch.stack([
                metric_dot(normal, value, adam_metric) for normal in normals
            ])
            if bool(torch.all(feasibility >= -1e-7)):
                distance = metric_dot(value - raw_correction, value - raw_correction, adam_metric)
                candidates.append((value, (labels[index],), distance))
        determinant = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
        if abs(float(determinant.detach().item())) > eps:
            multipliers = torch.linalg.solve(gram, -b)
            if bool(torch.all(multipliers >= -1e-7)):
                value = raw_correction + multipliers[0] * normals[0] + multipliers[1] * normals[1]
                feasibility = torch.stack([
                    metric_dot(normal, value, adam_metric) for normal in normals
                ])
                if bool(torch.all(feasibility >= -1e-7)):
                    distance = metric_dot(value - raw_correction, value - raw_correction, adam_metric)
                    candidates.append((value, labels, distance))
        if candidates:
            projected, active, _ = min(candidates, key=lambda item: float(item[2].detach().item()))
        else:
            # Degenerate parallel normals: alternating exact halfspace
            # projections converges in one pass when the normals agree and
            # remains safe after a few passes in the nearly parallel case.
            projected = raw_correction
            active_list = []
            for _ in range(8):
                for label, normal in zip(labels, normals):
                    before = projected
                    projected = _single_halfspace(projected, normal, adam_metric)
                    if not torch.equal(before, projected) and label not in active_list:
                        active_list.append(label)
            active = tuple(active_list)

    raw_norm = metric_dot(raw_correction, raw_correction, adam_metric).clamp_min(0).sqrt()
    projected_norm = metric_dot(projected, projected, adam_metric).clamp_min(0).sqrt()
    native_norm = metric_dot(native_gradient, native_gradient, adam_metric).clamp_min(0).sqrt()
    scale = torch.minimum(
        torch.ones_like(projected_norm),
        native_norm / projected_norm.clamp_min(float(eps)),
    )
    projected = projected * scale
    final_norm = metric_dot(projected, projected, adam_metric).clamp_min(0).sqrt()
    return projected, ProjectionDiagnostics(
        raw_norm=float(raw_norm.detach().item()),
        projected_norm=float(final_norm.detach().item()),
        native_alignment=float(metric_dot(native_gradient, projected, adam_metric).detach().item()),
        bridge_alignment=float(metric_dot(bridge_adversarial_gradient, projected, adam_metric).detach().item()),
        trust_scale=float(scale.detach().item()),
        active_constraints=active,
    )
