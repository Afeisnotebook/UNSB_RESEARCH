"""Mathematical state transports used by SEARCH-004.

The routines in this module never receive a plain state or a paired target.
They operate on the current optimizer state and target-blind native gradients.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

from .state import optimizer_bindings


def _cosine(dot: float, left_sq: float, right_sq: float) -> float:
    denominator = math.sqrt(max(left_sq * right_sq, 0.0))
    return dot / denominator if denominator else 0.0


def least_change_native_moment_projection(
    model,
    mean_gradients: Mapping[str, torch.Tensor],
    *,
    players: tuple[str, ...] = ("G", "F"),
) -> dict:
    """Project Adam's effective first moment onto native descent half-spaces.

    For one player let ``q = m / (sqrt(v) + eps)`` be the (uncorrected)
    effective Adam direction and let ``g`` be a mean native gradient measured
    at the current state.  Native Adam moves along ``-q``.  The least Euclidean
    change that makes this a non-ascent direction is

        q* = q + max(0, -<q,g>/||g||^2) g.

    Only ``exp_avg`` changes.  Parameters, second moments, optimizer age,
    schedulers and all network co-state remain exact.  The map is identity
    whenever the inherited moment is already compatible with native descent.
    """
    bindings = optimizer_bindings(model)
    rows: dict[str, dict] = {}
    changed_any = False
    for player in players:
        if player not in bindings:
            rows[player] = {"available": False, "reason": "optimizer_unavailable"}
            continue
        optimizer, parameters = bindings[player]
        dot = q_sq = gradient_sq = 0.0
        usable: list[tuple[torch.nn.Parameter, dict, torch.Tensor, float]] = []
        for parameter_name, parameter in parameters:
            state = optimizer.state.get(parameter, {})
            moment = state.get("exp_avg")
            second = state.get("exp_avg_sq")
            gradient = mean_gradients.get(f"{player}.{parameter_name}")
            if moment is None or second is None or gradient is None:
                continue
            eps = float(optimizer.param_groups[0].get("eps", 1e-8))
            g = gradient.to(device=parameter.device, dtype=parameter.dtype)
            scale = second.detach().sqrt().add(eps)
            q = moment.detach() / scale
            dot += float((q.double() * g.double()).sum().item())
            q_sq += float(q.double().square().sum().item())
            gradient_sq += float(g.double().square().sum().item())
            usable.append((parameter, state, g, eps))
        coefficient = max(0.0, -dot / gradient_sq) if gradient_sq > 0.0 else 0.0
        before_cosine = _cosine(dot, q_sq, gradient_sq)
        correction_sq = 0.0
        if coefficient > 0.0:
            with torch.no_grad():
                for parameter, state, gradient, eps in usable:
                    second = state["exp_avg_sq"]
                    scale = second.sqrt().add(eps)
                    effective = state["exp_avg"] / scale
                    correction = gradient.mul(coefficient)
                    effective.add_(correction)
                    state["exp_avg"].copy_(effective * scale)
                    correction_sq += float(
                        correction.detach().double().square().sum().item()
                    )
            changed_any = True
        after_dot = dot + coefficient * gradient_sq
        after_sq = q_sq + 2.0 * coefficient * dot + coefficient**2 * gradient_sq
        rows[player] = {
            "available": bool(usable),
            "native_gradient_norm": math.sqrt(gradient_sq),
            "effective_moment_norm": math.sqrt(q_sq),
            "before_dot": dot,
            "before_cosine": before_cosine,
            "constraint_violation_before": max(0.0, -before_cosine),
            "projection_coefficient": coefficient,
            "correction_norm": math.sqrt(correction_sq),
            "after_dot": after_dot,
            "after_cosine": _cosine(after_dot, after_sq, gradient_sq),
            "constraint_violation_after": max(
                0.0, -_cosine(after_dot, after_sq, gradient_sq)
            ),
            "identity": coefficient == 0.0,
        }
    before = sum(float(row.get("constraint_violation_before", 0.0)) for row in rows.values())
    after = sum(float(row.get("constraint_violation_after", 0.0)) for row in rows.values())
    return {
        "schema": "clean-unsb-search004-lcnmp-v1",
        "operator": "least_change_native_moment_projection",
        "players": rows,
        "target_blind_defect_before": before,
        "target_blind_defect_after": after,
        "target_blind_defect_reduction": (
            (before - after) / before if before > 1e-12 else 0.0
        ),
        "identity": not changed_any,
        "paired_target_access": False,
        "plain_reference_access": False,
    }


def variance_carried_native_moment_rebase(
    model,
    mean_gradients: Mapping[str, torch.Tensor],
    *,
    players: tuple[str, ...] = ("G", "F"),
) -> dict:
    """Reset a conflicting first moment while retaining Adam's variance metric.

    If LCNMP removes the native-opposing component but continuation still
    fails, the remaining method-frame first moment is not a safe native
    velocity.  VCMR maps that first moment to the neutral tangent while
    preserving ``exp_avg_sq``, optimizer age, parameters and schedulers.
    Canonical beta2 dynamics then adapt the inherited trust metric smoothly.
    """
    bindings = optimizer_bindings(model)
    rows: dict[str, dict] = {}
    changed_any = False
    total_before = total_after = 0.0
    for player in players:
        if player not in bindings:
            rows[player] = {"available": False, "reason": "optimizer_unavailable"}
            continue
        optimizer, parameters = bindings[player]
        dot = q_sq = gradient_sq = first_sq = second_sq = 0.0
        usable: list[dict] = []
        for parameter_name, parameter in parameters:
            state = optimizer.state.get(parameter, {})
            moment = state.get("exp_avg")
            second = state.get("exp_avg_sq")
            gradient = mean_gradients.get(f"{player}.{parameter_name}")
            if moment is None or second is None or gradient is None:
                continue
            eps = float(optimizer.param_groups[0].get("eps", 1e-8))
            g = gradient.to(device=parameter.device, dtype=parameter.dtype)
            q = moment.detach() / (second.detach().sqrt() + eps)
            dot += float((q.double() * g.double()).sum().item())
            q_sq += float(q.double().square().sum().item())
            gradient_sq += float(g.double().square().sum().item())
            first_sq += float(moment.detach().double().square().sum().item())
            second_sq += float(second.detach().double().square().sum().item())
            usable.append(state)
        before_cosine = _cosine(dot, q_sq, gradient_sq)
        conflict = bool(usable) and dot < 0.0
        before = max(0.0, -before_cosine)
        if conflict:
            with torch.no_grad():
                for state in usable:
                    state["exp_avg"].zero_()
            changed_any = True
        after = 0.0 if conflict else before
        total_before += before
        total_after += after
        rows[player] = {
            "available": bool(usable),
            "native_gradient_norm": math.sqrt(gradient_sq),
            "effective_moment_norm_before": math.sqrt(q_sq),
            "first_moment_norm_before": math.sqrt(first_sq),
            "second_moment_norm_preserved": math.sqrt(second_sq),
            "before_dot": dot,
            "before_cosine": before_cosine,
            "constraint_violation_before": before,
            "constraint_violation_after": after,
            "first_moment_zeroed": conflict,
            "second_moment_preserved": True,
            "optimizer_age_preserved": True,
            "identity": not conflict,
        }
    return {
        "schema": "clean-unsb-search004-vcmr-v1",
        "operator": "variance_carried_native_moment_rebase",
        "players": rows,
        "target_blind_defect_before": total_before,
        "target_blind_defect_after": total_after,
        "target_blind_defect_reduction": (
            (total_before - total_after) / total_before
            if total_before > 1e-12 else 0.0
        ),
        "identity": not changed_any,
        "paired_target_access": False,
        "plain_reference_access": False,
        "parameters_changed": False,
        "second_moments_changed": False,
        "optimizer_age_changed": False,
    }
