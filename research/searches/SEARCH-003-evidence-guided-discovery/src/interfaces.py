"""Candidate-construction interfaces shared after Generation 0.

These types contain no algorithm.  They make the boundary between a method's
proposal and SEARCH-003's target-blind audit explicit, so future candidates
cannot silently read development targets or commit an unaudited update.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

import torch

from .observations import StateObservation


FORBIDDEN_KEYS = {
    "target",
    "paired_target",
    "psnr",
    "ssim",
    "lpips",
    "confirmation",
}


def assert_target_blind(fields: Mapping[str, object]) -> None:
    for key in fields:
        normalized = str(key).strip().lower()
        if normalized in FORBIDDEN_KEYS or normalized.startswith("paired_"):
            raise ValueError(f"target-aware field is forbidden: {key}")


@dataclass(frozen=True)
class CandidateUpdate:
    hypothesis_id: str
    correction: torch.Tensor
    defect_before: float
    predicted_defect_direction: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        assert_target_blind(self.diagnostics)
        if self.correction.ndim != 1:
            raise ValueError("candidate correction must be a flattened vector")


class InterventionProposal(ABC):
    """A target-blind mechanism that proposes but cannot commit an update."""

    hypothesis_id: str

    @abstractmethod
    def observe(self) -> StateObservation:
        raise NotImplementedError

    @abstractmethod
    def propose(self, reference_update: torch.Tensor) -> CandidateUpdate:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class CounterfactualBranch:
    name: str
    horizon: int
    observation: StateObservation
    full_state_digest: str
    parent_state_digest: str
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.horizon <= 0:
            raise ValueError("counterfactual horizon must be positive")
        assert_target_blind(self.diagnostics)


class CounterfactualAuditor(ABC):
    """Fork serial branches from one immutable full state.

    Concrete runners must restore networks, every optimizer/scheduler, data
    streams and all RNG/controller state before each branch.  The interface
    exposes only target-blind observations; paired labels are joined outside
    the auditor after both branches have completed.
    """

    @abstractmethod
    def fork_plain(self, full_state: Mapping[str, object], horizon: int) -> CounterfactualBranch:
        raise NotImplementedError

    @abstractmethod
    def fork_proposal(
        self,
        full_state: Mapping[str, object],
        proposal: InterventionProposal,
        horizon: int,
    ) -> CounterfactualBranch:
        raise NotImplementedError

    @abstractmethod
    def assert_parent_unchanged(self, full_state: Mapping[str, object]) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ProjectionResult:
    correction: torch.Tensor
    accepted_fraction: float
    feasible: bool
    active_constraints: tuple[int, ...]


class ConstraintProjector:
    """Project a correction onto target-blind audit halfspaces and a trust ball.

    Each audit row ``a_j`` represents a first-order held-out native-loss
    direction and enforces ``a_j^T c <= tolerance_j``.  Sequential orthogonal
    projection is deterministic and sufficient for the small (1-3 constraint)
    systems allowed by SEARCH-003.  Failure returns the exact zero correction.
    """

    def __init__(self, *, passes: int = 8, feasibility_epsilon: float = 1e-8):
        if passes <= 0:
            raise ValueError("passes must be positive")
        self.passes = int(passes)
        self.feasibility_epsilon = float(feasibility_epsilon)

    def project(
        self,
        correction: torch.Tensor,
        audit_directions: torch.Tensor,
        tolerances: torch.Tensor,
        *,
        radius: float,
    ) -> ProjectionResult:
        if correction.ndim != 1:
            raise ValueError("correction must be one-dimensional")
        if audit_directions.ndim != 2 or audit_directions.shape[1] != correction.numel():
            raise ValueError("audit direction shape mismatch")
        if tolerances.shape != (audit_directions.shape[0],):
            raise ValueError("tolerance shape mismatch")
        if radius < 0:
            raise ValueError("radius must be nonnegative")
        original = correction.detach()
        value = original.clone()
        active: set[int] = set()
        for _ in range(self.passes):
            for index, (direction, tolerance) in enumerate(
                zip(audit_directions, tolerances)
            ):
                violation = torch.dot(direction, value) - tolerance
                if float(violation.item()) > self.feasibility_epsilon:
                    norm_sq = torch.dot(direction, direction)
                    if float(norm_sq.item()) <= self.feasibility_epsilon:
                        return ProjectionResult(
                            correction=torch.zeros_like(value),
                            accepted_fraction=0.0,
                            feasible=False,
                            active_constraints=tuple(sorted(active | {index})),
                        )
                    value = value - violation / norm_sq * direction
                    active.add(index)
            value_norm = torch.linalg.vector_norm(value)
            if float(value_norm.item()) > radius:
                value = value * (float(radius) / max(float(value_norm.item()), 1e-20))
        violations = audit_directions @ value - tolerances
        feasible = bool((violations <= self.feasibility_epsilon).all().item())
        if not feasible:
            value = torch.zeros_like(value)
        original_norm = float(torch.linalg.vector_norm(original).item())
        accepted = (
            float(torch.linalg.vector_norm(value).item()) / original_norm
            if original_norm > 0 else 0.0
        )
        return ProjectionResult(
            correction=value,
            accepted_fraction=accepted,
            feasible=feasible,
            active_constraints=tuple(sorted(active)),
        )
