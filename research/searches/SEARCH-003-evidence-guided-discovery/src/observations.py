"""Serializable target-blind observations and update geometry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch


@dataclass(frozen=True)
class StateObservation:
    source_probe: str
    source_state: str
    step: int
    domain: str | None = None
    bridge_time: int | None = None
    losses: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    paired_metrics_accessed: bool = False

    def __post_init__(self):
        if self.paired_metrics_accessed:
            raise ValueError("paired metrics are forbidden in StateObservation")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UpdateObservation:
    reference: StateObservation
    proposal: StateObservation
    global_reference_norm: float
    global_proposal_norm: float
    correction_norm: float
    reference_proposal_cosine: float
    correction_reference_cosine: float
    block_geometry: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["reference"] = self.reference.to_dict()
        value["proposal"] = self.proposal.to_dict()
        return value


def tensor_geometry(reference: torch.Tensor, proposal: torch.Tensor) -> dict[str, float]:
    """Return norms/cosines without retaining full vectors in an artifact."""
    left = reference.detach().double().reshape(-1)
    right = proposal.detach().double().reshape(-1)
    correction = right - left

    def norm(value: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(value).item())

    def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
        denominator = norm(first) * norm(second)
        if denominator == 0.0:
            return 0.0
        value = float(torch.dot(first, second).item() / denominator)
        return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0

    return {
        "reference_norm": norm(left),
        "proposal_norm": norm(right),
        "correction_norm": norm(correction),
        "reference_proposal_cosine": cosine(left, right),
        "correction_reference_cosine": cosine(correction, left),
    }


def state_dict_update_geometry(
    before: dict[str, torch.Tensor],
    reference_after: dict[str, torch.Tensor],
    proposal_after: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compare actual parameter updates, preserving top-level block identity."""
    keys = tuple(before.keys())
    if keys != tuple(reference_after.keys()) or keys != tuple(proposal_after.keys()):
        raise ValueError("state-dict identities differ")
    reference_parts = []
    proposal_parts = []
    block_reference: dict[str, list[torch.Tensor]] = {}
    block_proposal: dict[str, list[torch.Tensor]] = {}
    for key in keys:
        if not torch.is_floating_point(before[key]):
            continue
        ref_update = reference_after[key].detach().cpu() - before[key].detach().cpu()
        prop_update = proposal_after[key].detach().cpu() - before[key].detach().cpu()
        reference_parts.append(ref_update.reshape(-1))
        proposal_parts.append(prop_update.reshape(-1))
        block = key.split(".", 1)[0]
        block_reference.setdefault(block, []).append(ref_update.reshape(-1))
        block_proposal.setdefault(block, []).append(prop_update.reshape(-1))
    if not reference_parts:
        raise ValueError("no floating parameters found")
    global_geometry = tensor_geometry(
        torch.cat(reference_parts), torch.cat(proposal_parts)
    )
    blocks = {
        block: tensor_geometry(
            torch.cat(block_reference[block]), torch.cat(block_proposal[block])
        )
        for block in sorted(block_reference)
    }
    return global_geometry, blocks


def state_dict_delta_cosine(
    first_start: dict[str, torch.Tensor],
    first_end: dict[str, torch.Tensor],
    second_start: dict[str, torch.Tensor],
    second_end: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compare two state-dict displacement vectors without serializing them.

    SEARCH-003 uses this for the proposal correction at batch ``k`` versus
    the native UNSB displacement on the next independently drawn batch.  The
    four states are explicit so the helper cannot silently assume a common
    origin when the two displacements live at adjacent states.
    """
    keys = tuple(first_start)
    if any(keys != tuple(value) for value in (first_end, second_start, second_end)):
        raise ValueError("state-dict identities differ")
    first_parts = []
    second_parts = []
    for key in keys:
        if not torch.is_floating_point(first_start[key]):
            continue
        first_parts.append(
            (first_end[key].detach().cpu() - first_start[key].detach().cpu()).reshape(-1)
        )
        second_parts.append(
            (second_end[key].detach().cpu() - second_start[key].detach().cpu()).reshape(-1)
        )
    if not first_parts:
        raise ValueError("no floating parameters found")
    first = torch.cat(first_parts).double()
    second = torch.cat(second_parts).double()
    first_norm = float(torch.linalg.vector_norm(first).item())
    second_norm = float(torch.linalg.vector_norm(second).item())
    denominator = first_norm * second_norm
    cosine = 0.0 if denominator == 0.0 else float(torch.dot(first, second).item() / denominator)
    if not math.isfinite(cosine):
        cosine = 0.0
    return {
        "first_norm": first_norm,
        "second_norm": second_norm,
        "cosine": max(-1.0, min(1.0, cosine)),
    }
