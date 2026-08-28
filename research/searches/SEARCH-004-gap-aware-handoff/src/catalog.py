"""Registered full-state sources for the route-2 handoff audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HandoffCheckpoint:
    checkpoint_id: str
    family: str
    model: str
    stage: str
    step: int
    per_domain: int
    plain: Path
    method: Path
    source_mode: str = "legacy"
    initial_delta: float | None = None
    positive_source: bool = True

    def validate(self) -> None:
        for path in (self.plain, self.method):
            if not path.is_file():
                raise FileNotFoundError(path)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["plain"] = str(self.plain)
        value["method"] = str(self.method)
        return value


def audit_catalog(runs_root: Path) -> list[HandoffCheckpoint]:
    runs = Path(runs_root)
    directional = runs / "directional_search_20260826"
    search003 = runs / "evidence_guided_discovery_20260827" / "generation1_small25_seed2026"
    small = directional / "stage1_direction_screen"
    full = directional / "stage2_full_view"
    search005 = runs / "long_horizon_operator_discovery_20260827"
    search005_small = search005 / "generation1_small_2400"
    hj_handoff = runs / "dthj_rederivation_20260826" / "hj_finite_handoff"
    rows = [
        HandoffCheckpoint(
            "DT-400", "dt", "dtcov", "small25", 400, 25,
            small / "plain" / "step_400.pt",
            small / "dt_anchor" / "step_400.pt",
            initial_delta=0.1096,
        ),
        HandoffCheckpoint(
            "HJ-1200", "hj", "hj", "small25", 1200, 25,
            small / "plain" / "step_1200.pt",
            small / "hj_anchor" / "step_1200.pt",
            initial_delta=0.8045,
        ),
        HandoffCheckpoint(
            "HJ-HANDOFF-2000", "hj", "hj", "small25", 2000, 25,
            hj_handoff / "plain" / "step_2000.pt",
            hj_handoff / "hj_anchor" / "step_2000.pt",
            source_mode="native_handoff_from_hj1200",
            initial_delta=3.7900,
        ),
        HandoffCheckpoint(
            "G2HJ-FULL-800", "hj", "hj", "small25", 800, 25,
            search003 / "plain" / "step_800.pt",
            search003 / "G2-HJ-FBDFC8__full" / "step_800.pt",
            source_mode="g2_full", initial_delta=0.2244,
        ),
        HandoffCheckpoint(
            "HJPROP-1200", "hj", "hj", "small25", 1200, 25,
            search003 / "plain" / "step_1200.pt",
            search003 / "G2-HJ-FBDFC8__proposal_only" / "step_1200.pt",
            source_mode="proposal_only", initial_delta=0.6731,
        ),
        HandoffCheckpoint(
            "HJPROP-1600", "hj", "hj", "small25", 1600, 25,
            search003 / "plain" / "step_1600.pt",
            search003 / "G2-HJ-FBDFC8__proposal_only" / "step_1600.pt",
            source_mode="proposal_only", initial_delta=-0.7272, positive_source=False,
        ),
        HandoffCheckpoint(
            "HJPROP-2400", "hj", "hj", "small25", 2400, 25,
            search003 / "plain" / "step_2400.pt",
            search003 / "G2-HJ-FBDFC8__proposal_only" / "step_2400.pt",
            source_mode="proposal_only", initial_delta=1.0448,
        ),
        HandoffCheckpoint(
            "HNEK-3000", "hnek", "hnek_search", "full100", 3000, 100,
            full / "plain" / "step_3000.pt",
            full / "hnek_anchor" / "step_3000.pt",
            initial_delta=0.9490,
        ),
        HandoffCheckpoint(
            "HNEK-4000", "hnek", "hnek_search", "full100", 4000, 100,
            full / "plain" / "step_4000.pt",
            full / "hnek_anchor" / "step_4000.pt",
            initial_delta=-0.7129, positive_source=False,
        ),
        HandoffCheckpoint(
            "PCOA-1200", "pcoa", "sb", "small25", 1200, 25,
            search005_small / "plain" / "step_1200.pt",
            search005_small / "g1_game_pcoa" / "step_1200.pt",
            source_mode="search005_pcoa", initial_delta=0.1927,
        ),
        HandoffCheckpoint(
            "PCOA-1600", "pcoa", "sb", "small25", 1600, 25,
            search005_small / "plain" / "step_1600.pt",
            search005_small / "g1_game_pcoa" / "step_1600.pt",
            source_mode="search005_pcoa", initial_delta=-0.5698,
            positive_source=False,
        ),
    ]
    for row in rows:
        row.validate()
    return rows


def exact_historical_endpoint(
    checkpoint: HandoffCheckpoint, arm: str, horizon: int
) -> Path | None:
    """Return a same-lineage endpoint that already exists on disk.

    This is intentionally an explicit allow-list.  It never infers lineage
    from a matching step number or metric and therefore cannot accidentally
    reuse an endpoint produced from another parent state.
    """
    if int(horizon) != 800:
        return None
    target_step = checkpoint.step + int(horizon)
    if checkpoint.checkpoint_id == "DT-400" and arm in {
        "P_common_plain", "U_uninterrupted",
    }:
        source = checkpoint.plain if arm == "P_common_plain" else checkpoint.method
        return source.parent / f"step_{target_step}.pt"
    if checkpoint.checkpoint_id in {
        "G2HJ-FULL-800", "HJPROP-1200", "HJPROP-1600",
    } and arm in {"P_common_plain", "U_uninterrupted"}:
        source = checkpoint.plain if arm == "P_common_plain" else checkpoint.method
        return source.parent / f"step_{target_step}.pt"
    if checkpoint.checkpoint_id in {"PCOA-1200", "PCOA-1600"} and arm in {
        "P_common_plain", "U_uninterrupted",
    }:
        source = checkpoint.plain if arm == "P_common_plain" else checkpoint.method
        return source.parent / f"step_{target_step}.pt"
    return None
