"""Preserved checkpoint identities used as Generation-0 evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatchedCheckpoint:
    probe: str
    stage: str
    step: int
    per_domain: int
    plain: Path
    method: Path
    decisive: bool = False

    def validate(self) -> None:
        if not self.plain.is_file():
            raise FileNotFoundError(self.plain)
        if not self.method.is_file():
            raise FileNotFoundError(self.method)


def preserved_catalog(runs_root: Path) -> list[MatchedCheckpoint]:
    directional = Path(runs_root) / "directional_search_20260826"
    dthj = Path(runs_root) / "dthj_rederivation_20260826"
    rows: list[MatchedCheckpoint] = []

    small = directional / "stage1_direction_screen"
    lane_names = {
        "dt": "dt_anchor",
        "hj": "hj_anchor",
        "hnek": "hnek_anchor",
        "lbst": "lbst",
        "ptq": "ptq",
        "dcum": "dcum",
        "aeb": "aeb",
    }
    for probe, lane in lane_names.items():
        for step in (400, 800, 1200):
            rows.append(MatchedCheckpoint(
                probe=probe,
                stage="small",
                step=step,
                per_domain=25,
                plain=small / "plain" / f"step_{step}.pt",
                method=small / lane / f"step_{step}.pt",
                decisive=(probe in {"dt", "hj", "hnek"}),
            ))

    lttr = dthj / "screen"
    for lane in ("lttr_tangent", "lttr_direction"):
        for step in (400, 800):
            rows.append(MatchedCheckpoint(
                probe=lane,
                stage="small",
                step=step,
                per_domain=25,
                plain=lttr / "plain" / f"step_{step}.pt",
                method=lttr / lane / f"step_{step}.pt",
            ))

    full = directional / "stage2_full_view"
    for probe, lane in (("dt", "dt_anchor"), ("hnek", "hnek_anchor")):
        for step in (1000, 2000, 3000, 4000):
            rows.append(MatchedCheckpoint(
                probe=probe,
                stage="full",
                step=step,
                per_domain=100,
                plain=full / "plain" / f"step_{step}.pt",
                method=full / lane / f"step_{step}.pt",
                decisive=True,
            ))

    extension = directional / "stage3_extension"
    for step in (6000, 8000, 10000, 12000):
        rows.append(MatchedCheckpoint(
            probe="hnek",
            stage="extension",
            step=step,
            per_domain=100,
            plain=extension / "plain" / f"step_{step}.pt",
            method=extension / "hnek_anchor" / f"step_{step}.pt",
            decisive=True,
        ))

    handoff = dthj / "hj_finite_handoff"
    for step in (1600, 2000):
        rows.append(MatchedCheckpoint(
            probe="hj_handoff",
            stage="small_handoff",
            step=step,
            per_domain=25,
            plain=handoff / "plain" / f"step_{step}.pt",
            method=handoff / "hj_anchor" / f"step_{step}.pt",
            decisive=True,
        ))

    for row in rows:
        row.validate()
    return rows
