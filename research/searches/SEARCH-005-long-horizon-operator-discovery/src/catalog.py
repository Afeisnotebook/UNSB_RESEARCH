"""Minimal positive/reversal checkpoint pairs for causal audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditCheckpoint:
    checkpoint_id: str
    probe: str
    model: str
    stage: str
    step: int
    per_domain: int
    historical_delta: float
    plain: Path
    method: Path

    def validate(self) -> None:
        if self.probe not in {"dt", "hj", "hnek"}:
            raise ValueError(self.probe)
        for path in (self.plain, self.method):
            if not path.is_file():
                raise FileNotFoundError(path)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["plain"] = str(self.plain)
        value["method"] = str(self.method)
        return value


def causal_catalog(runs_root: Path) -> list[AuditCheckpoint]:
    root = Path(runs_root) / "directional_search_20260826"
    small = root / "stage1_direction_screen"
    full = root / "stage2_full_view"
    rows = [
        AuditCheckpoint(
            "DT-FULL-2000-POS", "dt", "dtcov", "full100", 2000, 100,
            0.566439, full / "plain" / "step_2000.pt",
            full / "dt_anchor" / "step_2000.pt",
        ),
        AuditCheckpoint(
            "DT-FULL-3000-NEG", "dt", "dtcov", "full100", 3000, 100,
            -0.806110, full / "plain" / "step_3000.pt",
            full / "dt_anchor" / "step_3000.pt",
        ),
        AuditCheckpoint(
            "DT-FULL-4000-NEG", "dt", "dtcov", "full100", 4000, 100,
            -1.519180, full / "plain" / "step_4000.pt",
            full / "dt_anchor" / "step_4000.pt",
        ),
        AuditCheckpoint(
            "HJ-SMALL-400-NEAR", "hj", "hj", "small25", 400, 25,
            0.046723, small / "plain" / "step_400.pt",
            small / "hj_anchor" / "step_400.pt",
        ),
        AuditCheckpoint(
            "HJ-SMALL-800-NEG", "hj", "hj", "small25", 800, 25,
            -0.724102, small / "plain" / "step_800.pt",
            small / "hj_anchor" / "step_800.pt",
        ),
        AuditCheckpoint(
            "HJ-SMALL-1200-POS", "hj", "hj", "small25", 1200, 25,
            0.804544, small / "plain" / "step_1200.pt",
            small / "hj_anchor" / "step_1200.pt",
        ),
        AuditCheckpoint(
            "HNEK-FULL-3000-POS", "hnek", "hnek_search", "full100", 3000, 100,
            0.953997, full / "plain" / "step_3000.pt",
            full / "hnek_anchor" / "step_3000.pt",
        ),
        AuditCheckpoint(
            "HNEK-FULL-4000-NEG", "hnek", "hnek_search", "full100", 4000, 100,
            -0.712401, full / "plain" / "step_4000.pt",
            full / "hnek_anchor" / "step_4000.pt",
        ),
    ]
    for row in rows:
        row.validate()
    return rows
