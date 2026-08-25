"""Target-access guard and append-only access ledger.

Before ``TRAINING_FROZEN.ok`` exists, training/diagnostics may only read the
unpaired T2 A/B images.  Any same-stem target, paired-development target, or
official test/confirmation path must be actively rejected, and every file read
is recorded in an append-only ledger.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AccessRecord:
    path: str
    role: str
    purpose: str
    resolved: str
    allowed: bool

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "role": self.role,
            "purpose": self.purpose,
            "resolved": self.resolved,
            "allowed": self.allowed,
        }


class TargetAccessGuard:
    """Guard reads so paired targets stay sealed until training is frozen."""

    def __init__(
        self,
        *,
        training_manifest: list[dict],
        paired_manifest: list[dict],
        ledger_path: Path,
        data_root: str,
        frozen_ok_path: Path,
        run_id: str = "",
        spec_sha256: str = "",
        code_sha256: str = "",
    ):
        self.training_manifest = training_manifest
        self.paired_manifest = paired_manifest
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root).resolve()
        self.frozen_ok_path = Path(frozen_ok_path)
        self.run_id = run_id
        self.spec_sha256 = spec_sha256
        self.code_sha256 = code_sha256
        self._target_read_count = 0
        self._ledger_header_written = False

        # Allowed unpaired A/B training stems per domain.
        self._allowed = set()
        for f in training_manifest:
            side = f["side"]
            if side in ("A", "B"):
                self._allowed.add((f["domain"], side, f["stem"]))

        # Forbidden same-stem target and paired-development targets.
        self._forbidden = set()
        for f in paired_manifest:
            self._forbidden.add((f["domain"], f["stem"]))

    @property
    def frozen(self) -> bool:
        return self.frozen_ok_path.exists()

    def _classify(self, path: str) -> dict:
        p = Path(path)
        try:
            resolved = p.resolve(strict=False)
        except Exception:
            resolved = p
        rel = resolved.relative_to(self.data_root)
        parts = rel.parts
        if len(parts) >= 3 and parts[-2] in ("input", "target"):
            domain = parts[0]
            stem = parts[-1].rsplit(".", 1)[0]
            side_kind = parts[-2]
            return {"domain": domain, "side": side_kind, "stem": stem, "resolved": str(resolved)}
        return {"domain": None, "side": None, "stem": None, "resolved": str(resolved)}

    def request(
        self,
        path: str,
        *,
        role: str,
        purpose: str,
        phase: str = "",
        lane: str = "",
        epoch: int | None = None,
    ) -> str:
        """Validate and record a file read; return the resolved path if allowed."""
        info = self._classify(path)
        domain, side, stem = info["domain"], info["side"], info["stem"]

        allowed = False
        if not self.frozen:
            if domain is not None and side in ("input", "target") and stem is not None:
                if side == "target":
                    # Paired targets are normally sealed; the isolated legacy
                    # evaluator calibration process is the only pre-freeze
                    # exception and must be recorded as such.
                    allowed = purpose == "legacy_evaluator_calibration"
                elif (domain, "A", stem) in self._allowed or (domain, "B", stem) in self._allowed:
                    allowed = True
            if side == "target":
                self._target_read_count += 1
        else:
            allowed = True

        self._append_csv(
            path=str(path),
            role=role,
            purpose=purpose,
            resolved=info["resolved"],
            allowed=allowed,
            phase=phase,
            lane=lane,
            epoch=epoch,
            stem=stem,
        )
        if not allowed:
            raise PermissionError(
                f"target access rejected: {path} (role={role}, purpose={purpose})"
            )
        return info["resolved"]

    def open_image(
        self,
        path: str,
        *,
        role: str,
        purpose: str,
        phase: str = "",
        lane: str = "",
        epoch: int | None = None,
    ):
        """The unique image-open entry point for all target/source reads."""
        from PIL import Image

        resolved = self.request(
            path,
            role=role,
            purpose=purpose,
            phase=phase,
            lane=lane,
            epoch=epoch,
        )
        return Image.open(resolved)

    def _append_csv(
        self,
        *,
        path,
        role,
        purpose,
        resolved,
        allowed,
        phase,
        lane,
        epoch,
        stem,
    ) -> None:
        header = [
            "timestamp_utc",
            "phase",
            "lane",
            "epoch",
            "purpose",
            "role",
            "path",
            "resolved_path",
            "stem",
            "allowed",
            "training_frozen_sha256",
            "run_id",
            "spec_sha256",
            "code_sha256",
        ]
        row = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "lane": lane,
            "epoch": "" if epoch is None else str(int(epoch)),
            "purpose": purpose,
            "role": role,
            "path": path,
            "resolved_path": resolved,
            "stem": "" if stem is None else stem,
            "allowed": "true" if allowed else "false",
            "training_frozen_sha256": "",
            "run_id": self.run_id,
            "spec_sha256": self.spec_sha256,
            "code_sha256": self.code_sha256,
        }
        import csv

        new = not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0
        with self.ledger_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            if new:
                writer.writeheader()
            writer.writerow(row)

    def target_read_count(self) -> int:
        return self._target_read_count
