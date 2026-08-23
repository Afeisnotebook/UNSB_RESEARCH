"""Target-access guard and append-only access ledger.

Before ``TRAINING_FROZEN.ok`` exists, training/diagnostics may only read the
unpaired T2 A/B images.  Any same-stem target, paired-development target, or
official test/confirmation path must be actively rejected, and every file read
is recorded in an append-only ledger.
"""

from __future__ import annotations

import json
import os
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
    ):
        self.training_manifest = training_manifest
        self.paired_manifest = paired_manifest
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root).resolve()
        self.frozen_ok_path = Path(frozen_ok_path)
        self._target_read_count = 0

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

    def request(self, path: str, *, role: str, purpose: str) -> str:
        """Validate and record a file read; return the resolved path if allowed."""
        info = self._classify(path)
        domain, side, stem = info["domain"], info["side"], info["stem"]

        allowed = False
        if not self.frozen:
            if domain is not None and side in ("input", "target") and stem is not None:
                if side == "target":
                    # Same-stem target and paired-development targets are forbidden.
                    allowed = False
                elif (domain, "A", stem) in self._allowed or (domain, "B", stem) in self._allowed:
                    allowed = True
            if side == "target":
                self._target_read_count += 1
        else:
            allowed = True

        rec = AccessRecord(
            path=str(path),
            role=role,
            purpose=purpose,
            resolved=info["resolved"],
            allowed=allowed,
        )
        self._append(rec)
        if not allowed:
            raise PermissionError(
                f"target access rejected: {path} (role={role}, purpose={purpose})"
            )
        return info["resolved"]

    def _append(self, rec: AccessRecord) -> None:
        line = json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def target_read_count(self) -> int:
        return self._target_read_count
