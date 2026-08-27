"""Append-only hypothesis lineage with explicit falsification contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "generation",
    "parents",
    "observed_failure",
    "unsb_object",
    "operator",
    "identity_condition",
    "self_null_condition",
    "paired_target_access",
    "falsification_test",
    "status",
}


class HypothesisLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: list[dict] = []
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != "clean-unsb-search003-hypotheses-v1":
                raise RuntimeError("hypothesis ledger schema mismatch")
            self.entries = list(payload.get("hypotheses", []))

    def append(self, entry: dict) -> None:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"hypothesis is missing fields: {sorted(missing)}")
        if entry["paired_target_access"] is not False:
            raise ValueError("candidate construction cannot access paired targets")
        if any(existing["id"] == entry["id"] for existing in self.entries):
            raise ValueError(f"duplicate hypothesis id: {entry['id']}")
        self.entries.append(copy.deepcopy(entry))
        self.flush()

    def transition(self, hypothesis_id: str, *, status: str, evidence: dict) -> None:
        for entry in self.entries:
            if entry["id"] == hypothesis_id:
                history = entry.setdefault("history", [])
                history.append({"from": entry["status"], "to": status, "evidence": evidence})
                entry["status"] = status
                self.flush()
                return
        raise KeyError(hypothesis_id)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "clean-unsb-search003-hypotheses-v1",
            "hypotheses": self.entries,
            "confirmation20_opened": False,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)
