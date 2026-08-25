"""Versioned frozen config with a canonical hash."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


def canonical_json(obj) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_config(
    *,
    name: str,
    phase: str,
    args: dict,
    meta: dict | None = None,
) -> dict:
    """Return a frozen config record with a canonical hash."""
    payload = {
        "schema_version": 1,
        "name": name,
        "phase": phase,
        "args": args,
        "meta": meta or {},
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["config_hash"] = sha256(canonical_json(payload))
    return payload
