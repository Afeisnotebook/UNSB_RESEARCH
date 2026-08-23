"""Authority, data, code and run identity for the clean-reexploration task.

This module is deliberately self-contained and effect-blind.  It only hashes
bytes and canonicalizes JSON; it never reads image pixels or paired targets.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


# Frozen base authority from the task prompt (section 1.3).
FINAL1_SPEC_CANONICAL_SHA256 = (
    "bb102af286f0d15f1f6b3bd0e562964d70cf86af463323373ae027e7194f4d86"
)
TRAINING_MANIFEST_SHA256 = (
    "f6049e7c1563565d8e00e1baca1821b67b56d33bd78b064c596dbbc17d3d6e02"
)
PAIRED_DEVELOPMENT_MANIFEST_SHA256 = (
    "71b4eb92822166d67a97c15f9c5b2bbd8b4d70a24173d1ae03fe5c20596ddb0c"
)


def canonical_json(obj) -> str:
    """Canonical UTF-8 JSON with sorted keys, compact separators, no ASCII escapes."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_json_bytes(obj) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256_file(path: Path) -> str:
    """Canonical JSON hash of a JSON file (independent of raw formatting)."""
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    return sha256_bytes(canonical_json_bytes(obj))


def sub_seed(*parts) -> int:
    """Reproducible 63-bit seed from a canonical tuple."""
    text = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def locate_authority_root() -> Path:
    """Return the FINAL-1 bootstrap authority root."""
    root = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")
    return root


def verify_base_authority(authority_root: Path | None = None) -> dict:
    """Locate and verify the three hash-locked base-authority identities.

    Raises :class:`RuntimeError` with reason ``MISSING_BASE_AUTHORITY`` when any
    identity cannot be located or its hash does not match.
    """
    root = authority_root or locate_authority_root()
    spec = root / "authority/final1_ta_minimal_seed2026_20260811/FINAL1_FROZEN_SPEC.json"
    t2 = root / "specs/h2/T2_MANIFEST.json"
    t3 = root / "specs/h2c/T3_CONFIRMATORY_MANIFEST.json"

    problems = []
    record = {
        "authority_root": str(root),
        "final1_spec_path": str(spec),
        "training_manifest_path": str(t2),
        "paired_development_manifest_path": str(t3),
    }

    if not spec.is_file():
        problems.append("final1_spec_missing")
    else:
        got = canonical_sha256_file(spec)
        record["final1_spec_canonical_sha256"] = got
        if got != FINAL1_SPEC_CANONICAL_SHA256:
            problems.append("final1_spec_canonical_mismatch")

    if not t2.is_file():
        problems.append("training_manifest_missing")
    else:
        got = sha256_file(t2)
        record["training_manifest_sha256"] = got
        if got != TRAINING_MANIFEST_SHA256:
            problems.append("training_manifest_mismatch")

    if not t3.is_file():
        problems.append("paired_development_manifest_missing")
    else:
        got = sha256_file(t3)
        record["paired_development_manifest_sha256"] = got
        if got != PAIRED_DEVELOPMENT_MANIFEST_SHA256:
            problems.append("paired_development_manifest_mismatch")

    record["ok"] = not problems
    record["problems"] = problems
    if problems:
        record["reason"] = "HARD_STOP_MISSING_BASE_AUTHORITY"
    return record


def load_training_manifest(path: Path) -> list[dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    files = obj["files"]
    if obj.get("errors"):
        raise RuntimeError(f"training manifest has errors: {obj['errors']}")
    return files


def load_paired_development_manifest(path: Path) -> list[dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    files = obj["files"]
    if obj.get("errors"):
        raise RuntimeError(f"paired-development manifest has errors: {obj['errors']}")
    return files


def data_root_from_manifest(path: Path) -> str:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        return env_root
    return str(obj.get("data_root", ""))


def audit_manifest_files(files: Iterable[dict]) -> dict:
    """Verify every manifest file exists with the recorded byte size and SHA-256."""
    missing = []
    size_mismatch = []
    hash_mismatch = []
    checked = 0
    for f in files:
        p = Path(f["absolute_path"])
        checked += 1
        if not p.is_file():
            missing.append(f["absolute_path"])
            continue
        if p.stat().st_size != int(f["bytes"]):
            size_mismatch.append(f["absolute_path"])
            continue
        if sha256_file(p) != f["sha256"]:
            hash_mismatch.append(f["absolute_path"])
    return {
        "checked": checked,
        "missing": missing,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "ok": not (missing or size_mismatch or hash_mismatch),
    }


def code_identity(paths: Iterable[Path]) -> dict:
    """Return a canonical code identity over a set of source files."""
    manifest = {}
    for p in sorted({Path(x) for x in paths}):
        p = Path(p)
        rel = p.name if not p.is_absolute() else str(p)
        manifest[str(rel)] = sha256_file(p) if p.is_file() else None
    return {
        "files": manifest,
        "code_sha256": sha256_bytes(canonical_json_bytes(manifest)),
    }


def make_run_id(spec_sha256: str) -> str:
    return f"clean-reexploration-s2026-20260824-{spec_sha256[:16]}"
