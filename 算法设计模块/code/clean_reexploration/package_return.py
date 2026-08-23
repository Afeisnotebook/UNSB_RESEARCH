"""Build and verify the single return ZIP and its external SHA-256 sidecar."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


FORBIDDEN_PARTS = {".git", "__pycache__", ".deps"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_NAMES = {".pth", ".pt", ".ckpt", ".zip", ".bundle", ".tar.gz", ".tar"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_forbidden(rel: Path) -> bool:
    parts = rel.parts
    if any(p in FORBIDDEN_PARTS for p in parts):
        return True
    name = rel.name.lower()
    if any(name.endswith(s) for s in FORBIDDEN_SUFFIXES):
        return True
    if any(name.endswith(s) for s in FORBIDDEN_NAMES):
        return True
    return False


def build_manifest(staging: Path) -> dict:
    entries = {}
    for p in sorted(staging.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(staging)
        if rel.name == "MANIFEST.sha256":
            continue
        if _is_forbidden(rel):
            raise RuntimeError(f"forbidden content in return staging: {rel}")
        entries[rel.as_posix()] = sha256_file(p)
    return entries


def package_return(
    *,
    staging: Path,
    output_dir: Path,
    zip_name: str,
) -> tuple[Path, Path]:
    """Create the single ZIP (outside staging) and an external sidecar."""
    staging = Path(staging)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = build_manifest(staging)
    manifest_text = "".join(f"{h}  {rel}\n" for rel, h in sorted(entries.items()))
    manifest_path = staging / "MANIFEST.sha256"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    zip_path = output_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(staging.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(staging)
            zf.write(p, arcname=rel.as_posix())

    sidecar_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    zip_sha = sha256_file(zip_path)
    sidecar_path.write_text(f"{zip_sha}  {zip_name}\n", encoding="utf-8")
    return zip_path, sidecar_path


def fresh_directory_acceptance(zip_path: Path, fresh_dir: Path) -> dict:
    """Extract into a fresh directory and run acceptance checks."""
    zip_path = Path(zip_path)
    fresh_dir = Path(fresh_dir)
    if fresh_dir.exists():
        shutil.rmtree(fresh_dir)
    fresh_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(fresh_dir)

    manifest = fresh_dir / "MANIFEST.sha256"
    if not manifest.is_file():
        return {"ok": False, "reason": "missing internal MANIFEST.sha256"}

    errors = []
    files = sorted(p for p in fresh_dir.rglob("*") if p.is_file())
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            expected[rel] = h
    actual = {p.relative_to(fresh_dir).as_posix() for p in files if p.name != "MANIFEST.sha256"}
    if set(expected) != actual:
        errors.append(
            f"manifest mismatch missing={sorted(actual - set(expected))} extra={sorted(set(expected) - actual)}"
        )
    for rel, h in expected.items():
        p = fresh_dir / rel
        if p.is_file() and sha256_file(p) != h:
            errors.append(f"hash mismatch {rel}")
    for p in files:
        if _is_forbidden(p.relative_to(fresh_dir)):
            errors.append(f"forbidden content {p.relative_to(fresh_dir)}")
        if p.suffix.lower() in {".json", ".md", ".csv", ".sha256"}:
            try:
                p.read_text(encoding="utf-8")
            except Exception as e:
                errors.append(f"unreadable UTF-8 {p.relative_to(fresh_dir)}: {e}")
    return {"ok": not errors, "errors": errors, "files": len(files)}
