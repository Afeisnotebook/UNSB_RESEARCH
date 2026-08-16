"""Dataset identity and split audit for the UNSB clean-room harness.

The contract is intentionally narrow: a dataset is a directory containing a
``manifest.csv`` with the standard columns plus hardlinked/real image views.
This module never reads pixels; it only audits identities, splits and overlap.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = (
    "dataset",
    "split",
    "filename",
    "input_view",
    "target_view",
    "source_input",
    "source_target",
)


def domain_key(filename: str) -> str:
    """Return the stable lowercase domain key encoded in a final6-style filename."""
    name = Path(str(filename).replace("\\", "/")).name
    if "__" in name:
        return name.split("__", 1)[0].strip().lower() or "unknown"
    stem = name.rsplit(".", 1)[0]
    return stem.split("__", 1)[0].strip().lower() if "__" in stem else "unknown"


def load_manifest(path: Path) -> list[dict]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != set(MANIFEST_COLUMNS):
            raise ValueError(
                f"manifest columns mismatch: {reader.fieldnames} != {MANIFEST_COLUMNS}"
            )
        return [dict(row) for row in reader]


def _identity(row: dict, key: str) -> str:
    if key == "source_target":
        return row.get("source_target") or row.get("filename") or ""
    if key == "filename":
        return row.get("filename") or ""
    raise ValueError(f"unknown identity key: {key}")


def manifest_sha256(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def audit_manifest(
    path: Path,
    *,
    expected_domain_split: dict[str, dict[str, int]] | None = None,
) -> dict:
    """Return identity/split facts for one manifest without reading pixels."""
    path = Path(path)
    rows = load_manifest(path)
    split_counts = Counter(r["split"] for r in rows)
    domain_split_counts: Counter = Counter((r["dataset"], r["split"]) for r in rows)
    filename_counts = Counter(r["filename"] for r in rows)
    source_target_counts = Counter(_identity(r, "source_target") for r in rows)

    duplicate_filenames = {k: v for k, v in filename_counts.items() if v > 1}
    duplicate_source_targets = {
        k: v for k, v in source_target_counts.items() if v > 1
    }

    report = {
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256(path),
        "n_rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "domain_split_counts": {
            f"{d}/{s}": n for (d, s), n in sorted(domain_split_counts.items())
        },
        "duplicate_filenames": len(duplicate_filenames),
        "duplicate_source_targets": len(duplicate_source_targets),
        "domains": sorted({r["dataset"] for r in rows}),
    }

    if expected_domain_split is not None:
        problems = []
        for domain, split_expected in expected_domain_split.items():
            for split, expected in split_expected.items():
                actual = domain_split_counts.get((domain, split), 0)
                if actual != expected:
                    problems.append(f"{domain}/{split}: {actual} != {expected}")
        report["domain_split_ok"] = not problems
        report["domain_split_problems"] = problems

    return report


def zero_overlap(
    path_a: Path,
    path_b: Path,
    *,
    split: str = "test",
    key: str = "source_target",
) -> dict:
    """Report overlap between two manifests restricted to one split."""
    rows_a = [r for r in load_manifest(path_a) if r["split"] == split]
    rows_b = [r for r in load_manifest(path_b) if r["split"] == split]
    ids_a = {_identity(r, key) for r in rows_a}
    ids_b = {_identity(r, key) for r in rows_b}
    overlap = sorted(ids_a & ids_b)
    return {
        "split": split,
        "key": key,
        "count_a": len(ids_a),
        "count_b": len(ids_b),
        "overlap": len(overlap),
        "examples": overlap[:10],
    }
