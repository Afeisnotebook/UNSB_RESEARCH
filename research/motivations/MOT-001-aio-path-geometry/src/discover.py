#!/usr/bin/env python3
"""Read-only discovery for the UNSB motivation baseline restart.

Writes PATH_MAP.json, CODE_IDENTITY.json and DATA_MANIFEST.{json,csv} into the
project root. It never reads pixels; it only lists identities and hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[2]
DATA_ROOT = Path(
    os.environ.get("UNSB_DATA_ROOT", PROJECT_ROOT / "_missing_source_dataset")
).expanduser().resolve()
BASELINE_ROOT = Path(
    os.environ.get(
        "UNSB_BASELINE_ROOT", REPOSITORY_ROOT / "foundation" / "canonical" / "src"
    )
).expanduser().resolve()
DTCOV_ROOT = Path(
    os.environ.get(
        "UNSB_DTCOV_ROOT",
        REPOSITORY_ROOT / "research" / "candidates" / "CAND-001-dt-covmatch",
    )
).expanduser().resolve()
HJ_ROOT = Path(
    os.environ.get(
        "UNSB_HJ_ROOT",
        REPOSITORY_ROOT / "research" / "candidates" / "CAND-002-hj-patchnce",
    )
).expanduser().resolve()
BOOTSTRAP_ROOT = Path(
    os.environ.get("UNSB_BOOTSTRAP_ROOT", PROJECT_ROOT / "_missing_bootstrap_reference")
).expanduser().resolve()
PYTHON = os.environ.get("UNSB_PYTHON", sys.executable)

DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
SPLIT = {"trainA": 100, "trainB": 100, "discovery": 80, "sealed": 20}
SEED = 2026
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(dirpath: Path) -> list[Path]:
    if not dirpath.exists():
        return []
    return sorted(p for p in dirpath.iterdir() if p.suffix.lower() in IMG_EXTS)


def write_path_map() -> None:
    path_map = {
        "project_root": str(PROJECT_ROOT),
        "source_root": str(BASELINE_ROOT),
        "dtcov_source_root": str(DTCOV_ROOT),
        "hj_source_root": str(HJ_ROOT),
        "measurement_reference_root": str(BOOTSTRAP_ROOT),
        "dataset_root": str(DATA_ROOT),
        "run_root": str(PROJECT_ROOT / "datasets"),
        "checkpoints_root": str(PROJECT_ROOT / "checkpoints"),
        "state_root": str(PROJECT_ROOT),
        "runtime_root": str(REPOSITORY_ROOT / "runs" / "MOT-001"),
        "python": PYTHON,
        "seed": SEED,
        "domains": DOMAINS,
        "split": SPLIT,
    }
    (PROJECT_ROOT / "PATH_MAP.json").write_text(
        json.dumps(path_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_code_identity() -> None:
    files = [
        BASELINE_ROOT / "train.py",
        BASELINE_ROOT / "test.py",
        BASELINE_ROOT / "models/__init__.py",
        BASELINE_ROOT / "models/sb_model.py",
        BASELINE_ROOT / "models/base_model.py",
        BASELINE_ROOT / "models/networks.py",
        BASELINE_ROOT / "models/ncsn_networks.py",
        BASELINE_ROOT / "models/patchnce.py",
        BASELINE_ROOT / "models/det_pad.py",
        BASELINE_ROOT / "models/hnek/hnek_kernel.py",
        BASELINE_ROOT / "data/unaligned_dataset.py",
        BASELINE_ROOT / "data/base_dataset.py",
        BASELINE_ROOT / "data/image_folder.py",
        DTCOV_ROOT / "dtcov/dtcovmatch.py",
        DTCOV_ROOT / "dtcov/model.py",
        HJ_ROOT / "hj/core.py",
        HJ_ROOT / "hj/structure.py",
        HJ_ROOT / "hj/projection.py",
        BOOTSTRAP_ROOT / "scripts/run_h2_diagnostics.py",
        BOOTSTRAP_ROOT / "diagnostics/vector_geometry.py",
    ]
    records = []
    for path in files:
        exists = path.exists()
        records.append(
            {
                "path": str(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    identity = {
        "generated_for": "motivation_baseline_restart",
        "unsb_tired_git_commit": "0ca907518d144f6e9cd948800e214b25ba5ec829",
        "upstream_cyclomon_unbs_commit": "d1f644f7777e19d5afe5aea3e5cb4bd3afd9b88b",
        "python": PYTHON,
        "files": records,
    }
    (PROJECT_ROOT / "CODE_IDENTITY.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_split() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    summary: dict = {}
    for domain in DOMAINS:
        in_dir = DATA_ROOT / domain / "input"
        tgt_dir = DATA_ROOT / domain / "target"
        inputs = list_images(in_dir)
        targets = list_images(tgt_dir)
        rng_in = random.Random(f"{SEED}:{domain}:input")
        rng_tgt = random.Random(f"{SEED}:{domain}:target")
        rng_in.shuffle(inputs)
        rng_tgt.shuffle(targets)

        n_tr_a = SPLIT["trainA"]
        n_tr_b = SPLIT["trainB"]
        n_disc = SPLIT["discovery"]
        n_sealed = SPLIT["sealed"]
        if len(inputs) < n_tr_a + n_disc + n_sealed:
            raise SystemExit(f"{domain}: not enough input images for split")
        if len(targets) < n_tr_b + n_disc:
            raise SystemExit(f"{domain}: not enough target images for split")

        train_a = inputs[:n_tr_a]
        discovery = inputs[n_tr_a : n_tr_a + n_disc]
        sealed = inputs[n_tr_a + n_disc : n_tr_a + n_disc + n_sealed]
        train_b = targets[:n_tr_b]
        discovery_target = targets[n_tr_b : n_tr_b + n_disc]

        def add(path: Path, side: str, role: str) -> None:
            rows.append(
                {
                    "domain": domain,
                    "side": side,
                    "role": role,
                    "stem": path.stem,
                    "ext": path.suffix.lstrip("."),
                    "source_path": str(path),
                }
            )

        for p in train_a:
            add(p, "input", "trainA")
        for p in train_b:
            add(p, "target", "trainB")
        for p in discovery:
            add(p, "input", "discovery")
        for p in discovery_target:
            add(p, "target", "discovery_target")
        for p in sealed:
            add(p, "input", "sealed")

        summary[domain] = {
            "input_total": len(inputs),
            "target_total": len(targets),
            "trainA": n_tr_a,
            "trainB": n_tr_b,
            "discovery": n_disc,
            "discovery_target": n_disc,
            "sealed": n_sealed,
        }
    return rows, summary


def write_data_manifest(rows: list[dict], summary: dict) -> None:
    payload = {
        "schema_version": 1,
        "seed": SEED,
        "split": SPLIT,
        "domains": DOMAINS,
        "dataset_root": str(DATA_ROOT),
        "pairing_used_for_training": False,
        "sealed_open_policy": "closed",
        "split_summary": summary,
        "files": rows,
    }
    (PROJECT_ROOT / "DATA_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cols = ["domain", "side", "role", "stem", "ext", "source_path"]
    with (PROJECT_ROOT / "DATA_MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r[c] for c in cols})


def main() -> int:
    if not DATA_ROOT.is_dir():
        raise SystemExit(
            "source dataset not found; set UNSB_DATA_ROOT to the five-domain "
            f"dataset root (resolved default: {DATA_ROOT})"
        )
    write_path_map()
    write_code_identity()
    rows, summary = build_split()
    write_data_manifest(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DATA_MANIFEST rows = {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
