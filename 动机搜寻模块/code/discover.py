#!/usr/bin/env python3
"""Read-only discovery for the UNSB motivation baseline restart.

Writes PATH_MAP.json, CODE_IDENTITY.json and DATA_MANIFEST.{json,csv} into the
project root. It never reads pixels; it only lists identities and hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/home/yc/UNSB_C21/dataset_all")
REFACTOR_ROOT = Path("/home/yc/unsb_tired/refactor")
BOOTSTRAP_ROOT = Path(
    "/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806"
)
PYTHON = "/home/yc/anaconda3/envs/unsb_cov/bin/python"

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
        "source_root": str(REFACTOR_ROOT / "baseline"),
        "dtcov_source_root": str(REFACTOR_ROOT / "dt_covmatch"),
        "hj_source_root": str(REFACTOR_ROOT / "hj_patchnce"),
        "measurement_reference_root": str(BOOTSTRAP_ROOT),
        "dataset_root": str(DATA_ROOT),
        "run_root": str(PROJECT_ROOT / "datasets"),
        "checkpoints_root": str(PROJECT_ROOT / "checkpoints"),
        "state_root": str(PROJECT_ROOT),
        "report_root": str(PROJECT_ROOT / "reports"),
        "raw_root": str(PROJECT_ROOT / "raw"),
        "figures_root": str(PROJECT_ROOT / "figures"),
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
        REFACTOR_ROOT / "baseline/train.py",
        REFACTOR_ROOT / "baseline/test.py",
        REFACTOR_ROOT / "baseline/models/__init__.py",
        REFACTOR_ROOT / "baseline/models/sb_model.py",
        REFACTOR_ROOT / "baseline/models/base_model.py",
        REFACTOR_ROOT / "baseline/models/networks.py",
        REFACTOR_ROOT / "baseline/models/ncsn_networks.py",
        REFACTOR_ROOT / "baseline/models/patchnce.py",
        REFACTOR_ROOT / "baseline/models/det_pad.py",
        REFACTOR_ROOT / "baseline/models/hnek/hnek_kernel.py",
        REFACTOR_ROOT / "baseline/data/unaligned_dataset.py",
        REFACTOR_ROOT / "baseline/data/base_dataset.py",
        REFACTOR_ROOT / "baseline/data/image_folder.py",
        REFACTOR_ROOT / "dt_covmatch/dtcov/dtcovmatch.py",
        REFACTOR_ROOT / "dt_covmatch/dtcov/model.py",
        REFACTOR_ROOT / "hj_patchnce/hj/core.py",
        REFACTOR_ROOT / "hj_patchnce/hj/structure.py",
        REFACTOR_ROOT / "hj_patchnce/hj/projection.py",
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
    write_path_map()
    write_code_identity()
    rows, summary = build_split()
    write_data_manifest(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DATA_MANIFEST rows = {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
