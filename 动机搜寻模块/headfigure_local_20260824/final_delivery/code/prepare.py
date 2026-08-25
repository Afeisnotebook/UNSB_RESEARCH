from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

try:
    from .common import DOMAINS, dump_json, sha256_file, stable_seed
except ImportError:  # direct script execution
    from common import DOMAINS, dump_json, sha256_file, stable_seed


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROTOCOL_ID = "unsb-initial-shared-bridge-fanout-dual-control-v1"


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def hardlink_checked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise RuntimeError(f"existing hardlink target differs: {destination}")
        return
    os.link(source, destination)


def historical_exclusions(old_csv_root: Path, repo_root: Path) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    for csv_path in old_csv_root.glob("epoch_*/DIRECTION_IMAGE_ROWS.csv"):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("domain") in DOMAINS and row.get("stem"):
                    excluded.add((row["domain"], row["stem"]))
    manifest_path = repo_root / "动机搜寻模块" / "MEASUREMENT_MANIFEST.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("c_subset", "b_medoids"):
            for row in payload.get(key, []):
                if row.get("domain") in DOMAINS and row.get("stem"):
                    excluded.add((row["domain"], row["stem"]))
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-view", required=True)
    parser.add_argument("--old-csv-root", required=True)
    parser.add_argument("--heldout-per-domain", type=int, default=20)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    source_view = Path(args.source_view).resolve()
    old_csv_root = Path(args.old_csv_root).resolve()
    single_source = source_view / "single_100"
    data_root = run_root / "data_views"
    single_root = data_root / "single"
    aio_root = data_root / "aio"
    state_root = run_root / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    excluded = historical_exclusions(old_csv_root, repo_root)
    manifest_rows: list[dict] = []
    heldout_rows: list[dict] = []

    for domain in DOMAINS:
        source_domain = single_source / domain
        for split in ("trainA", "trainB"):
            files = image_files(source_domain / split)
            if len(files) != 100:
                raise RuntimeError(f"expected 100 {domain}/{split}, found {len(files)}")
            for source in files:
                single_dest = single_root / domain / split / source.name
                aio_name = f"{domain}__{source.name}"
                aio_dest = aio_root / split / aio_name
                hardlink_checked(source, single_dest)
                hardlink_checked(source, aio_dest)
                digest = sha256_file(source)
                manifest_rows.append(
                    {
                        "role": "train",
                        "domain": domain,
                        "split": split,
                        "stem": source.stem,
                        "source_path": str(source),
                        "single_path": str(single_dest),
                        "aio_path": str(aio_dest),
                        "bytes": source.stat().st_size,
                        "sha256": digest,
                    }
                )

        candidates = [
            p for p in image_files(source_domain / "testA") if (domain, p.stem) not in excluded
        ]
        ranked = sorted(candidates, key=lambda p: stable_seed(PROTOCOL_ID, domain, p.stem))
        selected = ranked[: args.heldout_per_domain]
        if len(selected) != args.heldout_per_domain:
            raise RuntimeError(f"insufficient unmeasured heldout images for {domain}")
        for rank, source in enumerate(selected):
            digest = sha256_file(source)
            row = {
                "role": "heldout_within_discovery",
                "domain": domain,
                "stem": source.stem,
                "source_path": str(source),
                "bytes": source.stat().st_size,
                "sha256": digest,
                "selection_rank": rank,
                "representative": rank == 0,
            }
            heldout_rows.append(row)
            manifest_rows.append(row.copy())

    csv_path = state_root / "DATA_MANIFEST.csv"
    fields = sorted({key for row in manifest_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    dump_json(state_root / "HELDOUT_MANIFEST.json", heldout_rows)
    dump_json(
        state_root / "PATH_MAP.json",
        {
            "run_root": str(run_root),
            "repo_root": str(repo_root),
            "baseline_root": str(repo_root / "算法设计模块" / "code" / "baseline"),
            "source_view": str(source_view),
            "single_data_root": str(single_root),
            "aio_data_root": str(aio_root),
            "checkpoints_root": str(run_root / "checkpoints"),
            "raw_root": str(run_root / "raw"),
            "reports_root": str(run_root / "reports"),
            "figures_root": str(run_root / "figures"),
        },
    )
    per_domain = {
        domain: {
            "trainA": sum(r["domain"] == domain and r.get("split") == "trainA" for r in manifest_rows),
            "trainB": sum(r["domain"] == domain and r.get("split") == "trainB" for r in manifest_rows),
            "heldout": sum(r["domain"] == domain and r["role"] == "heldout_within_discovery" for r in manifest_rows),
            "excluded_historical": sum(d == domain for d, _ in excluded),
        }
        for domain in DOMAINS
    }
    audit = {
        "verdict": "PASS",
        "protocol_id": PROTOCOL_ID,
        "domains": DOMAINS,
        "per_domain": per_domain,
        "manifest_rows": len(manifest_rows),
        "heldout_rows": len(heldout_rows),
        "heldout_unique": len({(r["domain"], r["stem"]) for r in heldout_rows}),
        "historically_measured_stems_excluded": len(excluded),
        "pairing_used_for_training": False,
        "target_content_read": False,
        "sealed_content_read": False,
        "data_manifest_sha256": sha256_file(csv_path),
    }
    if any(v["trainA"] != 100 or v["trainB"] != 100 or v["heldout"] != args.heldout_per_domain for v in per_domain.values()):
        audit["verdict"] = "HOLD"
    dump_json(state_root / "PREPARE_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
