from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

try:
    from .common import DOMAINS, dump_json, sha256_file, stable_seed
except ImportError:
    from common import DOMAINS, dump_json, sha256_file, stable_seed


PROTOCOL_ID = "unsb-sixdomain-expanded-phase-field-v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_map(folder: Path) -> dict[str, Path]:
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.stem.casefold()
            if key in result:
                raise RuntimeError(f"duplicate case-folded stem in {folder}: {path.stem}")
            result[key] = path
    return result


def hardlink_checked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise RuntimeError(f"existing destination differs: {destination}")
        return
    os.link(source, destination)


def canonical_stem(domain: str, stem: str) -> str:
    prefix = f"{domain}__"
    value = stem
    while value.casefold().startswith(prefix.casefold()):
        value = value[len(prefix) :]
    return value.casefold()


def prior_phase_ids(paths: list[Path]) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                domain = row.get("domain", "")
                stem = row.get("stem", "")
                if domain in DOMAINS and stem:
                    identities.add((domain, canonical_stem(domain, stem)))
    return identities


def rank(paths: list[Path], domain: str, role: str) -> list[Path]:
    return sorted(paths, key=lambda path: stable_seed(PROTOCOL_ID, domain, path.stem, role))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prior-phase-csv", action="append", default=[])
    parser.add_argument("--train-per-domain", type=int, default=120)
    parser.add_argument("--heldout-per-domain", type=int, default=80)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    prior_paths = [Path(value).resolve() for value in args.prior_phase_csv]
    excluded = prior_phase_ids(prior_paths)

    single_root = run_root / "data_views" / "single"
    aio_root = run_root / "data_views" / "aio"
    heldout_root = run_root / "data_views" / "heldoutA"
    state_root = run_root / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    heldout_rows: list[dict] = []
    per_domain: dict[str, dict] = {}

    for domain in DOMAINS:
        inputs = image_map(dataset_root / domain / "input")
        targets = image_map(dataset_root / domain / "target")
        if set(inputs) != set(targets):
            raise RuntimeError(f"input/target stem mismatch for {domain}")
        input_paths = list(inputs.values())
        eligible_heldout = [
            path
            for path in input_paths
            if (domain, canonical_stem(domain, path.stem)) not in excluded
        ]
        heldout = rank(eligible_heldout, domain, "heldout")[: args.heldout_per_domain]
        if len(heldout) != args.heldout_per_domain:
            raise RuntimeError(f"{domain}: insufficient heldout images")
        heldout_stems = {path.stem.casefold() for path in heldout}
        train_a_candidates = [path for path in input_paths if path.stem.casefold() not in heldout_stems]
        train_a = rank(train_a_candidates, domain, "trainA")[: args.train_per_domain]
        if len(train_a) != args.train_per_domain:
            raise RuntimeError(f"{domain}: insufficient trainA images")
        train_a_stems = {path.stem.casefold() for path in train_a}

        # Prefer target identities not present in trainA.  RainDS-syn has only
        # 200 identities, so complete disjointness is impossible at 120/80;
        # the runtime loader remains explicitly unaligned and randomized.
        target_paths = list(targets.values())
        disjoint_b = [path for path in target_paths if path.stem.casefold() not in train_a_stems]
        overlap_b = [path for path in target_paths if path.stem.casefold() in train_a_stems]
        train_b = (rank(disjoint_b, domain, "trainB-disjoint") + rank(overlap_b, domain, "trainB-overlap"))[
            : args.train_per_domain
        ]
        if len(train_b) != args.train_per_domain:
            raise RuntimeError(f"{domain}: insufficient trainB images")

        for role, selected, split in (("trainA", train_a, "trainA"), ("trainB", train_b, "trainB")):
            for order, source in enumerate(selected):
                name = f"{domain}__{source.name}"
                single_dest = single_root / domain / split / name
                aio_dest = aio_root / split / name
                hardlink_checked(source, single_dest)
                hardlink_checked(source, aio_dest)
                manifest_rows.append(
                    {
                        "role": role,
                        "domain": domain,
                        "stem": source.stem,
                        "order": order,
                        "source_path": str(source),
                        "single_path": str(single_dest),
                        "aio_path": str(aio_dest),
                        "bytes": source.stat().st_size,
                        "sha256": sha256_file(source),
                    }
                )

        for order, source in enumerate(heldout):
            destination = heldout_root / domain / f"{domain}__{source.name}"
            hardlink_checked(source, destination)
            row = {
                "role": "expanded_internal_heldout",
                "domain": domain,
                "stem": f"{domain}__{source.stem}",
                "canonical_stem": source.stem.casefold(),
                "order": order,
                "source_path": str(destination),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "representative": order == 0,
                "prior_phase_overlap": (domain, source.stem.casefold()) in excluded,
            }
            heldout_rows.append(row)
            manifest_rows.append(row.copy())

        per_domain[domain] = {
            "source_input": len(inputs),
            "source_target": len(targets),
            "trainA": len(train_a),
            "trainB": len(train_b),
            "heldout": len(heldout),
            "trainA_heldout_overlap": len(train_a_stems & heldout_stems),
            "trainA_trainB_stem_overlap": len(train_a_stems & {path.stem.casefold() for path in train_b}),
            "prior_phase_identities_in_source": sum(identity[0] == domain for identity in excluded),
            "prior_phase_overlap_heldout": sum(
                (domain, path.stem.casefold()) in excluded for path in heldout
            ),
        }

    manifest_path = state_root / "DATA_MANIFEST.csv"
    fields = sorted({key for row in manifest_rows for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    heldout_path = state_root / "HELDOUT_MANIFEST.json"
    dump_json(heldout_path, heldout_rows)
    dump_json(
        state_root / "PATH_MAP.json",
        {
            "run_root": str(run_root),
            "repo_root": str(repo_root),
            "baseline_root": str(repo_root / "算法设计模块" / "code" / "baseline"),
            "dataset_root": str(dataset_root),
            "single_data_root": str(single_root),
            "aio_data_root": str(aio_root),
            "heldout_root": str(heldout_root),
            "checkpoints_root": str(run_root / "checkpoints"),
            "raw_root": str(run_root / "raw"),
            "reports_root": str(run_root / "reports"),
            "figures_root": str(run_root / "figures"),
        },
    )
    audit = {
        "verdict": "PASS",
        "protocol_id": PROTOCOL_ID,
        "heldout_status": "EXPANDED_INTERNAL_HELDOUT_NOT_EXTERNAL_OR_SEALED",
        "domains": DOMAINS,
        "per_domain": per_domain,
        "train_rows": len(DOMAINS) * args.train_per_domain * 2,
        "heldout_rows": len(heldout_rows),
        "heldout_unique": len({(row["domain"], row["canonical_stem"]) for row in heldout_rows}),
        "prior_phase_csv_hashes": {str(path): sha256_file(path) for path in prior_paths},
        "pairing_used_for_training": False,
        "training_loader": "unaligned_random_B",
        "target_content_read_for_measurement": False,
        "data_manifest_sha256": sha256_file(manifest_path),
        "heldout_manifest_sha256": sha256_file(heldout_path),
    }
    if any(
        values["trainA"] != args.train_per_domain
        or values["trainB"] != args.train_per_domain
        or values["heldout"] != args.heldout_per_domain
        or values["trainA_heldout_overlap"] != 0
        or values["prior_phase_overlap_heldout"] != 0
        for values in per_domain.values()
    ):
        audit["verdict"] = "HOLD"
    dump_json(state_root / "PREPARE_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
