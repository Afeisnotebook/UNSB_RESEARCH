from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import DOMAINS, dump_json, sha256_file, stable_seed
    from .prepare import historical_exclusions, image_files
except ImportError:  # direct script execution
    from common import DOMAINS, dump_json, sha256_file, stable_seed
    from prepare import historical_exclusions, image_files


PROTOCOL_ID = "unsb-shared-bridge-domain-phase-desynchronization-v1"


def canonical_stem(domain: str, stem: str) -> str:
    prefix = f"{domain}__"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--discovery-run-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-view", required=True)
    parser.add_argument("--old-csv-root", required=True)
    parser.add_argument("--per-domain", type=int, default=24)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    discovery_root = Path(args.discovery_run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    source_view = Path(args.source_view).resolve()
    old_csv_root = Path(args.old_csv_root).resolve()
    state_root = run_root / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    historical = historical_exclusions(old_csv_root, repo_root)
    discovery = json.loads(
        (discovery_root / "state" / "HELDOUT_MANIFEST.json").read_text(encoding="utf-8")
    )
    discovery_ids = {
        (row["domain"], canonical_stem(row["domain"], row["stem"])) for row in discovery
    }
    selected_rows: list[dict] = []
    availability: dict[str, dict] = {}
    for domain in DOMAINS:
        test_a = source_view / "single_100" / domain / "testA"
        all_files = image_files(test_a)
        eligible = [
            path
            for path in all_files
            if (domain, canonical_stem(domain, path.stem)) not in historical
            and (domain, canonical_stem(domain, path.stem)) not in discovery_ids
        ]
        ranked = sorted(
            eligible,
            key=lambda path: stable_seed(PROTOCOL_ID, domain, path.stem, "confirm"),
        )
        selected = ranked[: args.per_domain]
        if len(selected) != args.per_domain:
            raise RuntimeError(f"{domain}: only {len(selected)} eligible confirmation images")
        availability[domain] = {
            "testA_total": len(all_files),
            "historical_excluded_from_source_pool": sum(
                (domain, canonical_stem(domain, path.stem)) in historical for path in all_files
            ),
            "discovery_excluded": sum(d == domain for d, _ in discovery_ids),
            "eligible": len(eligible),
            "selected": len(selected),
        }
        for rank, path in enumerate(selected):
            selected_rows.append(
                {
                    "role": "internal_heldout_confirmation",
                    "domain": domain,
                    "stem": path.stem,
                    "source_path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "selection_rank": rank,
                    "representative": rank == 0,
                }
            )

    dump_json(state_root / "HELDOUT_MANIFEST.json", selected_rows)
    discovery_paths = json.loads(
        (discovery_root / "state" / "PATH_MAP.json").read_text(encoding="utf-8")
    )
    dump_json(
        state_root / "PATH_MAP.json",
        {
            "run_root": str(run_root),
            "repo_root": str(repo_root),
            "baseline_root": discovery_paths["baseline_root"],
            "source_view": str(source_view),
            "checkpoints_root": discovery_paths["checkpoints_root"],
            "raw_root": str(run_root / "raw"),
            "reports_root": str(run_root / "reports"),
            "figures_root": str(run_root / "figures"),
            "discovery_run_root": str(discovery_root),
        },
    )
    manifest_path = state_root / "HELDOUT_MANIFEST.json"
    audit = {
        "verdict": "PASS",
        "protocol_id": PROTOCOL_ID,
        "status": "INTERNAL_HELDOUT_CONFIRMATION_FROM_SAME_SOURCE_POOL_NOT_SEALED",
        "availability": availability,
        "selected_rows": len(selected_rows),
        "unique_domain_stems": len({(row["domain"], row["stem"]) for row in selected_rows}),
        "overlap_historical": sum(
            (row["domain"], canonical_stem(row["domain"], row["stem"])) in historical
            for row in selected_rows
        ),
        "overlap_discovery": sum(
            (row["domain"], canonical_stem(row["domain"], row["stem"])) in discovery_ids
            for row in selected_rows
        ),
        "target_content_read": False,
        "pixel_content_read": False,
        "manifest_sha256": sha256_file(manifest_path),
    }
    if (
        len(selected_rows) != len(DOMAINS) * args.per_domain
        or audit["unique_domain_stems"] != len(selected_rows)
        or audit["overlap_historical"]
        or audit["overlap_discovery"]
    ):
        audit["verdict"] = "HOLD"
    dump_json(state_root / "CONFIRMATION_SPLIT_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
