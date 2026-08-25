from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import DOMAINS, dump_json, sha256_file, stable_seed
    from .prepare import historical_exclusions, image_files
    from .prepare_phase_confirmation import canonical_stem
except ImportError:  # direct execution
    from common import DOMAINS, dump_json, sha256_file, stable_seed
    from prepare import historical_exclusions, image_files
    from prepare_phase_confirmation import canonical_stem


PROTOCOL_ID = "unsb-domain-phase-mapping-reconfirmation-v1"


def manifest_ids(path: Path) -> set[tuple[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["domain"], canonical_stem(row["domain"], row["stem"])) for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--discovery-root", required=True)
    parser.add_argument("--first-confirm-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-view", required=True)
    parser.add_argument("--old-csv-root", required=True)
    parser.add_argument("--per-domain", type=int, default=16)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    discovery_root = Path(args.discovery_root).resolve()
    confirm_root = Path(args.first_confirm_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    source_view = Path(args.source_view).resolve()
    historical = historical_exclusions(Path(args.old_csv_root).resolve(), repo_root)
    discovery_ids = manifest_ids(discovery_root / "state" / "HELDOUT_MANIFEST.json")
    confirm_ids = manifest_ids(confirm_root / "state" / "HELDOUT_MANIFEST.json")

    selected_rows: list[dict] = []
    availability = {}
    for domain in DOMAINS:
        files = image_files(source_view / "single_100" / domain / "testA")
        eligible = [
            path
            for path in files
            if (domain, canonical_stem(domain, path.stem)) not in historical
            and (domain, canonical_stem(domain, path.stem)) not in discovery_ids
            and (domain, canonical_stem(domain, path.stem)) not in confirm_ids
        ]
        ranked = sorted(
            eligible,
            key=lambda path: stable_seed(PROTOCOL_ID, domain, path.stem, "tertiary"),
        )
        selected = ranked[: args.per_domain]
        if len(selected) != args.per_domain:
            raise RuntimeError(f"{domain}: only {len(selected)} tertiary candidates")
        availability[domain] = {
            "source_total": len(files),
            "eligible_after_union_exclusion": len(eligible),
            "selected": len(selected),
        }
        for rank, path in enumerate(selected):
            selected_rows.append(
                {
                    "role": "tertiary_internal_heldout",
                    "domain": domain,
                    "stem": path.stem,
                    "source_path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "selection_rank": rank,
                    "representative": rank == 0,
                }
            )

    state = run_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    manifest_path = state / "HELDOUT_MANIFEST.json"
    dump_json(manifest_path, selected_rows)
    discovery_paths = json.loads(
        (discovery_root / "state" / "PATH_MAP.json").read_text(encoding="utf-8")
    )
    dump_json(
        state / "PATH_MAP.json",
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
            "first_confirmation_run_root": str(confirm_root),
        },
    )

    def overlap(ids: set[tuple[str, str]]) -> int:
        return sum(
            (row["domain"], canonical_stem(row["domain"], row["stem"])) in ids
            for row in selected_rows
        )

    audit = {
        "verdict": "PASS",
        "protocol_id": PROTOCOL_ID,
        "status": "TERTIARY_INTERNAL_HELDOUT_FROM_SAME_SOURCE_POOL_NOT_SEALED",
        "availability": availability,
        "selected_rows": len(selected_rows),
        "overlap_historical": overlap(historical),
        "overlap_discovery": overlap(discovery_ids),
        "overlap_first_confirmation": overlap(confirm_ids),
        "pixel_content_read": False,
        "target_content_read": False,
        "manifest_sha256": sha256_file(manifest_path),
    }
    if (
        len(selected_rows) != len(DOMAINS) * args.per_domain
        or audit["overlap_historical"]
        or audit["overlap_discovery"]
        or audit["overlap_first_confirmation"]
    ):
        audit["verdict"] = "HOLD"
    dump_json(state / "TERTIARY_SPLIT_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
