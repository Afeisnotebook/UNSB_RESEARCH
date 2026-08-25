from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import DOMAINS, dump_json, nested_domain_image_bootstrap, sha256_file
except ImportError:  # direct script execution
    from common import DOMAINS, dump_json, nested_domain_image_bootstrap, sha256_file


PROTOCOL_ID = "unsb-reciprocal-bridge-kernel-deflection-v1"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    primary_path = run_root / "raw" / "RECIPROCAL_KERNEL_PRIMARY.csv"
    age_path = run_root / "raw" / "RECIPROCAL_KERNEL_BY_AGE.csv"
    frame = pd.read_csv(primary_path)
    if len(frame) != 300 or set(frame["domain"]) != set(DOMAINS):
        raise RuntimeError("reciprocal primary evidence identity mismatch")

    time_rows: list[dict] = []
    domain_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    passing_times: list[bool] = []
    for t in (1, 2, 3):
        block = frame[frame["bridge_time_index"] == t]
        records = block[["domain", "stem", "log10_deflection_excess"]].to_dict("records")
        summary, draws = nested_domain_image_bootstrap(
            records,
            "log10_deflection_excess",
            draws=args.draws,
            seed=20410824 + t * 137,
        )
        above_fraction = float(block["above_reference"].astype(bool).mean())
        passes = bool(
            summary["ci_low"] > 0
            and summary["positive_domain_count"] >= 4
            and above_fraction >= 0.80
        )
        passing_times.append(passes)
        time_rows.append(
            {
                "bridge_time_index": t,
                "bridge_time_value": float(block["bridge_time_value"].iloc[0]),
                **summary,
                "above_reference_fraction": above_fraction,
                "median_min_KDD": float(block["min_reciprocal_KDD"].median()),
                "median_reference": float(block["primary_reference"].median()),
                "passes_frozen_gate": passes,
            }
        )
        bootstrap_rows.extend(
            {"bridge_time_index": t, "draw": index, "value": float(value)}
            for index, value in enumerate(draws)
        )
        for domain_index, domain in enumerate(DOMAINS):
            domain_block = block[block["domain"] == domain]
            values = domain_block["log10_deflection_excess"].to_numpy(dtype=np.float64)
            rng = np.random.default_rng(20410824 + t * 100 + domain_index)
            indices = rng.integers(0, len(values), size=(args.draws, len(values)))
            means = values[indices].mean(axis=1)
            domain_rows.append(
                {
                    "bridge_time_index": t,
                    "bridge_time_value": float(domain_block["bridge_time_value"].iloc[0]),
                    "domain": domain,
                    "n_images": len(values),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci_low": float(np.quantile(means, 0.025)),
                    "ci_high": float(np.quantile(means, 0.975)),
                    "above_reference_fraction": float(
                        domain_block["above_reference"].astype(bool).mean()
                    ),
                    "median_best_epoch": float(domain_block["best_single_epoch"].median()),
                }
            )

    n_pass = int(sum(passing_times))
    if n_pass >= 2:
        verdict = "SUPPORTED_SHARED_KERNEL_DEFLECTION"
    elif n_pass == 1:
        verdict = "PARTIAL_KERNEL_DEFLECTION"
    else:
        verdict = "NOT_DISTINCT_FROM_SINGLE_AGE_TRAJECTORY"

    best_age_counts = (
        frame.groupby(["bridge_time_index", "best_single_epoch"]).size().rename("count").reset_index()
    )
    best_age_path = run_root / "reports" / "RECIPROCAL_BEST_AGE_COUNTS.csv"
    best_age_counts.to_csv(best_age_path, index=False, encoding="utf-8")
    time_path = run_root / "reports" / "RECIPROCAL_TIME_SUMMARY.csv"
    domain_path = run_root / "reports" / "RECIPROCAL_DOMAIN_SUMMARY.csv"
    draws_path = run_root / "reports" / "RECIPROCAL_BOOTSTRAP_DRAWS.csv"
    write_csv(time_path, time_rows)
    write_csv(domain_path, domain_rows)
    write_csv(draws_path, bootstrap_rows)

    adjudication = {
        "protocol_id": PROTOCOL_ID,
        "verdict": verdict,
        "passing_bridge_times": passing_times,
        "n_passing_bridge_times": n_pass,
        "time_summary": time_rows,
        "best_age_counts": best_age_counts.to_dict("records"),
        "claim_boundary": {
            "allowed_if_supported": "At common bridge states, the initial shared model's conditional mean direction lies outside the stochastic floor and the task-specific epoch-1-to-5 direction trajectory.",
            "always_forbidden": [
                "causal downstream harm",
                "calibrated posterior uncertainty",
                "sealed confirmatory evidence",
                "a claim about all training stages",
            ],
            "heldout_status": "HELDOUT_WITHIN_DISCOVERY_NOT_CONFIRMATORY",
        },
        "evidence_hashes": {
            primary_path.name: sha256_file(primary_path),
            age_path.name: sha256_file(age_path),
            time_path.name: sha256_file(time_path),
            domain_path.name: sha256_file(domain_path),
        },
    }
    dump_json(run_root / "reports" / "RECIPROCAL_ADJUDICATION.json", adjudication)
    print(json.dumps(adjudication, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
