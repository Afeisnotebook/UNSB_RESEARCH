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


CONTRASTS = ["exposure_gap", "clock_gap", "age_envelope_excess"]


def load_historic_consensus(repo_root: Path) -> list[dict]:
    rows = []
    for seed in range(2026, 2032):
        path = (
            repo_root / "experiments" / "L1-local"
            / "EXP-L1-MOTIVATION-WINDOW-20260824" / "evidence"
            / f"seed{seed}_window_audit.json"
        )
        audit = json.loads(path.read_text(encoding="utf-8"))
        epoch1 = audit["candidate_windows"]["1-1"]
        for t_text, result in epoch1["per_bridge_time"].items():
            rows.append(
                {
                    "seed": seed,
                    "bridge_time_index": int(t_text),
                    "domain_agree_count": epoch1["agree_count"],
                    "n_domains": epoch1["n_domains"],
                    "pooled_mean": result["mean"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "ci_positive": result["ci_low"] > 0,
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
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
    path_map = json.loads((run_root / "state" / "PATH_MAP.json").read_text(encoding="utf-8"))
    repo_root = Path(path_map["repo_root"])
    raw_path = run_root / "raw" / "DIRECTION_STATISTICS.csv"
    frame = pd.read_csv(raw_path)
    expected_arms = {"aio_e1", "single_e1", "single_e2", "single_e3", "single_e4", "single_e5"}
    if set(frame["arm"]) != expected_arms:
        raise RuntimeError(f"arm mismatch: {set(frame['arm'])}")
    pivot = frame.pivot(index=["domain", "stem", "bridge_time_index"], columns="arm", values="D_sph").reset_index()
    for arm in sorted(expected_arms):
        pivot[f"log_{arm}"] = np.log10(pivot[arm].to_numpy(dtype=float) + 1e-12)
    pivot["exposure_gap"] = pivot["log_aio_e1"] - pivot["log_single_e1"]
    pivot["clock_gap"] = pivot["log_aio_e1"] - pivot["log_single_e5"]
    single_log_cols = [f"log_single_e{epoch}" for epoch in range(1, 6)]
    pivot["single_envelope_log"] = pivot[single_log_cols].max(axis=1)
    pivot["age_envelope_excess"] = pivot["log_aio_e1"] - pivot["single_envelope_log"]
    pivot["single_envelope_epoch"] = (
        pivot[single_log_cols].to_numpy().argmax(axis=1) + 1
    )
    contrast_path = run_root / "reports" / "PAIRED_CONTRAST_ROWS.csv"
    contrast_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(contrast_path, index=False, encoding="utf-8")

    summary_rows: list[dict] = []
    draw_rows: list[dict] = []
    time_passes = []
    for t in (1, 2, 3):
        block = pivot[pivot["bridge_time_index"] == t]
        by_contrast = {}
        for contrast_idx, contrast in enumerate(CONTRASTS):
            records = block[["domain", "stem", contrast]].to_dict("records")
            result, draws = nested_domain_image_bootstrap(
                records,
                contrast,
                draws=args.draws,
                seed=20410824 + t * 101 + contrast_idx,
            )
            result.update({"bridge_time_index": t, "contrast": contrast})
            summary_rows.append(result)
            by_contrast[contrast] = result
            draw_rows.extend(
                {
                    "bridge_time_index": t,
                    "contrast": contrast,
                    "draw": idx,
                    "value": float(value),
                }
                for idx, value in enumerate(draws)
            )
        passes = (
            by_contrast["age_envelope_excess"]["ci_low"] > 0
            and by_contrast["age_envelope_excess"]["positive_domain_count"] >= 4
            and by_contrast["exposure_gap"]["ci_low"] > 0
            and by_contrast["clock_gap"]["ci_low"] > 0
        )
        time_passes.append(bool(passes))

    domain_rows = []
    for t in (1, 2, 3):
        for domain in DOMAINS:
            block = pivot[(pivot["bridge_time_index"] == t) & (pivot["domain"] == domain)]
            for contrast in CONTRASTS:
                values = block[contrast].to_numpy(dtype=float)
                rng = np.random.default_rng(20410824 + t * 1000 + DOMAINS.index(domain) * 10 + CONTRASTS.index(contrast))
                indices = rng.integers(0, len(values), size=(5000, len(values)))
                means = values[indices].mean(axis=1)
                domain_rows.append(
                    {
                        "bridge_time_index": t,
                        "domain": domain,
                        "contrast": contrast,
                        "n_images": len(values),
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "ci_low": float(np.quantile(means, 0.025)),
                        "ci_high": float(np.quantile(means, 0.975)),
                        "positive_fraction": float(np.mean(values > 0)),
                    }
                )

    historic_rows = load_historic_consensus(repo_root)
    historic_path = run_root / "reports" / "HISTORIC_E1_CONSENSUS.csv"
    write_rows(historic_path, historic_rows)
    write_rows(run_root / "reports" / "BOOTSTRAP_SUMMARY.csv", summary_rows)
    write_rows(run_root / "reports" / "BOOTSTRAP_DRAWS.csv", draw_rows)
    write_rows(run_root / "reports" / "DOMAIN_SUMMARY.csv", domain_rows)

    n_pass = int(sum(time_passes))
    if n_pass >= 2:
        verdict = "SUPPORTED_LOCAL_VALIDATION"
    else:
        exposure_pass = sum(r["ci_low"] > 0 for r in summary_rows if r["contrast"] == "exposure_gap")
        clock_pass = sum(r["ci_low"] > 0 for r in summary_rows if r["contrast"] == "clock_gap")
        if exposure_pass >= 2 and clock_pass >= 2:
            verdict = "PARTIAL_DUAL_CONTROL"
        elif exposure_pass >= 2:
            verdict = "EXPOSURE_ONLY"
        else:
            verdict = "NOT_REPRODUCED_LOCAL"

    adjudication = {
        "protocol_id": "unsb-initial-shared-bridge-fanout-dual-control-v1",
        "new_seed": 2041,
        "verdict": verdict,
        "primary_time_passes": time_passes,
        "n_passing_bridge_times": n_pass,
        "summary": summary_rows,
        "historic_epoch1": {
            "seeds": list(range(2026, 2032)),
            "seed_count": 6,
            "seed_domain_positive": 30,
            "seed_domain_total": 30,
            "seed_time_ci_positive": sum(bool(r["ci_positive"]) for r in historic_rows),
            "seed_time_total": len(historic_rows),
        },
        "claim_boundary": {
            "allowed": "Initial shared-bridge fan-out under a fresh local seed, with exposure and optimizer-clock controls, if supported by the frozen gate.",
            "forbidden": [
                "causal harm",
                "calibrated posterior uncertainty",
                "all-stage fan-out",
                "training-seed confirmation from image bootstrap",
            ],
            "heldout_status": "HELDOUT_WITHIN_DISCOVERY_NOT_CONFIRMATORY",
        },
        "evidence_hashes": {
            "direction_statistics": sha256_file(raw_path),
            "paired_contrasts": sha256_file(contrast_path),
            "historic_consensus": sha256_file(historic_path),
        },
    }
    dump_json(run_root / "reports" / "ADJUDICATION.json", adjudication)
    print(json.dumps(adjudication, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
