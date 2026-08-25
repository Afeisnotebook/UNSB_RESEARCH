from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import DOMAINS, dump_json, sha256_file
except ImportError:  # direct script execution
    from common import DOMAINS, dump_json, sha256_file


PROTOCOL_ID = "unsb-shared-bridge-domain-phase-desynchronization-v1"


def effective_ages(arrays: list[np.ndarray]) -> np.ndarray:
    """Domain arrays are [images, ages]; np.argmin supplies the frozen lower-age tie rule."""
    return np.asarray([int(np.argmin(array.mean(axis=0))) + 1 for array in arrays], dtype=int)


def phase_variance(ages: np.ndarray) -> float:
    ages = np.asarray(ages, dtype=np.float64)
    return float(np.mean((ages - ages.mean()) ** 2))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    raw_path = run_root / "raw" / "RECIPROCAL_KERNEL_BY_AGE.csv"
    frame = pd.read_csv(raw_path)
    expected = len(DOMAINS) * 24 * 3 * 5
    if len(frame) != expected or set(frame["domain"]) != set(DOMAINS):
        raise RuntimeError(f"confirmation evidence identity mismatch: {len(frame)} != {expected}")

    cell_rows: list[dict] = []
    time_rows: list[dict] = []
    permutation_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    passing_times: list[bool] = []
    agreement_count = 0
    for t in (1, 2, 3):
        time_block = frame[frame["bridge_time_index"] == t]
        arrays = []
        arrays_m16 = []
        for domain in DOMAINS:
            block = time_block[time_block["domain"] == domain]
            pivot = block.pivot(index="stem", columns="single_epoch", values="reciprocal_KDD")
            pivot_m16 = block.pivot(index="stem", columns="single_epoch", values="reciprocal_KDD_M16")
            if pivot.shape != (24, 5) or list(pivot.columns) != [1, 2, 3, 4, 5]:
                raise RuntimeError(f"bad confirmation cell: {domain}/t{t}/{pivot.shape}")
            arrays.append(pivot.to_numpy(dtype=np.float64))
            arrays_m16.append(pivot_m16.to_numpy(dtype=np.float64))

        observed_ages = effective_ages(arrays)
        m16_ages = effective_ages(arrays_m16)
        agreement_count += int(np.sum(observed_ages == m16_ages))
        observed_t = phase_variance(observed_ages)

        stacked = np.concatenate(arrays, axis=0)
        rng_perm = np.random.default_rng(20260824 + t * 101)
        perm_stats = np.empty(args.permutations, dtype=np.float64)
        for draw in range(args.permutations):
            shuffled = stacked[rng_perm.permutation(len(stacked))]
            perm_arrays = [shuffled[index * 24 : (index + 1) * 24] for index in range(5)]
            perm_stats[draw] = phase_variance(effective_ages(perm_arrays))
        p_value = float((1 + np.sum(perm_stats >= observed_t)) / (args.permutations + 1))
        permutation_rows.extend(
            {"bridge_time_index": t, "draw": draw, "phase_variance": float(value)}
            for draw, value in enumerate(perm_stats)
        )

        rng_boot = np.random.default_rng(20260824 + t * 1009)
        boot_ages = np.empty((args.bootstrap_draws, len(DOMAINS)), dtype=int)
        for draw in range(args.bootstrap_draws):
            sampled = [
                array[rng_boot.integers(0, len(array), size=len(array))]
                for array in arrays
            ]
            boot_ages[draw] = effective_ages(sampled)
        modal_ages = []
        modal_shares = []
        for domain_index, domain in enumerate(DOMAINS):
            counts = np.bincount(boot_ages[:, domain_index], minlength=6)[1:]
            modal_age = int(np.argmax(counts)) + 1
            modal_share = float(counts[modal_age - 1] / args.bootstrap_draws)
            modal_ages.append(modal_age)
            modal_shares.append(modal_share)
            means = arrays[domain_index].mean(axis=0)
            order = np.argsort(means)
            cell_rows.append(
                {
                    "bridge_time_index": t,
                    "bridge_time_value": float(time_block["bridge_time_value"].iloc[0]),
                    "domain": domain,
                    "effective_age_M32": int(observed_ages[domain_index]),
                    "effective_age_M16": int(m16_ages[domain_index]),
                    "bootstrap_modal_age": modal_age,
                    "bootstrap_modal_share": modal_share,
                    "best_mean_KDD": float(means[order[0]]),
                    "second_best_mean_KDD": float(means[order[1]]),
                    "profile_margin": float(means[order[1]] - means[order[0]]),
                }
            )
        ranges = boot_ages.max(axis=1) - boot_ages.min(axis=1)
        range_ge2_fraction = float(np.mean(ranges >= 2))
        stable_domain_count = int(sum(share >= 0.80 for share in modal_shares))
        distinct_modal_ages = int(len(set(modal_ages)))
        passes = bool(
            p_value <= 0.01
            and stable_domain_count >= 4
            and distinct_modal_ages >= 3
            and range_ge2_fraction >= 0.90
        )
        passing_times.append(passes)
        time_rows.append(
            {
                "bridge_time_index": t,
                "bridge_time_value": float(time_block["bridge_time_value"].iloc[0]),
                "observed_phase_variance": observed_t,
                "permutation_p": p_value,
                "effective_ages_M32": "|".join(map(str, observed_ages.tolist())),
                "effective_ages_M16": "|".join(map(str, m16_ages.tolist())),
                "modal_ages": "|".join(map(str, modal_ages)),
                "stable_domain_count": stable_domain_count,
                "distinct_modal_ages": distinct_modal_ages,
                "bootstrap_range_ge2_fraction": range_ge2_fraction,
                "passes_frozen_time_gate": passes,
            }
        )
        bootstrap_rows.extend(
            {
                "bridge_time_index": t,
                "draw": draw,
                **{f"age_{domain}": int(boot_ages[draw, index]) for index, domain in enumerate(DOMAINS)},
                "age_range": int(ranges[draw]),
            }
            for draw in range(args.bootstrap_draws)
        )

    m16_gate = agreement_count >= 12
    n_pass = int(sum(passing_times))
    if n_pass >= 2 and m16_gate:
        verdict = "SUPPORTED_DOMAIN_PHASE_DESYNCHRONIZATION"
    elif n_pass >= 1:
        verdict = "PARTIAL_DOMAIN_PHASE_STRUCTURE"
    else:
        verdict = "NOT_REPRODUCED_ON_CONFIRMATION_SPLIT"

    reports = run_root / "reports"
    cell_path = reports / "PHASE_CELL_SUMMARY.csv"
    time_path = reports / "PHASE_TIME_ADJUDICATION.csv"
    perm_path = reports / "PHASE_PERMUTATION_DRAWS.csv"
    boot_path = reports / "PHASE_BOOTSTRAP_DRAWS.csv"
    write_csv(cell_path, cell_rows)
    write_csv(time_path, time_rows)
    write_csv(perm_path, permutation_rows)
    write_csv(boot_path, bootstrap_rows)
    adjudication = {
        "protocol_id": PROTOCOL_ID,
        "verdict": verdict,
        "n_passing_bridge_times": n_pass,
        "passing_bridge_times": passing_times,
        "M16_M32_agreement_count": agreement_count,
        "M16_M32_total_cells": 15,
        "M16_robustness_gate": m16_gate,
        "time_summary": time_rows,
        "claim_boundary": {
            "allowed_if_supported": "One AIO checkpoint occupies distinct task-specific conditional-kernel phases across weather domains at common bridge times.",
            "forbidden": [
                "causal restoration harm",
                "a calibrated uncertainty claim",
                "external or sealed confirmation",
                "proof that per-domain phase correction improves final quality",
            ],
        },
        "hashes": {
            raw_path.name: sha256_file(raw_path),
            cell_path.name: sha256_file(cell_path),
            time_path.name: sha256_file(time_path),
            perm_path.name: sha256_file(perm_path),
            boot_path.name: sha256_file(boot_path),
        },
    }
    dump_json(reports / "PHASE_CONFIRMATION_ADJUDICATION.json", adjudication)
    print(json.dumps(adjudication, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
