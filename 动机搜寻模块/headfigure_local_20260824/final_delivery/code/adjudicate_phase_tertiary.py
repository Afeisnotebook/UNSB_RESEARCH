from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import DOMAINS, dump_json, sha256_file
except ImportError:  # direct execution
    from common import DOMAINS, dump_json, sha256_file


PROTOCOL_ID = "unsb-domain-phase-mapping-reconfirmation-v1"
FIXED = {
    1: np.asarray([4, 3, 2, 4, 5], dtype=int),
    2: np.asarray([4, 3, 2, 4, 5], dtype=int),
    3: np.asarray([4, 3, 2, 2, 5], dtype=int),
}


def mapping_accuracy(observed: dict[int, np.ndarray], predicted: dict[int, np.ndarray]) -> int:
    return int(sum(np.sum(observed[t] == predicted[t]) for t in (1, 2, 3)))


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
    root = Path(args.run_root).resolve()
    raw_path = root / "raw" / "RECIPROCAL_KERNEL_BY_AGE.csv"
    frame = pd.read_csv(raw_path)
    expected_rows = len(DOMAINS) * 16 * 3 * 5
    if len(frame) != expected_rows:
        raise RuntimeError(f"tertiary row mismatch: {len(frame)} != {expected_rows}")

    observed: dict[int, np.ndarray] = {}
    observed_m16: dict[int, np.ndarray] = {}
    cell_rows = []
    bootstrap_stable = 0
    m16_agree = 0
    time_structure_pass = []
    for t in (1, 2, 3):
        ages = []
        ages_m16 = []
        for domain_index, domain in enumerate(DOMAINS):
            block = frame[(frame["bridge_time_index"] == t) & (frame["domain"] == domain)]
            pivot = block.pivot(index="stem", columns="single_epoch", values="reciprocal_KDD")
            pivot16 = block.pivot(index="stem", columns="single_epoch", values="reciprocal_KDD_M16")
            if pivot.shape != (16, 5):
                raise RuntimeError(f"bad tertiary cell: {domain}/t{t}/{pivot.shape}")
            array = pivot.to_numpy(dtype=np.float64)
            array16 = pivot16.to_numpy(dtype=np.float64)
            age = int(np.argmin(array.mean(axis=0))) + 1
            age16 = int(np.argmin(array16.mean(axis=0))) + 1
            ages.append(age)
            ages_m16.append(age16)
            m16_agree += int(age == age16)

            rng = np.random.default_rng(20260824 + t * 100 + domain_index)
            boot = np.empty(args.bootstrap_draws, dtype=int)
            for draw in range(args.bootstrap_draws):
                sample = array[rng.integers(0, len(array), size=len(array))]
                boot[draw] = int(np.argmin(sample.mean(axis=0))) + 1
            counts = np.bincount(boot, minlength=6)[1:]
            mode = int(np.argmax(counts)) + 1
            share = float(counts[mode - 1] / args.bootstrap_draws)
            predicted = int(FIXED[t][domain_index])
            cell_stable = bool(mode == predicted and share >= 0.80)
            bootstrap_stable += int(cell_stable)
            means = array.mean(axis=0)
            order = np.argsort(means)
            cell_rows.append(
                {
                    "bridge_time_index": t,
                    "bridge_time_value": float(block["bridge_time_value"].iloc[0]),
                    "domain": domain,
                    "fixed_predicted_age": predicted,
                    "effective_age_M32": age,
                    "effective_age_M16": age16,
                    "bootstrap_modal_age": mode,
                    "bootstrap_modal_share": share,
                    "stable_predicted_cell": cell_stable,
                    "best_mean_KDD": float(means[order[0]]),
                    "second_best_mean_KDD": float(means[order[1]]),
                    "profile_margin": float(means[order[1]] - means[order[0]]),
                }
            )
        observed[t] = np.asarray(ages, dtype=int)
        observed_m16[t] = np.asarray(ages_m16, dtype=int)
        time_structure_pass.append(len(set(ages)) >= 3 and max(ages) - min(ages) >= 2)

    accuracy = mapping_accuracy(observed, FIXED)
    rng_perm = np.random.default_rng(20260824)
    perm_rows = []
    perm_accuracy = np.empty(args.permutations, dtype=int)
    for draw in range(args.permutations):
        prediction = {t: rng_perm.permutation(FIXED[t]) for t in (1, 2, 3)}
        value = mapping_accuracy(observed, prediction)
        perm_accuracy[draw] = value
        perm_rows.append({"draw": draw, "mapping_accuracy": int(value)})
    p_value = float((1 + np.sum(perm_accuracy >= accuracy)) / (args.permutations + 1))

    requirements = {
        "mapping_accuracy_ge_14": accuracy >= 14,
        "mapping_permutation_p_le_0_001": p_value <= 0.001,
        "stable_predicted_cells_ge_13": bootstrap_stable >= 13,
        "M16_M32_agreement_ge_14": m16_agree >= 14,
        "all_times_structured": all(time_structure_pass),
    }
    n_pass = sum(bool(value) for value in requirements.values())
    if all(requirements.values()):
        verdict = "SUPPORTED_DOMAIN_PHASE_MAPPING_REPLICATION"
    elif n_pass >= 3:
        verdict = "PARTIAL_PHASE_MAPPING_REPLICATION"
    else:
        verdict = "NOT_REPLICATED_TERTIARY_SPLIT"

    cell_path = root / "reports" / "TERTIARY_PHASE_CELL_SUMMARY.csv"
    perm_path = root / "reports" / "TERTIARY_MAPPING_PERMUTATION_DRAWS.csv"
    write_csv(cell_path, cell_rows)
    write_csv(perm_path, perm_rows)
    adjudication = {
        "protocol_id": PROTOCOL_ID,
        "verdict": verdict,
        "fixed_prediction": {str(t): FIXED[t].tolist() for t in (1, 2, 3)},
        "observed_M32": {str(t): observed[t].tolist() for t in (1, 2, 3)},
        "observed_M16": {str(t): observed_m16[t].tolist() for t in (1, 2, 3)},
        "mapping_accuracy": accuracy,
        "mapping_total_cells": 15,
        "permutation_p": p_value,
        "stable_predicted_cells": bootstrap_stable,
        "M16_M32_agreement_cells": m16_agree,
        "time_structure_pass": time_structure_pass,
        "requirements": requirements,
        "claim_boundary": {
            "allowed_if_supported": "A fixed domain-to-task-specific conditional-kernel-age map replicated across three zero-overlap image splits under one training seed.",
            "forbidden": [
                "causal quality harm",
                "multi-training-seed confirmation",
                "external/sealed confirmation",
                "calibrated uncertainty",
            ],
        },
        "hashes": {
            raw_path.name: sha256_file(raw_path),
            cell_path.name: sha256_file(cell_path),
            perm_path.name: sha256_file(perm_path),
        },
    }
    dump_json(root / "reports" / "TERTIARY_PHASE_ADJUDICATION.json", adjudication)
    print(json.dumps(adjudication, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
