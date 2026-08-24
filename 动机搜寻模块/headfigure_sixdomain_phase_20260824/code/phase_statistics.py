from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def age_from_profile(profile: np.ndarray) -> int:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1 or not np.all(np.isfinite(profile)):
        raise ValueError("profile must be one-dimensional and finite")
    return int(np.argmin(profile))


def common_age(domain_profiles: np.ndarray) -> int:
    """Return the equal-domain clock minimizing mean within-domain regret.

    Subtracting each profile minimum does not change the minimizer, but makes
    the scientific meaning explicit: the common clock pays regret relative to
    each domain's own best task-specific phase.
    """
    profiles = np.asarray(domain_profiles, dtype=np.float64)
    if profiles.ndim != 2 or not np.all(np.isfinite(profiles)):
        raise ValueError("domain_profiles must be finite [domain, age]")
    regrets = profiles - profiles.min(axis=1, keepdims=True)
    return int(np.argmin(regrets.mean(axis=0)))


def in_sample_sync_regret(domain_profiles: np.ndarray) -> dict[str, float | int | list[int]]:
    profiles = np.asarray(domain_profiles, dtype=np.float64)
    clock = common_age(profiles)
    ages = np.argmin(profiles, axis=1)
    contributions = profiles[np.arange(len(profiles)), clock] - profiles[
        np.arange(len(profiles)), ages
    ]
    scale = float(np.mean(np.ptp(profiles, axis=1)))
    regret = float(np.mean(contributions))
    return {
        "common_age_zero_based": clock,
        "domain_ages_zero_based": ages.astype(int).tolist(),
        "regret": regret,
        "profile_range_scale": scale,
        "normalized_regret": float(regret / max(scale, EPS)),
    }


def two_fold_sync_regret(
    arrays: dict[str, np.ndarray],
    fold_indices: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    rng: np.random.Generator | None = None,
    resample: bool = False,
) -> tuple[float, dict[str, float], list[dict]]:
    """Cross-fit domain clocks and the shared clock on opposite image folds.

    Each domain array is [image, age].  For A->B and B->A, clocks are selected
    only on the training fold and regret is evaluated only on the other fold.
    The returned value averages both directions and domains.
    """
    if resample and rng is None:
        raise ValueError("resampling requires an RNG")
    domains = list(arrays)
    contributions = {domain: [] for domain in domains}
    direction_rows: list[dict] = []
    for direction, train_slot in (("A_to_B", 0), ("B_to_A", 1)):
        train_profiles = []
        eval_profiles = []
        domain_ages = []
        for domain in domains:
            array = np.asarray(arrays[domain], dtype=np.float64)
            train_idx = np.asarray(fold_indices[domain][train_slot], dtype=int)
            eval_idx = np.asarray(fold_indices[domain][1 - train_slot], dtype=int)
            if resample:
                train_idx = train_idx[rng.integers(0, len(train_idx), size=len(train_idx))]
                eval_idx = eval_idx[rng.integers(0, len(eval_idx), size=len(eval_idx))]
            train_profile = array[train_idx].mean(axis=0)
            eval_profile = array[eval_idx].mean(axis=0)
            train_profiles.append(train_profile)
            eval_profiles.append(eval_profile)
            domain_ages.append(age_from_profile(train_profile))
        train_profiles_array = np.stack(train_profiles)
        eval_profiles_array = np.stack(eval_profiles)
        shared_age = common_age(train_profiles_array)
        for domain_index, domain in enumerate(domains):
            domain_age = domain_ages[domain_index]
            value = float(
                eval_profiles_array[domain_index, shared_age]
                - eval_profiles_array[domain_index, domain_age]
            )
            contributions[domain].append(value)
            direction_rows.append(
                {
                    "direction": direction,
                    "domain": domain,
                    "shared_age_zero_based": shared_age,
                    "domain_age_zero_based": domain_age,
                    "heldout_regret": value,
                }
            )
    per_domain = {domain: float(np.mean(values)) for domain, values in contributions.items()}
    return float(np.mean(list(per_domain.values()))), per_domain, direction_rows


def bootstrap_phase_distributions(
    arrays: dict[str, np.ndarray], *, draws: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    phase_draws: dict[str, np.ndarray] = {}
    n_ages = next(iter(arrays.values())).shape[1]
    for domain, array in arrays.items():
        samples = np.empty(draws, dtype=np.int64)
        for draw in range(draws):
            indices = rng.integers(0, len(array), size=len(array))
            samples[draw] = age_from_profile(array[indices].mean(axis=0))
        phase_draws[domain] = samples

    # In one dimension the W2 barycenter is obtained by averaging quantile
    # functions.  Sorting each bootstrap phase panel evaluates those quantiles
    # on a common grid without a temperature or a soft-min hyperparameter.
    quantiles = np.stack([np.sort(values) for values in phase_draws.values()])
    barycenter = quantiles.mean(axis=0)
    energy = float(np.mean((quantiles - barycenter[None, :]) ** 2))
    max_variance = ((n_ages - 1) ** 2) / 4.0
    means = np.array([values.mean() for values in phase_draws.values()], dtype=np.float64)
    within = float(np.mean([values.var() for values in phase_draws.values()]))
    between = float(means.var())
    summary = {
        "wasserstein_barycenter_energy": energy,
        "normalized_wasserstein_energy": float(energy / max(max_variance, EPS)),
        "between_domain_phase_variance": between,
        "mean_bootstrap_phase_variance": within,
        "phase_reliability_ratio": float(between / max(between + within, EPS)),
    }
    return phase_draws, summary


def load_frames(paths: Iterable[Path], labels: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path, label in zip(paths, labels, strict=True):
        frame = pd.read_csv(path)
        frame["evidence_split"] = label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    required = {
        "domain",
        "stem",
        "bridge_time_index",
        "bridge_time_value",
        "single_epoch",
        "reciprocal_KDD",
    }
    missing = required - set(combined.columns)
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")
    key = ["domain", "stem", "bridge_time_index", "single_epoch"]
    duplicates = combined.duplicated(key, keep=False)
    if duplicates.any():
        examples = combined.loc[duplicates, key].head().to_dict("records")
        raise RuntimeError(f"duplicate image/time/age identities across inputs: {examples}")
    if not np.isfinite(combined["reciprocal_KDD"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("non-finite reciprocal_KDD")
    return combined


def make_arrays(
    frame: pd.DataFrame,
    domains: list[str],
    time_index: int,
    ages: list[int],
    *,
    value_column: str = "reciprocal_KDD",
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    arrays: dict[str, np.ndarray] = {}
    stems: dict[str, list[str]] = {}
    for domain in domains:
        block = frame[
            (frame["domain"] == domain) & (frame["bridge_time_index"] == time_index)
        ]
        pivot = block.pivot(index="stem", columns="single_epoch", values=value_column)
        if list(pivot.columns.astype(int)) != ages:
            raise RuntimeError(f"{domain}/t{time_index}: ages {list(pivot.columns)} != {ages}")
        if pivot.isna().any().any():
            raise RuntimeError(f"{domain}/t{time_index}: incomplete KDD profile")
        arrays[domain] = pivot.to_numpy(dtype=np.float64)
        stems[domain] = pivot.index.astype(str).tolist()
    counts = {domain: len(array) for domain, array in arrays.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"unbalanced image counts: {counts}")
    return arrays, stems


def frozen_folds(stems: dict[str, list[str]], protocol_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for domain, identities in stems.items():
        ranked = sorted(
            range(len(identities)),
            key=lambda index: hashlib.sha256(
                f"{protocol_id}|crossfit|{domain}|{identities[index]}".encode("utf-8")
            ).hexdigest(),
        )
        midpoint = len(ranked) // 2
        if midpoint < 2 or len(ranked) - midpoint < 2:
            raise RuntimeError(f"too few images for two-fold cross-fitting: {domain}")
        result[domain] = (
            np.asarray(ranked[:midpoint], dtype=int),
            np.asarray(ranked[midpoint:], dtype=int),
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", action="append", required=True)
    parser.add_argument("--split-label", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if len(args.input_csv) != len(args.split_label):
        raise ValueError("--input-csv and --split-label counts differ")

    paths = [Path(value).resolve() for value in args.input_csv]
    labels = list(args.split_label)
    frame = load_frames(paths, labels)
    domains = args.domains or sorted(frame["domain"].unique().tolist())
    observed_domains = sorted(frame["domain"].unique().tolist())
    if sorted(domains) != observed_domains:
        raise RuntimeError(f"domain mismatch: requested={domains}, observed={observed_domains}")
    ages = sorted(frame["single_epoch"].astype(int).unique().tolist())
    times = sorted(frame["bridge_time_index"].astype(int).unique().tolist())
    output_dir = Path(args.output_dir).resolve()

    cell_rows: list[dict] = []
    crossfit_rows: list[dict] = []
    draw_rows: list[dict] = []
    time_results = []
    phase_mean_matrix = np.empty((len(domains), len(times)), dtype=np.float64)
    has_m16 = "reciprocal_KDD_M16" in frame and frame["reciprocal_KDD_M16"].notna().all()
    m16_m32_agreement = 0

    for time_position, time_index in enumerate(times):
        arrays, stems = make_arrays(frame, domains, time_index, ages)
        arrays_m16 = (
            make_arrays(
                frame,
                domains,
                time_index,
                ages,
                value_column="reciprocal_KDD_M16",
            )[0]
            if has_m16
            else None
        )
        folds = frozen_folds(stems, args.protocol_id)
        full_profiles = np.stack([arrays[domain].mean(axis=0) for domain in domains])
        in_sample = in_sample_sync_regret(full_profiles)
        crossfit, per_domain, direction_rows = two_fold_sync_regret(arrays, folds)

        rng = np.random.default_rng(stable_seed(args.protocol_id, time_index, "crossfit-bootstrap"))
        draws = np.empty(args.bootstrap_draws, dtype=np.float64)
        for draw in range(args.bootstrap_draws):
            value, _, _ = two_fold_sync_regret(
                arrays, folds, rng=rng, resample=True
            )
            draws[draw] = value
            draw_rows.append(
                {
                    "bridge_time_index": time_index,
                    "draw": draw,
                    "crossfit_sync_regret": value,
                }
            )

        phase_draws, phase_summary = bootstrap_phase_distributions(
            arrays,
            draws=args.bootstrap_draws,
            seed=stable_seed(args.protocol_id, time_index, "phase-bootstrap"),
        )
        for domain_index, domain in enumerate(domains):
            profile = full_profiles[domain_index]
            effective_age_m16 = None
            if arrays_m16 is not None:
                effective_age_m16 = ages[age_from_profile(arrays_m16[domain].mean(axis=0))]
                m16_m32_agreement += int(effective_age_m16 == ages[int(np.argmin(profile))])
            phase_values = phase_draws[domain]
            counts = np.bincount(phase_values, minlength=len(ages))
            phase_mode_index = int(np.argmax(counts))
            phase_mean_matrix[domain_index, time_position] = float(phase_values.mean() + 1)
            order = np.argsort(profile)
            cell_rows.append(
                {
                    "bridge_time_index": time_index,
                    "bridge_time_value": float(
                        frame.loc[
                            frame["bridge_time_index"] == time_index, "bridge_time_value"
                        ].iloc[0]
                    ),
                    "domain": domain,
                    "n_images": len(arrays[domain]),
                    "effective_age": ages[int(order[0])],
                    "effective_age_M16": effective_age_m16,
                    "second_best_age": ages[int(order[1])],
                    "profile_margin": float(profile[order[1]] - profile[order[0]]),
                    "bootstrap_phase_mean": float(phase_values.mean() + 1),
                    "bootstrap_phase_sd": float(phase_values.std()),
                    "bootstrap_phase_mode": ages[phase_mode_index],
                    "bootstrap_modal_share": float(counts[phase_mode_index] / len(phase_values)),
                    "crossfit_domain_regret": per_domain[domain],
                }
            )
        for row in direction_rows:
            crossfit_rows.append(
                {
                    "bridge_time_index": time_index,
                    **row,
                    "shared_age": ages[int(row["shared_age_zero_based"])],
                    "domain_age": ages[int(row["domain_age_zero_based"])],
                }
            )

        profile_scale = float(in_sample["profile_range_scale"])
        result = {
            "bridge_time_index": time_index,
            "bridge_time_value": float(
                frame.loc[frame["bridge_time_index"] == time_index, "bridge_time_value"].iloc[0]
            ),
            "n_images_per_domain": int(next(iter(arrays.values())).shape[0]),
            "domain_effective_ages": {
                domain: ages[int(in_sample["domain_ages_zero_based"][index])]
                for index, domain in enumerate(domains)
            },
            "best_common_age": ages[int(in_sample["common_age_zero_based"])],
            "in_sample_sync_regret": float(in_sample["regret"]),
            "in_sample_normalized_regret": float(in_sample["normalized_regret"]),
            "crossfit_sync_regret": crossfit,
            "crossfit_normalized_regret": float(crossfit / max(profile_scale, EPS)),
            "bootstrap_ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "bootstrap_one_sided_95_lower": float(np.quantile(draws, 0.05)),
            "bootstrap_positive_fraction": float(np.mean(draws > 0)),
            **phase_summary,
        }
        time_results.append(result)

    slopes = np.diff(phase_mean_matrix, axis=1)
    shear_energy = float(np.mean(np.var(slopes, axis=0))) if slopes.size else 0.0
    requirements = {
        "crossfit_regret_positive_all_times": all(
            result["crossfit_sync_regret"] > 0 for result in time_results
        ),
        "one_sided_95_lower_positive_at_least_two_times": sum(
            result["bootstrap_one_sided_95_lower"] > 0 for result in time_results
        )
        >= 2,
        "bootstrap_positive_fraction_ge_0_95_at_least_two_times": sum(
            result["bootstrap_positive_fraction"] >= 0.95 for result in time_results
        )
        >= 2,
        "at_least_three_distinct_domain_phases_at_least_two_times": sum(
            len(set(result["domain_effective_ages"].values())) >= 3
            for result in time_results
        )
        >= 2,
    }
    if all(requirements.values()):
        verdict = "SUPPORTED_SIXDOMAIN_SHARED_CLOCK_REGRET"
    elif sum(requirements.values()) >= 2:
        verdict = "PARTIAL_SIXDOMAIN_PHASE_STRUCTURE"
    else:
        verdict = "NOT_REPLICATED_SIXDOMAIN"
    payload = {
        "protocol_id": args.protocol_id,
        "status": "complete",
        "verdict": verdict,
        "statistic_name_cn": "交叉拟合共享时钟遗憾与相位 Wasserstein 离散",
        "statistic_name_en": "cross-fitted shared-clock regret and phase-Wasserstein dispersion",
        "domains": domains,
        "ages": ages,
        "input_splits": [
            {"label": label, "path": str(path), "sha256": sha256_file(path)}
            for path, label in zip(paths, labels, strict=True)
        ],
        "row_count": int(len(frame)),
        "unique_images_per_domain": {
            domain: int(frame.loc[frame["domain"] == domain, "stem"].nunique())
            for domain in domains
        },
        "time_results": time_results,
        "domain_time_phase_shear_energy": shear_energy,
        "M16_available": has_m16,
        "M16_M32_effective_age_agreement": m16_m32_agreement if has_m16 else None,
        "M16_M32_total_cells": len(domains) * len(times) if has_m16 else None,
        "requirements": requirements,
        "interpretation": {
            "sync_regret": "Held-out excess reciprocal KDD paid by a single shared task phase relative to clocks selected separately for each domain.",
            "wasserstein": "Distance among bootstrap phase distributions; it uses the full sampling uncertainty of the argmin and has no soft-min temperature.",
            "shear": "Across-domain variance of phase increments over adjacent bridge-time observations.",
        },
        "claim_boundary": {
            "allowed": "A positive cross-fitted regret means a common phase is statistically less compatible with the observed domain-specific conditional-kernel trajectories than domain-specific phases.",
            "forbidden": [
                "causal restoration harm",
                "calibrated posterior uncertainty",
                "method effectiveness",
                "external or multi-training-seed confirmation",
            ],
        },
    }
    write_json(output_dir / "PHASE_STATISTICS.json", payload)
    write_csv(output_dir / "PHASE_CELL_SUMMARY.csv", cell_rows)
    write_csv(output_dir / "CROSSFIT_DIRECTION_DETAILS.csv", crossfit_rows)
    write_csv(output_dir / "CROSSFIT_BOOTSTRAP_DRAWS.csv", draw_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
