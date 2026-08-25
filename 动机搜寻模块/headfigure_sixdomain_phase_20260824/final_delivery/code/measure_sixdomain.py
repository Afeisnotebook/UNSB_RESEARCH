from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .common import DOMAINS, bridge_times, dump_json, sha256_file, stable_seed
except ImportError:
    from common import DOMAINS, bridge_times, dump_json, sha256_file, stable_seed


PROTOCOL_ID = "unsb-sixdomain-expanded-phase-field-v1"
SEED = 2051
N_AGES = 6
EPS = 1e-12


def import_helpers(repo_root: Path):
    geometry_root = repo_root / "动机搜寻模块" / "code"
    legacy_root = (
        repo_root
        / "动机搜寻模块"
        / "headfigure_local_20260824"
        / "final_delivery"
        / "code"
    )
    sys.path.insert(0, str(geometry_root))
    sys.path.insert(0, str(legacy_root))
    from measure_path_geometry import bridge_state, build_generator, load_image
    from measure_reciprocal import (
        cosine_distance,
        mean_direction_and_floor,
        proposal_panel,
        single_age_span,
    )

    return (
        bridge_state,
        build_generator,
        load_image,
        cosine_distance,
        mean_direction_and_floor,
        proposal_panel,
        single_age_span,
    )


def make_panel(domain: str, stem: str, time_index: int, m: int, device: str) -> torch.Tensor:
    seed = stable_seed(PROTOCOL_ID, domain, stem, time_index, "proposal")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(m, 256, generator=generator, dtype=torch.float32).to(device)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pca_rows(
    domain: str,
    stem: str,
    time_index: int,
    aio_mean: torch.Tensor,
    single_means: list[torch.Tensor],
) -> list[dict]:
    labels = ["AIO e1", *[f"Single e{age}" for age in range(1, N_AGES + 1)]]
    vectors = torch.stack([aio_mean, *single_means]).cpu().numpy().astype(np.float64)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    coordinates = eigenvectors[:, order[:2]] * np.sqrt(eigenvalues[:2])[None, :]
    return [
        {
            "protocol_id": PROTOCOL_ID,
            "domain": domain,
            "stem": stem,
            "bridge_time_index": time_index,
            "role": label,
            "pca1": float(xy[0]),
            "pca2": float(xy[1]),
            "eigenvalue_1": float(eigenvalues[0]),
            "eigenvalue_2": float(eigenvalues[1]),
        }
        for label, xy in zip(labels, coordinates, strict=True)
    ]


def shard_valid(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return len(frame) == expected_rows and np.isfinite(
        frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    ).all()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
    if args.m < 4 or args.m % 2:
        raise ValueError("M must be even and >=4")
    run_root = Path(args.run_root).resolve()
    path_map = json.loads((run_root / "state" / "PATH_MAP.json").read_text(encoding="utf-8"))
    heldout = json.loads((run_root / "state" / "HELDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    if any("target" in key.casefold() for row in heldout for key in row):
        raise RuntimeError("heldout manifest must not expose target paths")
    repo_root = Path(path_map["repo_root"])
    baseline_root = Path(path_map["baseline_root"])
    checkpoints_root = Path(path_map["checkpoints_root"])
    (
        bridge_state_fn,
        build_generator,
        load_image_fn,
        cosine_distance,
        mean_direction_and_floor,
        proposal_panel,
        single_age_span,
    ) = import_helpers(repo_root)

    if args.device.startswith("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()

    times = bridge_times(5)
    per_domain_count = {
        domain: sum(row["domain"] == domain for row in heldout) for domain in DOMAINS
    }
    if set(per_domain_count.values()) != {80}:
        raise RuntimeError(f"expected 80 heldout images per domain: {per_domain_count}")
    aio_checkpoint = checkpoints_root / f"aio6_plain_s{SEED}_e1_n120" / "1_net_G.pth"
    aio_hash = sha256_file(aio_checkpoint)
    shard_root = run_root / "raw" / "shards"
    checkpoint_hashes = {"aio_e1": aio_hash}
    state_path = run_root / "state" / "MEASUREMENT_STATE.json"
    state = {
        "status": "running",
        "protocol_id": PROTOCOL_ID,
        "M": args.m,
        "domains_complete": [],
        "started": time.time(),
        "target_content_read": False,
    }
    dump_json(state_path, state)

    for domain_index, domain in enumerate(DOMAINS):
        age_shard = shard_root / f"{domain}_KDD_BY_AGE.csv"
        primary_shard = shard_root / f"{domain}_PRIMARY.csv"
        pca_shard = shard_root / f"{domain}_PCA.csv"
        if (
            shard_valid(age_shard, 80 * 3 * N_AGES)
            and shard_valid(primary_shard, 80 * 3)
            and shard_valid(pca_shard, N_AGES + 1)
        ):
            state["domains_complete"].append({"domain": domain, "status": "cached"})
            dump_json(state_path, state)
            continue

        domain_started = time.time()
        images = [row for row in heldout if row["domain"] == domain]
        aio_net = build_generator(str(aio_checkpoint), str(baseline_root), args.device)
        cache: dict[tuple[str, int], dict] = {}
        for image_row in images:
            source = load_image_fn(image_row["source_path"], 128, args.device)
            for time_index in (1, 2, 3):
                proposals = make_panel(domain, image_row["stem"], time_index, args.m, args.device)
                rollout_seed = stable_seed(
                    PROTOCOL_ID, domain, image_row["stem"], time_index, "rollout"
                )
                aio_state = bridge_state_fn(
                    aio_net,
                    source,
                    time_index,
                    proposals,
                    tau=0.01,
                    times=times,
                    rollout_seed=rollout_seed,
                )
                aio_panel = proposal_panel(
                    aio_net, aio_state, time_index, proposals, chunk_size=args.chunk_size
                )
                aio_mean, aio_floor = mean_direction_and_floor(aio_panel)
                aio_mean_m16, aio_floor_m16 = mean_direction_and_floor(aio_panel[:16])
                cache[(image_row["stem"], time_index)] = {
                    "source": source.detach(),
                    "proposals": proposals,
                    "rollout_seed": rollout_seed,
                    "aio_state": aio_state.detach(),
                    "aio_mean": aio_mean.cpu(),
                    "aio_floor": aio_floor,
                    "aio_mean_m16": aio_mean_m16.cpu(),
                    "aio_floor_m16": aio_floor_m16,
                    "single_means_on_aio": [],
                }

        age_rows: list[dict] = []
        rows_by_key: dict[tuple[str, int], list[dict]] = {
            (row["stem"], time_index): [] for row in images for time_index in (1, 2, 3)
        }
        for age in range(1, N_AGES + 1):
            single_checkpoint = (
                checkpoints_root
                / f"single6_{domain}_s{SEED}_e{N_AGES}"
                / f"{age}_net_G.pth"
            )
            single_hash = sha256_file(single_checkpoint)
            checkpoint_hashes[f"single_{domain}_e{age}"] = single_hash
            single_net = build_generator(str(single_checkpoint), str(baseline_root), args.device)
            for image_row in images:
                stem = image_row["stem"]
                for time_index in (1, 2, 3):
                    entry = cache[(stem, time_index)]
                    single_state = bridge_state_fn(
                        single_net,
                        entry["source"],
                        time_index,
                        entry["proposals"],
                        tau=0.01,
                        times=times,
                        rollout_seed=entry["rollout_seed"],
                    )
                    single_on_aio = proposal_panel(
                        single_net,
                        entry["aio_state"],
                        time_index,
                        entry["proposals"],
                        chunk_size=args.chunk_size,
                    )
                    aio_on_single = proposal_panel(
                        aio_net,
                        single_state,
                        time_index,
                        entry["proposals"],
                        chunk_size=args.chunk_size,
                    )
                    single_on_single = proposal_panel(
                        single_net,
                        single_state,
                        time_index,
                        entry["proposals"],
                        chunk_size=args.chunk_size,
                    )
                    single_mean_aio, floor_single_aio = mean_direction_and_floor(single_on_aio)
                    aio_mean_single, floor_aio_single = mean_direction_and_floor(aio_on_single)
                    single_mean_single, floor_single_single = mean_direction_and_floor(single_on_single)
                    single_mean_aio_m16, floor_single_aio_m16 = mean_direction_and_floor(
                        single_on_aio[:16]
                    )
                    aio_mean_single_m16, floor_aio_single_m16 = mean_direction_and_floor(
                        aio_on_single[:16]
                    )
                    single_mean_single_m16, floor_single_single_m16 = mean_direction_and_floor(
                        single_on_single[:16]
                    )
                    left = cosine_distance(entry["aio_mean"], single_mean_aio)
                    right = cosine_distance(aio_mean_single, single_mean_single)
                    left_m16 = cosine_distance(entry["aio_mean_m16"], single_mean_aio_m16)
                    right_m16 = cosine_distance(aio_mean_single_m16, single_mean_single_m16)
                    entry["single_means_on_aio"].append(single_mean_aio.cpu())
                    row = {
                        "protocol_id": PROTOCOL_ID,
                        "domain": domain,
                        "stem": stem,
                        "bridge_time_index": time_index,
                        "bridge_time_value": float(times[time_index]),
                        "single_epoch": age,
                        "reciprocal_KDD": 0.5 * (left + right),
                        "reciprocal_KDD_M16": 0.5 * (left_m16 + right_m16),
                        "aio_anchor_deflection": left,
                        "single_anchor_deflection": right,
                        "proposal_floor": max(
                            entry["aio_floor"],
                            floor_single_aio,
                            floor_aio_single,
                            floor_single_single,
                        ),
                        "proposal_floor_M16": max(
                            entry["aio_floor_m16"],
                            floor_single_aio_m16,
                            floor_aio_single_m16,
                            floor_single_single_m16,
                        ),
                        "aio_checkpoint_sha256": aio_hash,
                        "single_checkpoint_sha256": single_hash,
                        "image_sha256": image_row["sha256"],
                        "proposal_seed": stable_seed(
                            PROTOCOL_ID, domain, stem, time_index, "proposal"
                        ),
                        "rollout_seed": entry["rollout_seed"],
                        "M": args.m,
                        "target_content_read": False,
                    }
                    age_rows.append(row)
                    rows_by_key[(stem, time_index)].append(row)
            del single_net
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        primary_rows: list[dict] = []
        representative_rows: list[dict] = []
        for image_row in images:
            stem = image_row["stem"]
            for time_index in (1, 2, 3):
                entry = cache[(stem, time_index)]
                rows = rows_by_key[(stem, time_index)]
                if len(rows) != N_AGES or len(entry["single_means_on_aio"]) != N_AGES:
                    raise RuntimeError(f"incomplete trajectory: {domain}/{stem}/t{time_index}")
                best = min(rows, key=lambda row: row["reciprocal_KDD"])
                age_span = single_age_span(entry["single_means_on_aio"])
                max_floor = max(row["proposal_floor"] for row in rows)
                reference = max(max_floor, age_span)
                primary_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "domain": domain,
                        "stem": stem,
                        "bridge_time_index": time_index,
                        "bridge_time_value": float(times[time_index]),
                        "best_single_epoch": best["single_epoch"],
                        "min_reciprocal_KDD": best["reciprocal_KDD"],
                        "clock_matched_KDD_e6": next(
                            row["reciprocal_KDD"] for row in rows if row["single_epoch"] == 6
                        ),
                        "max_proposal_floor": max_floor,
                        "single_age_span": age_span,
                        "primary_reference": reference,
                        "log10_deflection_excess": math.log10(best["reciprocal_KDD"] + EPS)
                        - math.log10(reference + EPS),
                        "above_reference": best["reciprocal_KDD"] > reference,
                        "image_sha256": image_row["sha256"],
                        "target_content_read": False,
                    }
                )
                if image_row["representative"] and time_index == 2:
                    representative_rows.extend(
                        pca_rows(
                            domain,
                            stem,
                            time_index,
                            entry["aio_mean"],
                            entry["single_means_on_aio"],
                        )
                    )
        write_csv(age_shard, age_rows)
        write_csv(primary_shard, primary_rows)
        write_csv(pca_shard, representative_rows)
        del aio_net, cache
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        state["domains_complete"].append(
            {
                "domain": domain,
                "status": "complete",
                "elapsed_seconds": time.time() - domain_started,
                "age_rows": len(age_rows),
            }
        )
        dump_json(state_path, state)
        print(
            f"completed {domain_index + 1}/{len(DOMAINS)} {domain} "
            f"in {(time.time() - domain_started) / 60:.1f} min",
            flush=True,
        )

    def combine(suffix: str, destination: str) -> Path:
        frames = [pd.read_csv(shard_root / f"{domain}_{suffix}.csv") for domain in DOMAINS]
        output = run_root / "raw" / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(output, index=False)
        return output

    age_path = combine("KDD_BY_AGE", "RECIPROCAL_KERNEL_BY_AGE.csv")
    primary_path = combine("PRIMARY", "RECIPROCAL_KERNEL_PRIMARY.csv")
    pca_path = combine("PCA", "RECIPROCAL_REPRESENTATIVE_PCA.csv")
    age_frame = pd.read_csv(age_path)
    primary_frame = pd.read_csv(primary_path)
    pca_frame = pd.read_csv(pca_path)
    expected_age = len(DOMAINS) * 80 * 3 * N_AGES
    expected_primary = len(DOMAINS) * 80 * 3
    expected_pca = len(DOMAINS) * (N_AGES + 1)
    state.update(
        {
            "status": "complete",
            "finished": time.time(),
            "age_rows": len(age_frame),
            "expected_age_rows": expected_age,
            "primary_rows": len(primary_frame),
            "expected_primary_rows": expected_primary,
            "pca_rows": len(pca_frame),
            "expected_pca_rows": expected_pca,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda")
            else 0,
            "checkpoint_hashes": checkpoint_hashes,
            "hashes": {
                age_path.name: sha256_file(age_path),
                primary_path.name: sha256_file(primary_path),
                pca_path.name: sha256_file(pca_path),
            },
        }
    )
    if len(age_frame) != expected_age or len(primary_frame) != expected_primary or len(pca_frame) != expected_pca:
        state["status"] = "HOLD"
    dump_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
