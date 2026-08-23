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
import torch

try:
    from .common import DOMAINS, bridge_times, dump_json, sha256_file, stable_seed
except ImportError:  # direct script execution
    from common import DOMAINS, bridge_times, dump_json, sha256_file, stable_seed


PROTOCOL_ID = "unsb-reciprocal-bridge-kernel-deflection-v1"
EPS = 1e-12


def import_measurement_helpers(repo_root: Path):
    code_root = repo_root / "动机搜寻模块" / "code"
    sys.path.insert(0, str(code_root))
    from measure_path_geometry import bridge_state, build_generator, load_image

    return bridge_state, build_generator, load_image


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    flat = vector.reshape(-1).to(dtype=torch.float64)
    return flat / flat.norm().clamp_min(EPS)


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = unit_vector(left)
    # Cached AIO centroids live on CPU while the active Single panel is on
    # CUDA.  The scientific quantity is device-independent; co-locate the
    # second normalized vector before the dot product.
    right = unit_vector(right).to(left.device)
    return float(torch.clamp(1.0 - torch.dot(left, right), min=0.0, max=2.0).item())


def mean_direction_and_floor(panel: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return unit centroid and odd/even split stability floor for [M,C,H,W]."""
    if panel.ndim != 4 or panel.shape[0] < 4 or panel.shape[0] % 2:
        raise ValueError("expected an even [M,C,H,W] panel with M>=4")
    flat = panel.reshape(panel.shape[0], -1).to(dtype=torch.float64)
    unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(EPS)
    centroid = unit_vector(unit.mean(dim=0))
    odd = unit_vector(unit[0::2].mean(dim=0))
    even = unit_vector(unit[1::2].mean(dim=0))
    return centroid, cosine_distance(odd, even)


def single_age_span(mean_directions: list[torch.Tensor]) -> float:
    if len(mean_directions) < 2:
        raise ValueError("at least two Single ages are required")
    return max(
        cosine_distance(mean_directions[left], mean_directions[right])
        for left in range(len(mean_directions))
        for right in range(left + 1, len(mean_directions))
    )


@torch.no_grad()
def proposal_panel(
    netG,
    anchor: torch.Tensor,
    t: int,
    z_panel: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, z_panel.shape[0], chunk_size):
        z = z_panel[start : start + chunk_size]
        batch = z.shape[0]
        anchor_batch = anchor.expand(batch, -1, -1, -1).contiguous()
        time_idx = torch.full((batch,), t, dtype=torch.long, device=anchor.device)
        chunks.append(netG(anchor_batch, time_idx, z) - anchor_batch)
    return torch.cat(chunks, dim=0)


def make_panel(domain: str, stem: str, t: int, m: int, device: str) -> torch.Tensor:
    seed = stable_seed(PROTOCOL_ID, domain, stem, t, "proposal")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(m, 256, generator=generator, dtype=torch.float32).to(device)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def representative_pca_rows(
    domain: str,
    stem: str,
    t: int,
    aio_mean: torch.Tensor,
    single_means: list[torch.Tensor],
) -> list[dict]:
    labels = ["AIO e1", *[f"Single e{epoch}" for epoch in range(1, 6)]]
    vectors = torch.stack([aio_mean, *single_means]).cpu().numpy().astype(np.float64)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    coords = evecs[:, order[:2]] * np.sqrt(evals[:2])[None, :]
    return [
        {
            "protocol_id": PROTOCOL_ID,
            "domain": domain,
            "stem": stem,
            "bridge_time_index": t,
            "role": label,
            "pca1": float(xy[0]),
            "pca2": float(xy[1]),
            "eigenvalue_1": float(evals[0]),
            "eigenvalue_2": float(evals[1]),
        }
        for label, xy in zip(labels, coords)
    ]


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
    repo_root = Path(path_map["repo_root"])
    baseline_root = Path(path_map["baseline_root"])
    checkpoints_root = Path(path_map["checkpoints_root"])
    heldout = json.loads((run_root / "state" / "HELDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    bridge_state_fn, build_generator, load_image_fn = import_measurement_helpers(repo_root)

    if args.device.startswith("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    times = bridge_times(5)
    age_rows: list[dict] = []
    primary_rows: list[dict] = []
    pca_rows: list[dict] = []
    checkpoint_hashes: dict[str, str] = {}

    aio_checkpoint = checkpoints_root / "aio5_plain_s2041_e1" / "1_net_G.pth"
    checkpoint_hashes["aio_e1"] = sha256_file(aio_checkpoint)

    for domain_index, domain in enumerate(DOMAINS):
        images = [row for row in heldout if row["domain"] == domain]
        if len(images) != 20:
            raise RuntimeError(f"{domain}: expected 20 images, got {len(images)}")
        aio_net = build_generator(str(aio_checkpoint), str(baseline_root), args.device)
        # Cache the common AIO anchor and its own mean direction once.  A cache
        # entry contains GPU tensors only for the active domain.
        cache: dict[tuple[str, int], dict] = {}
        for image_row in images:
            x = load_image_fn(image_row["source_path"], 128, args.device)
            for t in (1, 2, 3):
                z_panel = make_panel(domain, image_row["stem"], t, args.m, args.device)
                rollout_seed = stable_seed(PROTOCOL_ID, domain, image_row["stem"], t, "rollout")
                x_aio = bridge_state_fn(
                    aio_net,
                    x,
                    t,
                    z_panel,
                    tau=0.01,
                    times=times,
                    rollout_seed=rollout_seed,
                )
                aio_panel = proposal_panel(aio_net, x_aio, t, z_panel, chunk_size=args.chunk_size)
                aio_mean, aio_floor = mean_direction_and_floor(aio_panel)
                aio_mean_m16, aio_floor_m16 = mean_direction_and_floor(aio_panel[:16])
                cache[(image_row["stem"], t)] = {
                    "x": x.detach(),
                    "z": z_panel,
                    "rollout_seed": rollout_seed,
                    "x_aio": x_aio.detach(),
                    "aio_mean_on_aio": aio_mean.cpu(),
                    "aio_floor_on_aio": aio_floor,
                    "aio_mean_on_aio_M16": aio_mean_m16.cpu(),
                    "aio_floor_on_aio_M16": aio_floor_m16,
                    "single_means_on_aio": [],
                }

        rows_by_key: dict[tuple[str, int], list[dict]] = {
            (row["stem"], t): [] for row in images for t in (1, 2, 3)
        }
        for epoch in range(1, 6):
            single_checkpoint = checkpoints_root / f"single_{domain}_s2041_e5" / f"{epoch}_net_G.pth"
            checkpoint_hashes[f"single_{domain}_e{epoch}"] = sha256_file(single_checkpoint)
            single_net = build_generator(str(single_checkpoint), str(baseline_root), args.device)
            for image_row in images:
                stem = image_row["stem"]
                for t in (1, 2, 3):
                    entry = cache[(stem, t)]
                    x_single = bridge_state_fn(
                        single_net,
                        entry["x"],
                        t,
                        entry["z"],
                        tau=0.01,
                        times=times,
                        rollout_seed=entry["rollout_seed"],
                    )
                    single_on_aio = proposal_panel(
                        single_net, entry["x_aio"], t, entry["z"], chunk_size=args.chunk_size
                    )
                    aio_on_single = proposal_panel(
                        aio_net, x_single, t, entry["z"], chunk_size=args.chunk_size
                    )
                    single_on_single = proposal_panel(
                        single_net, x_single, t, entry["z"], chunk_size=args.chunk_size
                    )
                    single_mean_aio, floor_single_aio = mean_direction_and_floor(single_on_aio)
                    aio_mean_single, floor_aio_single = mean_direction_and_floor(aio_on_single)
                    single_mean_single, floor_single_single = mean_direction_and_floor(single_on_single)
                    single_mean_aio_m16, floor_single_aio_m16 = mean_direction_and_floor(single_on_aio[:16])
                    aio_mean_single_m16, floor_aio_single_m16 = mean_direction_and_floor(aio_on_single[:16])
                    single_mean_single_m16, floor_single_single_m16 = mean_direction_and_floor(single_on_single[:16])
                    deflection_aio_anchor = cosine_distance(entry["aio_mean_on_aio"], single_mean_aio)
                    deflection_single_anchor = cosine_distance(aio_mean_single, single_mean_single)
                    kdd = 0.5 * (deflection_aio_anchor + deflection_single_anchor)
                    deflection_aio_anchor_m16 = cosine_distance(
                        entry["aio_mean_on_aio_M16"], single_mean_aio_m16
                    )
                    deflection_single_anchor_m16 = cosine_distance(
                        aio_mean_single_m16, single_mean_single_m16
                    )
                    kdd_m16 = 0.5 * (
                        deflection_aio_anchor_m16 + deflection_single_anchor_m16
                    )
                    proposal_floor = max(
                        entry["aio_floor_on_aio"],
                        floor_single_aio,
                        floor_aio_single,
                        floor_single_single,
                    )
                    proposal_floor_m16 = max(
                        entry["aio_floor_on_aio_M16"],
                        floor_single_aio_m16,
                        floor_aio_single_m16,
                        floor_single_single_m16,
                    )
                    entry["single_means_on_aio"].append(single_mean_aio.cpu())
                    row = {
                        "protocol_id": PROTOCOL_ID,
                        "domain": domain,
                        "stem": stem,
                        "bridge_time_index": t,
                        "bridge_time_value": float(times[t]),
                        "single_epoch": epoch,
                        "reciprocal_KDD": kdd,
                        "reciprocal_KDD_M16": kdd_m16,
                        "aio_anchor_deflection": deflection_aio_anchor,
                        "single_anchor_deflection": deflection_single_anchor,
                        "proposal_floor": proposal_floor,
                        "proposal_floor_M16": proposal_floor_m16,
                        "floor_aio_on_aio": entry["aio_floor_on_aio"],
                        "floor_single_on_aio": floor_single_aio,
                        "floor_aio_on_single": floor_aio_single,
                        "floor_single_on_single": floor_single_single,
                        "aio_checkpoint_sha256": checkpoint_hashes["aio_e1"],
                        "single_checkpoint_sha256": checkpoint_hashes[f"single_{domain}_e{epoch}"],
                        "image_sha256": image_row["sha256"],
                        "proposal_seed": stable_seed(PROTOCOL_ID, domain, stem, t, "proposal"),
                        "rollout_seed": entry["rollout_seed"],
                        "M": args.m,
                        "target_content_read": False,
                    }
                    rows_by_key[(stem, t)].append(row)
                    age_rows.append(row)
            del single_net
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        for image_row in images:
            stem = image_row["stem"]
            for t in (1, 2, 3):
                entry = cache[(stem, t)]
                rows = rows_by_key[(stem, t)]
                if len(rows) != 5 or len(entry["single_means_on_aio"]) != 5:
                    raise RuntimeError(f"incomplete age trajectory: {domain}/{stem}/t{t}")
                age_span = single_age_span(entry["single_means_on_aio"])
                best = min(rows, key=lambda row: row["reciprocal_KDD"])
                max_floor = max(row["proposal_floor"] for row in rows)
                reference = max(max_floor, age_span)
                effect = math.log10(best["reciprocal_KDD"] + EPS) - math.log10(reference + EPS)
                e5 = next(row for row in rows if row["single_epoch"] == 5)
                primary_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "domain": domain,
                        "stem": stem,
                        "bridge_time_index": t,
                        "bridge_time_value": float(times[t]),
                        "best_single_epoch": best["single_epoch"],
                        "min_reciprocal_KDD": best["reciprocal_KDD"],
                        "clock_matched_KDD_e5": e5["reciprocal_KDD"],
                        "best_aio_anchor_deflection": best["aio_anchor_deflection"],
                        "best_single_anchor_deflection": best["single_anchor_deflection"],
                        "max_proposal_floor": max_floor,
                        "single_age_span": age_span,
                        "primary_reference": reference,
                        "log10_deflection_excess": effect,
                        "above_reference": best["reciprocal_KDD"] > reference,
                        "reference_source": "single_age_span" if age_span >= max_floor else "proposal_floor",
                        "image_sha256": image_row["sha256"],
                        "target_content_read": False,
                    }
                )
                if image_row.get("representative") and t == 2:
                    pca_rows.extend(
                        representative_pca_rows(
                            domain,
                            stem,
                            t,
                            entry["aio_mean_on_aio"],
                            entry["single_means_on_aio"],
                        )
                    )
        del aio_net, cache
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"completed reciprocal measurement for {domain_index + 1}/{len(DOMAINS)}: {domain}", flush=True)

    age_path = run_root / "raw" / "RECIPROCAL_KERNEL_BY_AGE.csv"
    primary_path = run_root / "raw" / "RECIPROCAL_KERNEL_PRIMARY.csv"
    pca_path = run_root / "raw" / "RECIPROCAL_REPRESENTATIVE_PCA.csv"
    write_csv(age_path, age_rows)
    write_csv(primary_path, primary_rows)
    write_csv(pca_path, pca_rows)

    all_numeric = [
        float(row[key])
        for row in primary_rows
        for key in ("min_reciprocal_KDD", "primary_reference", "log10_deflection_excess")
    ]
    state = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "age_rows": len(age_rows),
        "expected_age_rows": 5 * 20 * 3 * 5,
        "primary_rows": len(primary_rows),
        "expected_primary_rows": 5 * 20 * 3,
        "representative_pca_rows": len(pca_rows),
        "M": args.m,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        "target_content_read": False,
        "hashes": {
            age_path.name: sha256_file(age_path),
            primary_path.name: sha256_file(primary_path),
            pca_path.name: sha256_file(pca_path),
        },
        "checkpoint_hashes": checkpoint_hashes,
    }
    if (
        state["age_rows"] != state["expected_age_rows"]
        or state["primary_rows"] != state["expected_primary_rows"]
        or len(pca_rows) != 30
        or not all(math.isfinite(value) for value in all_numeric)
    ):
        state["status"] = "HOLD"
    dump_json(run_root / "state" / "RECIPROCAL_MEASUREMENT_STATE.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
