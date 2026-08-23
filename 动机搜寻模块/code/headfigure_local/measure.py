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
    from .common import (
        DOMAINS,
        bridge_times,
        dump_json,
        sha256_file,
        spherical_dispersion,
        stable_seed,
    )
except ImportError:  # direct script execution
    from common import (
        DOMAINS,
        bridge_times,
        dump_json,
        sha256_file,
        spherical_dispersion,
        stable_seed,
    )


PROTOCOL_ID = "unsb-initial-shared-bridge-fanout-dual-control-v1"


def import_measurement_helpers(repo_root: Path):
    code_root = repo_root / "动机搜寻模块" / "code"
    sys.path.insert(0, str(code_root))
    from measure_path_geometry import bridge_state, build_generator, load_image

    return bridge_state, build_generator, load_image


@torch.no_grad()
def sample_directions_chunked(
    netG,
    x: torch.Tensor,
    t: int,
    z_panel: torch.Tensor,
    *,
    tau: float,
    times: np.ndarray,
    rollout_seed: int,
    chunk_size: int,
    bridge_state_fn,
) -> torch.Tensor:
    xt = bridge_state_fn(
        netG,
        x,
        t,
        z_panel,
        tau=tau,
        times=times,
        rollout_seed=rollout_seed,
    )
    denom = max(1.0 - float(times[t]), 1e-8)
    chunks = []
    for start in range(0, z_panel.shape[0], chunk_size):
        z = z_panel[start : start + chunk_size]
        batch = z.shape[0]
        xt_batch = xt.expand(batch, -1, -1, -1).contiguous()
        time_idx = torch.full((batch,), t, dtype=torch.long, device=x.device)
        y = netG(xt_batch, time_idx, z)
        chunks.append((y - xt_batch) / denom)
    return torch.cat(chunks, dim=0)


def joint_pca_rows(direction_groups: dict[str, np.ndarray], domain: str, stem: str) -> list[dict]:
    arms = ["single_e1", "single_e5", "aio_e1"]
    if any(arm not in direction_groups for arm in arms):
        raise RuntimeError(f"missing representative directions for {domain}: {direction_groups.keys()}")
    x = np.concatenate([direction_groups[arm] for arm in arms], axis=0).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]
    coords = evecs[:, :2] * np.sqrt(evals[:2])[None, :]
    rows = []
    offset = 0
    for arm in arms:
        n = direction_groups[arm].shape[0]
        for proposal, xy in enumerate(coords[offset : offset + n]):
            rows.append(
                {
                    "domain": domain,
                    "stem": stem,
                    "arm": arm,
                    "proposal": proposal,
                    "pca1": float(xy[0]),
                    "pca2": float(xy[1]),
                    "joint_eigenvalue_1": float(evals[0]),
                    "joint_eigenvalue_2": float(evals[1]),
                }
            )
        offset += n
    return rows


def measure_checkpoint(
    *,
    netG,
    checkpoint: Path,
    arm: str,
    epoch: int,
    images: list[dict],
    device: str,
    m: int,
    chunk_size: int,
    times: np.ndarray,
    bridge_state_fn,
    load_image_fn,
    representative_store: dict,
) -> list[dict]:
    rows: list[dict] = []
    checkpoint_hash = sha256_file(checkpoint)
    for image_row in images:
        domain = image_row["domain"]
        stem = image_row["stem"]
        x = load_image_fn(image_row["source_path"], 128, device)
        for t in (1, 2, 3):
            proposal_seed = stable_seed(PROTOCOL_ID, domain, stem, t, "proposal")
            rollout_seed = stable_seed(PROTOCOL_ID, domain, stem, t, "rollout")
            generator = torch.Generator(device="cpu").manual_seed(proposal_seed)
            z_panel = torch.randn(m, 256, generator=generator, dtype=torch.float32).to(device)
            directions = sample_directions_chunked(
                netG,
                x,
                t,
                z_panel,
                tau=0.01,
                times=times,
                rollout_seed=rollout_seed,
                chunk_size=chunk_size,
                bridge_state_fn=bridge_state_fn,
            )
            flat = directions.reshape(m, -1)
            unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            unit_np = unit.detach().cpu().numpy().astype(np.float32)
            stats = spherical_dispersion(unit_np)
            stats_m8 = spherical_dispersion(unit_np[:8])
            stats_m16 = spherical_dispersion(unit_np[:16])
            rows.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "arm": arm,
                    "epoch": epoch,
                    "domain": domain,
                    "stem": stem,
                    "bridge_time_index": t,
                    "bridge_time_value": float(times[t]),
                    "M": m,
                    "D_sph": stats["D_sph"],
                    "D_sph_M8": stats_m8["D_sph"],
                    "D_sph_M16": stats_m16["D_sph"],
                    "log10_D_sph": math.log10(stats["D_sph"] + 1e-12),
                    "R2": stats["R2"],
                    "legacy_U": stats["legacy_U"],
                    "mean_pair_cos": stats["mean_pair_cos"],
                    "mean_pair_angle_deg": stats["mean_pair_angle_deg"],
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "image_sha256": image_row["sha256"],
                    "proposal_seed": proposal_seed,
                    "rollout_seed": rollout_seed,
                    "target_content_read": False,
                }
            )
            if image_row.get("representative") and t == 2 and arm in {"single_e1", "single_e5", "aio_e1"}:
                representative_store.setdefault(domain, {})[arm] = unit_np
        del x
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
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

    start = time.time()
    times = bridge_times(5)
    all_rows: list[dict] = []
    representatives: dict[str, dict[str, np.ndarray]] = {}

    for domain in DOMAINS:
        images = [row for row in heldout if row["domain"] == domain]
        for epoch in range(1, 6):
            checkpoint = checkpoints_root / f"single_{domain}_s2041_e5" / f"{epoch}_net_G.pth"
            netG = build_generator(str(checkpoint), str(baseline_root), args.device)
            all_rows.extend(
                measure_checkpoint(
                    netG=netG,
                    checkpoint=checkpoint,
                    arm=f"single_e{epoch}",
                    epoch=epoch,
                    images=images,
                    device=args.device,
                    m=args.m,
                    chunk_size=args.chunk_size,
                    times=times,
                    bridge_state_fn=bridge_state_fn,
                    load_image_fn=load_image_fn,
                    representative_store=representatives,
                )
            )
            del netG
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    aio_checkpoint = checkpoints_root / "aio5_plain_s2041_e1" / "1_net_G.pth"
    aio_net = build_generator(str(aio_checkpoint), str(baseline_root), args.device)
    all_rows.extend(
        measure_checkpoint(
            netG=aio_net,
            checkpoint=aio_checkpoint,
            arm="aio_e1",
            epoch=1,
            images=heldout,
            device=args.device,
            m=args.m,
            chunk_size=args.chunk_size,
            times=times,
            bridge_state_fn=bridge_state_fn,
            load_image_fn=load_image_fn,
            representative_store=representatives,
        )
    )
    del aio_net
    raw_path = run_root / "raw" / "DIRECTION_STATISTICS.csv"
    write_csv(raw_path, all_rows)

    pca_rows: list[dict] = []
    representative_lookup = {(row["domain"], row["stem"]): row for row in heldout if row.get("representative")}
    for domain in DOMAINS:
        key = next(key for key in representative_lookup if key[0] == domain)
        pca_rows.extend(joint_pca_rows(representatives[domain], domain, key[1]))
    pca_path = run_root / "raw" / "REPRESENTATIVE_JOINT_PCA.csv"
    write_csv(pca_path, pca_rows)

    elapsed = time.time() - start
    state = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "rows": len(all_rows),
        "expected_rows": 5 * 5 * 20 * 3 + 5 * 20 * 3,
        "pca_rows": len(pca_rows),
        "M": args.m,
        "bridge_times": times.tolist(),
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        "target_content_read": False,
        "raw_sha256": sha256_file(raw_path),
        "pca_sha256": sha256_file(pca_path),
    }
    if state["rows"] != state["expected_rows"] or not all(math.isfinite(float(r["D_sph"])) for r in all_rows):
        state["status"] = "HOLD"
    dump_json(run_root / "state" / "MEASUREMENT_STATE.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
