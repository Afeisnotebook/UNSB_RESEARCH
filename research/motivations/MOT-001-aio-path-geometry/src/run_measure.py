#!/usr/bin/env python3
"""Measurement orchestrator (GPU host): extract c/d/e rows and panel-b PCA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from measure_path_geometry import (
    bridge_times,
    build_generator,
    extract_one,
    joint_pca,
    load_image,
    make_latent_panel,
    sample_directions,
)


def load_manifest(root: Path) -> dict:
    return json.loads((root / "MEASUREMENT_MANIFEST.json").read_text(encoding="utf-8"))


def pick_images(manifest: dict, key: str, domain: str | None) -> list[dict]:
    images = manifest[key]
    if domain:
        images = [r for r in images if r["domain"] == domain]
    return images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--refactor-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    manifest = load_manifest(root)
    common = plan.get("common", {})
    times = bridge_times(common.get("num_timesteps", 5))
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for entry in plan["measurements"]:
        images = pick_images(manifest, entry["images_key"], entry.get("domain"))
        if not images:
            print(f"SKIP (no images) {entry['tag']}")
            continue
        netG = build_generator(entry["ckpt"], args.refactor_root, args.device)
        extract_one(
            netG,
            images,
            bridge_times_idx=entry.get("bridge_times", common.get("bridge_times", [1, 2, 3])),
            m=common.get("m", 64),
            ngf=common.get("ngf", 64),
            tau=common.get("tau", 0.01),
            times=times,
            region_patch=common.get("region_patch", 32),
            device=args.device,
            seed=common.get("seed", 2026),
            out_dir=raw_dir,
            tag=entry["tag"],
            extra={"method": entry["method"], "epoch": entry["epoch"]},
        )
        print(f"MEASURED {entry['tag']} ({len(images)} images)")
        del netG
        torch.cuda.empty_cache()

    panel_b = plan.get("panel_b")
    if panel_b:
        medoids = manifest["b_medoids"]
        z_panel = make_latent_panel(
            common.get("ngf", 64), common.get("m", 64), seed=common.get("seed", 2026)
        ).to(args.device)
        directions_by_method: dict[str, list[torch.Tensor]] = {}
        for method, cfg in panel_b["methods"].items():
            netG = build_generator(cfg["ckpt"], args.refactor_root, args.device)
            collected = []
            for im in medoids:
                x = load_image(im["source_path"], 128, args.device)
                for t in common.get("bridge_times", [1, 2, 3]):
                    rollout_seed = common.get("seed", 2026) + t * 100003
                    directions = sample_directions(
                        netG,
                        x,
                        t,
                        z_panel,
                        tau=common.get("tau", 0.01),
                        times=times,
                        rollout_seed=rollout_seed,
                    )
                    flat = directions.reshape(directions.shape[0], -1)
                    unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    collected.append(unit.detach().cpu())
            directions_by_method[method] = collected
            del netG
            torch.cuda.empty_cache()

        pca = joint_pca(directions_by_method, n_components=2)
        np.savez_compressed(
            raw_dir / "panel_b_directions.npz",
            **{f"unit_{m}": np.stack([u.numpy() for u in v]) for m, v in directions_by_method.items()},
        )
        (raw_dir / "panel_b_pca.json").write_text(
            json.dumps(
                {
                    "proj": pca["proj"].tolist(),
                    "singular_values": pca["singular_values"].tolist(),
                    "method_lengths": {m: len(v) for m, v in directions_by_method.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("PANEL_B_DONE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
