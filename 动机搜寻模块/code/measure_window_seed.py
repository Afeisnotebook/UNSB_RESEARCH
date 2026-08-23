#!/usr/bin/env python3
"""Measure only the window-audit epochs for one additional seed.

This is a deliberately narrow measurement script.  It loads the existing
measurement image manifest and extracts the same b/c/d/e primitives as
``measure_path_geometry.extract_one`` for the selected epochs only.  It does
not train models and does not touch sealed/paired targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from measure_path_geometry import bridge_times, build_generator, extract_one


DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]


def pick_images(manifest: dict, domain: str | None) -> list[dict]:
    images = manifest["c_subset"]
    if domain:
        return [r for r in images if r["domain"] == domain]
    return images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--refactor-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", default="1,2,3,4,5,6,17,20")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.root)
    seed = args.seed
    raw_dir = root / "raw" / f"seed{seed}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "MEASUREMENT_MANIFEST.json").read_text(encoding="utf-8"))
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    times = bridge_times(5)

    common = {
        "bridge_times_idx": [1, 2, 3],
        "m": 64,
        "ngf": 64,
        "tau": 0.01,
        "times": times,
        "region_patch": 32,
        "device": args.device,
        "seed": seed,
    }

    n_rows = 0
    for domain in DOMAINS:
        method = f"single_{domain}_s{seed}"
        images = pick_images(manifest, domain)
        for epoch in epochs:
            ckpt = root / "checkpoints" / method / f"{epoch}_net_G.pth"
            if not ckpt.exists():
                print(f"SKIP missing checkpoint {ckpt}")
                continue
            netG = build_generator(str(ckpt), args.refactor_root, args.device)
            result = extract_one(
                netG,
                images,
                bridge_times_idx=common["bridge_times_idx"],
                m=common["m"],
                ngf=common["ngf"],
                tau=common["tau"],
                times=common["times"],
                region_patch=common["region_patch"],
                device=common["device"],
                seed=common["seed"],
                out_dir=raw_dir,
                tag=f"{method}__e{epoch:02d}",
                extra={"method": method, "epoch": epoch},
            )
            n_rows += result["rows"]
            print(f"MEASURED {method}__e{epoch:02d} ({len(images)} images)")
            del netG
            torch.cuda.empty_cache()

    aio_method = f"aio_plain_s{seed}"
    aio_images = pick_images(manifest, None)
    for epoch in epochs:
        ckpt = root / "checkpoints" / aio_method / f"{epoch}_net_G.pth"
        if not ckpt.exists():
            print(f"SKIP missing checkpoint {ckpt}")
            continue
        netG = build_generator(str(ckpt), args.refactor_root, args.device)
        result = extract_one(
            netG,
            aio_images,
            bridge_times_idx=common["bridge_times_idx"],
            m=common["m"],
            ngf=common["ngf"],
            tau=common["tau"],
            times=common["times"],
            region_patch=common["region_patch"],
            device=common["device"],
            seed=common["seed"],
            out_dir=raw_dir,
            tag=f"{aio_method}__e{epoch:02d}",
            extra={"method": aio_method, "epoch": epoch},
        )
        n_rows += result["rows"]
        print(f"MEASURED {aio_method}__e{epoch:02d} ({len(aio_images)} images)")
        del netG
        torch.cuda.empty_cache()

    print(json.dumps({"seed": seed, "raw_rows": n_rows, "raw_dir": str(raw_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
