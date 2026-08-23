#!/usr/bin/env python3
"""Compute direction-field rank/spectrum statistics from saved checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechanism_common import (
    BRIDGE_TIMES,
    DOMAINS,
    EPOCHS,
    ckpt_path,
    direction_statistics,
    medoid_images,
    sample_unit_directions,
)
from measure_path_geometry import build_generator


def _aggregate(stats_list: list[dict]) -> dict:
    keys = ["effective_rank", "top1_energy", "top3_energy", "spectral_entropy", "mean_energy"]
    out = {}
    for k in keys:
        vals = [s[k] for s in stats_list if np.isfinite(s.get(k, float("nan")))]
        out[k] = float(np.median(vals)) if vals else float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--refactor-root", required=True)
    parser.add_argument("--epochs", default=",".join(str(e) for e in EPOCHS))
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    out = {"seed": 2026, "m": args.m, "single": {}, "aio": {}}

    for domain in DOMAINS:
        method = f"single_{domain}_s2026"
        images = medoid_images(domain)
        out["single"][domain] = {}
        for epoch in epochs:
            ckpt = Path(args.root) / "checkpoints" / method / f"{epoch}_net_G.pth"
            if not ckpt.exists():
                out["single"][domain][str(epoch)] = {"error": "missing_checkpoint"}
                continue
            netG = build_generator(str(ckpt), args.refactor_root, args.device)
            directions = sample_unit_directions(
                netG,
                images,
                bridge_times_idx=BRIDGE_TIMES,
                m=args.m,
                device=args.device,
                seed=2026,
            )
            stats_list = [direction_statistics(v) for v in directions.values()]
            out["single"][domain][str(epoch)] = _aggregate(stats_list)
            del netG
            torch.cuda.empty_cache()
            print(f"RANK single {domain} e{epoch}")

    method = "aio_plain_s2026"
    images = medoid_images(None)
    for epoch in epochs:
        ckpt = Path(args.root) / "checkpoints" / method / f"{epoch}_net_G.pth"
        if not ckpt.exists():
            out["aio"][str(epoch)] = {"error": "missing_checkpoint"}
            continue
        netG = build_generator(str(ckpt), args.refactor_root, args.device)
        directions = sample_unit_directions(
            netG,
            images,
            bridge_times_idx=BRIDGE_TIMES,
            m=args.m,
            device=args.device,
            seed=2026,
        )
        by_domain = {d: [] for d in DOMAINS}
        for (stem, t), v in directions.items():
            im = next(r for r in images if r["stem"] == stem)
            by_domain[im["domain"]].append(direction_statistics(v))
        per_domain = {d: _aggregate(v) for d, v in by_domain.items()}
        out["aio"][str(epoch)] = {
            "overall": _aggregate([direction_statistics(v) for v in directions.values()]),
            "per_domain": per_domain,
        }
        del netG
        torch.cuda.empty_cache()
        print(f"RANK aio e{epoch}")

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out_json}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
