#!/usr/bin/env python3
"""Upgraded direction-field rank/spectrum measurement with M=64 and multi-image."""

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
    load_manifest,
    sample_unit_directions,
)
from measure_path_geometry import build_generator


def _images_for_domain(domain: str | None, limit: int = 3) -> list[dict]:
    manifest = load_manifest()
    images = manifest["c_subset"]
    if domain is not None:
        images = [r for r in images if r["domain"] == domain]
    return images[:limit] if domain is None else images[:limit]


def _aggregate(stats_list: list[dict]) -> dict:
    keys = ["effective_rank", "top1_energy", "top3_energy", "spectral_entropy", "mean_energy", "cov_trace"]
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
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--images-per-domain", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    out = {"seed": 2026, "m": args.m, "single": {}, "aio": {}}

    for domain in DOMAINS:
        method = f"single_{domain}_s2026"
        images = _images_for_domain(domain, args.images_per_domain)
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
            stats_by_t = {}
            for t in BRIDGE_TIMES:
                mats = [directions[(im["stem"], t)] for im in images]
                stacked = np.concatenate(mats, axis=0)
                stat = direction_statistics(stacked)
                stat["cov_trace"] = float(np.trace(np.cov(stacked.T, bias=False)))
                stats_by_t[f"t{t}"] = stat
            out["single"][domain][str(epoch)] = {
                "aggregate": _aggregate(list(stats_by_t.values())),
                "per_bridge_time": stats_by_t,
            }
            del netG
            torch.cuda.empty_cache()
            print(f"RANK64 single {domain} e{epoch}")

    method = "aio_plain_s2026"
    for epoch in epochs:
        ckpt = Path(args.root) / "checkpoints" / method / f"{epoch}_net_G.pth"
        if not ckpt.exists():
            out["aio"][str(epoch)] = {"error": "missing_checkpoint"}
            continue
        netG = build_generator(str(ckpt), args.refactor_root, args.device)
        per_domain = {}
        for domain in DOMAINS:
            images = _images_for_domain(domain, args.images_per_domain)
            directions = sample_unit_directions(
                netG,
                images,
                bridge_times_idx=BRIDGE_TIMES,
                m=args.m,
                device=args.device,
                seed=2026,
            )
            stats_by_t = {}
            for t in BRIDGE_TIMES:
                mats = [directions[(im["stem"], t)] for im in images]
                stacked = np.concatenate(mats, axis=0)
                stat = direction_statistics(stacked)
                stat["cov_trace"] = float(np.trace(np.cov(stacked.T, bias=False)))
                stats_by_t[f"t{t}"] = stat
            per_domain[domain] = {
                "aggregate": _aggregate(list(stats_by_t.values())),
                "per_bridge_time": stats_by_t,
            }
        overall_stats = []
        for d in DOMAINS:
            for t in BRIDGE_TIMES:
                overall_stats.append(per_domain[d]["per_bridge_time"][f"t{t}"])
        out["aio"][str(epoch)] = {
            "overall": _aggregate(overall_stats),
            "per_domain": per_domain,
        }
        del netG
        torch.cuda.empty_cache()
        print(f"RANK64 aio e{epoch}")

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out_json}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
