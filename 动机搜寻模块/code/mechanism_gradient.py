#!/usr/bin/env python3
"""Cross-domain gradient-conflict proxy from conditional direction fields.

The full parameter-gradient conflict requires a complete training graph
(discriminator/encoder/optimizer), which is not saved in this bypath.  As the
nearest checkpoint-based alternative, this script computes per-domain mean unit
directions and their pairwise cosine / conflict fraction for the shared AIO
generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechanism_common import BRIDGE_TIMES, DOMAINS, EPOCHS, ckpt_path, medoid_images, sample_unit_directions
from measure_path_geometry import build_generator


def _pairwise_cosine(vectors: dict[str, np.ndarray]) -> dict:
    domains = sorted(vectors)
    cosines = []
    pairs = []
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            va = vectors[a]
            vb = vectors[b]
            denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
            c = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
            cosines.append(c)
            pairs.append((a, b, c))
    arr = np.asarray(cosines, dtype=np.float64)
    return {
        "mean_cosine": float(np.mean(arr)) if arr.size else float("nan"),
        "conflict_fraction": float(np.mean(arr < 0.0)) if arr.size else float("nan"),
        "pairs": pairs,
    }


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
    images = medoid_images(None)
    out = {"seed": 2026, "method": "aio_plain_s2026", "proxy": "direction_cosine_conflict"}
    results = {}
    for epoch in epochs:
        ckpt = Path(args.root) / "checkpoints" / "aio_plain_s2026" / f"{epoch}_net_G.pth"
        if not ckpt.exists():
            results[str(epoch)] = {"error": "missing_checkpoint"}
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
        by_t = {}
        for t in BRIDGE_TIMES:
            vectors = {}
            for domain in DOMAINS:
                mats = []
                for im in images:
                    if im["domain"] != domain:
                        continue
                    mats.append(directions[(im["stem"], t)])
                if mats:
                    vectors[domain] = np.mean(np.concatenate(mats, axis=0), axis=0)
            by_t[f"t{t}"] = _pairwise_cosine(vectors)
        results[str(epoch)] = by_t
        del netG
        torch.cuda.empty_cache()
        print(f"GRADIENT_PROXY aio e{epoch}")

    out["epochs"] = results
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out_json}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
