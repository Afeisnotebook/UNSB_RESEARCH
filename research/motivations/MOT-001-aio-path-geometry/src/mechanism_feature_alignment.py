#!/usr/bin/env python3
"""Domain feature alignment proxy using intermediate-generator CKA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechanism_common import DOMAINS, EPOCHS, ckpt_path, hook_features, linear_cka, medoid_images
from measure_path_geometry import build_generator, load_image, make_latent_panel


LAYER_NAMES = [
    "model_res.2.conv_fin.2",
    "model_res.5.conv_fin.2",
    "model.4",
]


def _domain_features(netG, images, *, device, z_samples: int = 8) -> dict[str, dict[str, np.ndarray]]:
    ngf = 64
    z_panel = make_latent_panel(ngf, z_samples, seed=2026).to(device)
    per_domain: dict[str, dict[str, list[np.ndarray]]] = {d: {l: [] for l in LAYER_NAMES} for d in DOMAINS}
    for im in images:
        x = load_image(im["source_path"], 128, device)
        time_idx = torch.full((x.shape[0],), 1, dtype=torch.long, device=device)
        for zi in range(z_samples):
            z = z_panel[zi].unsqueeze(0).to(device)
            feats = hook_features(netG, x, time_idx, z, LAYER_NAMES)
            for layer, feat in feats.items():
                per_domain[im["domain"]][layer].append(feat)
    out = {}
    for d in DOMAINS:
        out[d] = {l: np.concatenate(v, axis=0) for l, v in per_domain[d].items() if v}
    return out


def _pairwise_mean_cka(features: dict[str, np.ndarray]) -> dict:
    domains = sorted(features)
    vals = []
    pairs = []
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            c = linear_cka(features[a], features[b])
            vals.append(c)
            pairs.append((a, b, c))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean_cka": float(np.mean(arr)) if arr.size else float("nan"),
        "min_cka": float(np.min(arr)) if arr.size else float("nan"),
        "max_cka": float(np.max(arr)) if arr.size else float("nan"),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--refactor-root", required=True)
    parser.add_argument("--epochs", default=",".join(str(e) for e in EPOCHS))
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--z-samples", type=int, default=8)
    args = parser.parse_args()

    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    images = medoid_images(None)
    out = {"seed": 2026, "method": "aio_plain_s2026", "layers": LAYER_NAMES, "epochs": {}}

    for epoch in epochs:
        ckpt = Path(args.root) / "checkpoints" / "aio_plain_s2026" / f"{epoch}_net_G.pth"
        if not ckpt.exists():
            out["epochs"][str(epoch)] = {"error": "missing_checkpoint"}
            continue
        netG = build_generator(str(ckpt), args.refactor_root, args.device)
        feats = _domain_features(netG, images, device=args.device, z_samples=args.z_samples)
        layer_result = {}
        for layer in LAYER_NAMES:
            layer_features = {d: feats[d][layer] for d in DOMAINS if layer in feats.get(d, {})}
            layer_result[layer] = _pairwise_mean_cka(layer_features)
        out["epochs"][str(epoch)] = layer_result
        del netG
        torch.cuda.empty_cache()
        print(f"FEATURE_ALIGN aio e{epoch}")

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out_json}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
