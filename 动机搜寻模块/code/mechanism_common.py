#!/usr/bin/env python3
"""Shared helpers for mechanism screening.

All functions are read-only with respect to upstream code.  Direction sampling
imports the existing clean refactor through ``measure_path_geometry``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from measure_path_geometry import (
    bridge_times,
    build_generator,
    load_image,
    make_latent_panel,
    sample_directions,
)


ROOT = Path(
    os.environ.get("UNSB_MOTIVATION_ROOT", Path(__file__).resolve().parent.parent)
).expanduser().resolve()
DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
EPOCHS = [1, 3, 4, 5, 6, 17, 20]
BRIDGE_TIMES = [1, 2, 3]


def load_manifest() -> dict:
    return json.loads((ROOT / "MEASUREMENT_MANIFEST.json").read_text(encoding="utf-8"))


def ckpt_path(method: str, epoch: int) -> Path:
    return ROOT / "checkpoints" / method / f"{epoch}_net_G.pth"


def medoid_images(domain: str | None = None) -> list[dict]:
    manifest = load_manifest()
    images = manifest["b_medoids"]
    if domain is not None:
        return [r for r in images if r["domain"] == domain]
    return images


def sample_unit_directions(
    netG,
    images: list[dict],
    *,
    bridge_times_idx: list[int] | None = None,
    m: int = 32,
    ngf: int = 64,
    tau: float = 0.01,
    num_timesteps: int = 5,
    device: str = "cuda",
    seed: int = 2026,
) -> dict[tuple[str, int], np.ndarray]:
    """Return flattened unit directions for each image and bridge time."""
    bridge_times_idx = bridge_times_idx or BRIDGE_TIMES
    times = bridge_times(num_timesteps)
    z_panel = make_latent_panel(ngf, m, seed=seed).to(device)
    out: dict[tuple[str, int], np.ndarray] = {}
    for im in images:
        x = load_image(im["source_path"], 128, device)
        for t in bridge_times_idx:
            rollout_seed = seed + t * 100003
            directions = sample_directions(
                netG,
                x,
                t,
                z_panel,
                tau=tau,
                times=times,
                rollout_seed=rollout_seed,
            )
            flat = directions.reshape(directions.shape[0], -1)
            unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
            out[(im["stem"], t)] = unit.detach().cpu().numpy().astype(np.float32)
    return out


def direction_statistics(directions: np.ndarray) -> dict:
    """SVD-based statistics for an [M, D] direction matrix."""
    X = directions.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    s = s[s > 1e-12]
    energy = s**2
    total = float(energy.sum())
    if total <= 0:
        return {
            "effective_rank": 0.0,
            "top1_energy": float("nan"),
            "top3_energy": float("nan"),
            "spectral_entropy": 0.0,
            "mean_energy": 0.0,
        }
    p = energy / total
    p_pos = p[p > 0]
    entropy = float(-(p_pos * np.log(p_pos)).sum())
    mean_vec = directions.mean(axis=0)
    return {
        "effective_rank": float((energy.sum() ** 2) / float((energy**2).sum())),
        "top1_energy": float(p[0]) if p.size >= 1 else float("nan"),
        "top3_energy": float(p[:3].sum()) if p.size >= 3 else float("nan"),
        "spectral_entropy": entropy,
        "mean_energy": float(np.sum(mean_vec**2)),
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered kernel alignment for [n, d] feature matrices."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic = float(np.sum((Xc @ Xc.T) * (Yc @ Yc.T)))
    hsic_xx = float(np.sqrt(np.sum((Xc @ Xc.T) ** 2)))
    hsic_yy = float(np.sqrt(np.sum((Yc @ Yc.T) ** 2)))
    denom = hsic_xx * hsic_yy
    return hsic / denom if denom > 1e-12 else 0.0


def hook_features(netG, x, time_idx, z, layer_names: list[str]) -> dict[str, np.ndarray]:
    """Extract flattened intermediate features at requested module names."""
    handles = []
    activations: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            activations[name] = out.detach().reshape(out.shape[0], -1).cpu()

        return hook_fn

    by_name = dict(netG.named_modules())
    for name in layer_names:
        if name not in by_name:
            raise KeyError(f"module not found: {name}")
        handles.append(by_name[name].register_forward_hook(make_hook(name)))
    with torch.no_grad():
        netG(x, time_idx, z)
    for h in handles:
        h.remove()
    return {k: v.numpy().astype(np.float32) for k, v in activations.items()}
