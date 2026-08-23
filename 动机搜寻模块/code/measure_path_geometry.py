#!/usr/bin/env python3
"""Path-geometry measurement for the UNSB motivation figure (b/c/d/e).

Pure-math primitives are importable without the refactor (CPU-testable).  The
generator loader imports the clean refactor on demand and requires CUDA for the
real run (executed on the GPU host).

Canonical quantities (frozen in MOTIVATION_FROZEN_SPEC.json):
  d_k      = (y_k - X_t) / (1 - t)           endpoint bridge direction
  d_k_norm = d_k / (||d_k||_2 + eps)         unit direction (panel b)
  U        = trace(Cov({d_k_norm})) / (||mean(d_k_norm)||^2 + eps)   (panel c/e)
  U_reg    = per-region Var_k(d_k) / (mean(d_k)^2 + eps)             (panel d)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


EPS = 1e-8


def bridge_times(num_timesteps: int = 5) -> np.ndarray:
    """Official UNSB physical schedule (T+1 grid points)."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be >= 2")
    incs = np.array([0.0] + [1.0 / (i + 1) for i in range(num_timesteps - 1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    return np.concatenate([np.zeros(1), times])


def make_latent_panel(ngf: int, m: int, seed: int = 2026) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(m, 4 * ngf, generator=gen)


@torch.no_grad()
def bridge_state(
    netG,
    x: torch.Tensor,
    t: int,
    z_panel: torch.Tensor,
    *,
    tau: float,
    times: np.ndarray,
    rollout_seed: int,
) -> torch.Tensor:
    """Deterministic rollout up to time index ``t`` (matches SBModel.forward)."""
    rng = torch.Generator(device=x.device).manual_seed(rollout_seed)
    Xt = x
    Xt_prev = None
    for tt in range(0, t + 1):
        if tt > 0:
            delta = float(times[tt] - times[tt - 1])
            denom = float(times[-1] - times[tt - 1])
            inter = delta / denom
            scale = delta * (1.0 - delta / denom)
            noise = torch.randn(
                Xt.shape, generator=rng, device=Xt.device, dtype=Xt.dtype
            )
            Xt = (1.0 - inter) * Xt + inter * Xt_prev.detach() + (scale * tau) ** 0.5 * noise
        time_idx = torch.full((x.shape[0],), tt, dtype=torch.long, device=x.device)
        z = z_panel[tt % len(z_panel)].unsqueeze(0).to(x.device)
        Xt_prev = netG(Xt, time_idx, z)
    return Xt


@torch.no_grad()
def sample_directions(
    netG,
    x: torch.Tensor,
    t: int,
    z_panel: torch.Tensor,
    *,
    tau: float,
    times: np.ndarray,
    rollout_seed: int,
) -> torch.Tensor:
    """Return stacked directions d_k, shape [M, C, H, W]."""
    Xt = bridge_state(netG, x, t, z_panel, tau=tau, times=times, rollout_seed=rollout_seed)
    time_idx = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
    horizon = float(times[t])
    denom = max(1.0 - horizon, EPS)
    out = []
    for m in range(z_panel.shape[0]):
        z = z_panel[m].unsqueeze(0).to(x.device)
        y = netG(Xt, time_idx, z)
        out.append((y - Xt) / denom)
    return torch.stack(out, dim=0).squeeze(1)


def unit_direction_dispersion(d_k_norm: torch.Tensor) -> float:
    """U = trace(Cov(unit dirs)) / ||mean(unit dirs)||^2 (image-level scalar)."""
    flat = d_k_norm.reshape(d_k_norm.shape[0], -1)
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    trace_cov = (centered * centered).sum() / float(flat.shape[0] - 1)
    mean_energy = (mean * mean).sum() + EPS
    return float((trace_cov / mean_energy).item())


def region_direction_dispersion(
    directions: torch.Tensor, *, region_patch: int = 32
) -> np.ndarray:
    """Per-region U_reg map using DT-style variance / mean^2 (panel d)."""
    M, C, H, W = directions.shape
    rh, rw = H // region_patch, W // region_patch
    if rh <= 0 or rw <= 0:
        raise ValueError("region_patch larger than image")
    d = directions.reshape(M, C, rh, region_patch, rw, region_patch)
    mean = d.mean(dim=0, keepdim=True)
    var = ((d - mean) ** 2).sum(dim=0) / float(M - 1)
    var_ch = var.mean(dim=0)
    signal_ch = (mean.squeeze(0) ** 2).mean(dim=0) + EPS
    u_reg = var_ch / signal_ch
    # average over each region's pixels
    u_map = u_reg.mean(dim=(1, 3))
    return u_map.detach().cpu().numpy()


def joint_pca(directions_by_method: dict[str, list[torch.Tensor]], n_components: int = 2):
    """Fit one PCA over all methods' flattened unit directions."""
    mats = []
    lengths = {}
    for method, list_of_d in directions_by_method.items():
        mats.extend([d.reshape(-1).cpu().numpy().astype(np.float64) for d in list_of_d])
        lengths[method] = len(list_of_d)
    if not mats:
        raise ValueError("no directions supplied")
    X = np.stack(mats, axis=0)
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt.T[:, :n_components]
    return {"proj": proj, "mean": mean, "singular_values": S[:n_components], "V": Vt[:n_components]}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000, seed: int = 2026, alpha: float = 0.05):
    """Paired (a-b) image-level mean with percentile bootstrap CI."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    diff = a - b
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return {
        "n": int(n),
        "mean": float(diff.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
    }


def build_generator(ckpt_path: str, refactor_root: str, device: str = "cuda"):
    """Build the clean refactor generator and load a netG checkpoint."""
    # ``run_all.sh`` passes the source root that contains ``models`` directly,
    # while some callers may pass the refactor checkout root instead.  Accept
    # both layouts without touching the upstream checkout.
    refactor_root = Path(refactor_root)
    for candidate in (refactor_root, refactor_root / "baseline"):
        if (candidate / "models" / "networks.py").exists():
            sys.path.insert(0, str(candidate))
    from models.networks import define_G  # noqa: E402

    class Opt:
        n_mlp = 3

    netG = define_G(
        3,
        3,
        64,
        "resnet_9blocks_cond",
        norm="instance",
        use_dropout=False,
        init_type="xavier",
        init_gain=0.02,
        no_antialias=False,
        no_antialias_up=False,
        gpu_ids=([0] if device.startswith("cuda") else []),
        opt=Opt(),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    netG.load_state_dict(state)
    netG.to(device)
    netG.eval()
    return netG


def load_image(path: str, res: int = 128, device: str = "cuda") -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)


def extract_one(
    netG,
    images: list[dict],
    *,
    bridge_times_idx: list[int],
    m: int,
    ngf: int,
    tau: float,
    times: np.ndarray,
    region_patch: int,
    device: str,
    seed: int,
    out_dir: Path,
    tag: str,
    extra: dict | None = None,
) -> dict:
    """Extract per-image U, unit directions and region-U map for one checkpoint."""
    z_panel = make_latent_panel(ngf, m, seed=seed).to(device)
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for im in images:
        x = load_image(im["source_path"], 128, device)
        for t in bridge_times_idx:
            rollout_seed = seed + t * 100003
            directions = sample_directions(
                netG, x, t, z_panel, tau=tau, times=times, rollout_seed=rollout_seed
            )
            flat = directions.reshape(directions.shape[0], -1)
            norms = flat.norm(dim=1, keepdim=True)
            d_norm = flat / norms.clamp_min(EPS)
            u = unit_direction_dispersion(d_norm.reshape_as(directions))
            u_map = region_direction_dispersion(directions, region_patch=region_patch)
            row = {
                    "tag": tag,
                    "domain": im["domain"],
                    "stem": im["stem"],
                    "bridge_time_index": t,
                    "bridge_time_value": float(times[t]),
                    "U": u,
                    "log_U": math.log(u + EPS),
                    "u_map": u_map.tolist(),
                }
            if extra:
                row.update(extra)
            rows.append(row)
    with (out_dir / f"{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"tag": tag, "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--refactor-root", required=True)
    parser.add_argument("--images", required=True, help="JSON list of {domain,stem,source_path}")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="measure")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bridge-times", default="1,2,3")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--num-timesteps", type=int, default=5)
    parser.add_argument("--region-patch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--extra", default=None, help="JSON object merged into each row")
    args = parser.parse_args()

    images = json.loads(Path(args.images).read_text(encoding="utf-8"))
    bridge_idx = [int(x) for x in args.bridge_times.split(",") if x.strip()]
    times = bridge_times(args.num_timesteps)
    netG = build_generator(args.ckpt, args.refactor_root, args.device)
    extra = json.loads(args.extra) if args.extra else None
    result = extract_one(
        netG,
        images,
        bridge_times_idx=bridge_idx,
        m=args.m,
        ngf=args.ngf,
        tau=args.tau,
        times=times,
        region_patch=args.region_patch,
        device=args.device,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        tag=args.tag,
        extra=extra,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
