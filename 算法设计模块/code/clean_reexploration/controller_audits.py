"""Target-blind controller signal computation from saved checkpoints.

These signals are diagnostic evidence only; the mechanical labels come from the
paired evaluator.  Every statistic is computed from the unpaired diagnostic
panel, never from paired targets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path("/home/yc/unsb_tired")
CODE_ROOT = REPO_ROOT / "算法设计模块/code"
RUNTIME_ROOT = Path(
    os.environ.get(
        "UNSB_REPAIR_RUNTIME",
        str(REPO_ROOT / "runtime_4090/clean_reexploration_repair_20260825"),
    )
)
RUNS_ROOT = RUNTIME_ROOT / "runs"

sys.path.insert(0, str(CODE_ROOT / "baseline"))
sys.path.insert(0, str(CODE_ROOT))


def _load_panel_rows(panel: dict, training_manifest: list[dict]) -> list[dict]:
    by_key = {(f["domain"], f["side"], f["stem"]): f for f in training_manifest if f["side"] in ("A", "B")}
    rows = []
    for domain, sides in panel.items():
        for side in ("A", "B"):
            for stem in sides[side]:
                rows.append(by_key[(domain, side, stem)])
    return rows


def compute_hnek_c_h(
    netG,
    panel_rows: list[dict],
    *,
    gamma: float,
    num_timesteps: int,
    tau: float,
    ngf: int,
    seed: int = 2026,
) -> dict:
    """Compute the HNEK cross-time remaining-horizon energy-distance statistic.

    The returned bundle now includes ``repeat_floor`` (the effect-blind 99th
    percentile of independent same-state repeat estimates) and ``repeat_estimates``
    so callers must subtract the floor before checking ``upper <= 0``.
    """
    from clean_reexploration.evaluate import _bridge_schedule, _img_to_tensor
    from clean_reexploration.diagnostics import energy_distance

    device = next(netG.parameters()).device
    times = _bridge_schedule(num_timesteps, device)

    def _raw_clusters(run_seed: int) -> dict:
        per_domain: dict[str, list[float]] = {}
        noise_gen = torch.Generator(device=device).manual_seed(run_seed)
        z_gen = torch.Generator(device=device).manual_seed(run_seed ^ 0x5D4E9A73)
        with torch.no_grad():
            for f in panel_rows:
                A = _img_to_tensor(f["absolute_path"]).unsqueeze(0).to(device)
                Xt = A
                Xt_1 = None
                r_at_t = {}
                for t in range(num_timesteps):
                    if t > 0:
                        delta = times[t] - times[t - 1]
                        denom = times[-1] - times[t - 1]
                        inter = (delta / denom).reshape(-1, 1, 1, 1)
                        scale = (delta * (1.0 - delta / denom)).reshape(-1, 1, 1, 1)
                        Xt = (
                            (1 - inter) * Xt
                            + inter * Xt_1.detach()
                            + (scale * tau).sqrt()
                            * torch.randn(Xt.shape, generator=noise_gen, device=Xt.device, dtype=Xt.dtype)
                        )
                    time_idx = (t * torch.ones(size=[A.shape[0]], device=device)).long()
                    z = torch.randn(size=[A.shape[0], 4 * ngf], device=device, generator=z_gen)
                    Xt_1 = netG(Xt, time_idx, z)
                    h = float(1.0 - times[t].item())
                    if h > 0:
                        r_at_t[t] = ((Xt_1 - Xt) / (h ** gamma)).flatten().cpu().numpy()

                ts = sorted(r_at_t)
                vals = []
                for j in range(len(ts) - 1):
                    vals.append(energy_distance(r_at_t[ts[j]], r_at_t[ts[j + 1]]))
                per_domain.setdefault(f["domain"], []).append(np.mean(vals) if vals else 0.0)

        return {d: [[v] for v in vals] for d, vals in per_domain.items()}

    raw_clusters = _raw_clusters(seed)
    repeat_estimates = []
    for i in range(20):
        clusters_i = _raw_clusters(seed + 1 + i)
        repeat_estimates.append(point_estimate(clusters_i, "mean"))
    finite = [x for x in repeat_estimates if np.isfinite(x)]
    repeat_floor = float(np.quantile(finite, 0.99)) if finite else 0.0

    return {
        "raw_clusters": raw_clusters,
        "repeat_floor": repeat_floor,
        "repeat_estimates": repeat_estimates,
    }


def _legacy_hnek_raw_clusters(netG, panel_rows, *, gamma, num_timesteps, tau, ngf, seed):
    """Backwards-compatible raw-cluster-only variant used by existing callers."""
    return compute_hnek_c_h(
        netG,
        panel_rows,
        gamma=gamma,
        num_timesteps=num_timesteps,
        tau=tau,
        ngf=ngf,
        seed=seed,
    )["raw_clusters"]


def compute_dt_logu(
    netG,
    panel_rows: list[dict],
    *,
    m: int,
    ngf: int,
    num_timesteps: int,
    tau: float,
    seed: int = 2026,
) -> dict:
    """Compute DT signal-normalized region disagreement logU per source cluster."""
    sys.path.insert(0, str(CODE_ROOT / "dt_covmatch"))
    from dtcov.dtcovmatch import compute_direction_statistics
    from clean_reexploration.evaluate import _bridge_schedule, _img_to_tensor

    device = next(netG.parameters()).device
    times = _bridge_schedule(num_timesteps, device)
    logu_by_domain: dict[str, list[float]] = {}
    noise_gen = torch.Generator(device=device).manual_seed(seed)
    z_gen = torch.Generator(device=device).manual_seed(seed ^ 0x5D4E9A73)

    with torch.no_grad():
        for f in panel_rows:
            A = _img_to_tensor(f["absolute_path"]).unsqueeze(0).to(device)
            Xt = A
            Xt_1 = None
            for t in range(num_timesteps):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1.0 - delta / denom)).reshape(-1, 1, 1, 1)
                    Xt = (
                        (1 - inter) * Xt
                        + inter * Xt_1.detach()
                        + (scale * tau).sqrt()
                        * torch.randn(Xt.shape, generator=noise_gen, device=Xt.device, dtype=Xt.dtype)
                    )
                time_idx = (t * torch.ones(size=[A.shape[0]], device=device)).long()
                endpoints = []
                for _ in range(m):
                    z = torch.randn(size=[A.shape[0], 4 * ngf], device=device, generator=z_gen)
                    endpoints.append(netG(Xt, time_idx, z))
                ep = torch.stack(endpoints, dim=0)
                Xt_1 = endpoints[0]
                t_norm = float(times[t].item() / times[-1].item())
                stats = compute_direction_statistics(
                    X_t=Xt, endpoint_samples=ep, t_norm=t_norm,
                    region_patch=32, detach_uncertainty=True, signal_normalize=True,
                )
                logu = torch.log(stats.U_reg_norm.clamp_min(1e-8)).flatten().cpu().numpy()
                logu_by_domain.setdefault(f["domain"], []).extend(logu.tolist())

    return {d: [v] for d, v in logu_by_domain.items()}


def compute_hj_structure_loss(
    netG,
    panel_rows: list[dict],
    *,
    ngf: int,
    num_timesteps: int,
    tau: float,
    seed: int = 2026,
) -> dict:
    """Compute the HJ joint edge+SSIM structure functional on source-only panels."""
    import torch.nn.functional as F
    from clean_reexploration.evaluate import _bridge_schedule, _img_to_tensor

    device = next(netG.parameters()).device
    times = _bridge_schedule(num_timesteps, device)
    structure_by_domain: dict[str, list[float]] = {}
    noise_gen = torch.Generator(device=device).manual_seed(seed)
    z_gen = torch.Generator(device=device).manual_seed(seed ^ 0x5D4E9A73)

    def sobel_mag(x):
        gray = x[:, 0:1] * 0.2989 + x[:, 1:2] * 0.5870 + x[:, 2:3] * 0.1140
        kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        ky = kx.transpose(2, 3)
        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kx)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), ky)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)

    with torch.no_grad():
        for f in panel_rows:
            A = _img_to_tensor(f["absolute_path"]).unsqueeze(0).to(device)
            Xt = A
            Xt_1 = None
            for t in range(num_timesteps):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1.0 - delta / denom)).reshape(-1, 1, 1, 1)
                    Xt = (
                        (1 - inter) * Xt
                        + inter * Xt_1.detach()
                        + (scale * tau).sqrt()
                        * torch.randn(Xt.shape, generator=noise_gen, device=Xt.device, dtype=Xt.dtype)
                    )
                time_idx = (t * torch.ones(size=[A.shape[0]], device=device)).long()
                z = torch.randn(size=[A.shape[0], 4 * ngf], device=device, generator=z_gen)
                Xt_1 = netG(Xt, time_idx, z)
            out = Xt_1
            edge = torch.sqrt((sobel_mag(out) - sobel_mag(A.detach())).square() + 1e-6).mean()
            structure_by_domain.setdefault(f["domain"], []).append(float(edge.item()))

    return {d: [v] for d, v in structure_by_domain.items()}


def audit_main() -> int:
    import argparse
    from clean_reexploration import diagnostics, evaluate, identity

    p = argparse.ArgumentParser()
    p.add_argument("--lane", choices=["dt", "hnek_full", "hj"], default="dt")
    p.add_argument("--epoch", type=int, default=20)
    args = p.parse_args()

    t2 = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806/specs/h2/T2_MANIFEST.json")
    files = identity.load_training_manifest(t2)
    panel = diagnostics.build_diagnostic_panel(files)
    rows = _load_panel_rows(panel, files)

    ckpt = RUNS_ROOT / args.lane / f"full_state_e{args.epoch}.pt"
    netG, _ = evaluate._load_netG(ckpt, "hnek_search" if args.lane == "hnek_full" else "sb")
    netG.eval()

    if args.lane == "hnek_full":
        sig = compute_hnek_c_h(netG, rows, gamma=0.25, num_timesteps=5, tau=0.01, ngf=64, seed=2026)
    else:
        sig = compute_dt_logu(netG, rows, m=4, ngf=64, num_timesteps=5, tau=0.01, seed=2026)
    print(json.dumps(sig, ensure_ascii=False))
    return 0


def determine_hnek_handoff(run_id: str) -> dict:
    """Compute C_H at each HNEK FULL audit epoch and apply the frozen handoff rule.

    Returns the target-blind handoff epoch ``e_star`` (or None when the signal is
    never exhausted) together with the raw audit records.
    """
    from clean_reexploration import controllers, diagnostics, evaluate, identity
    from clean_reexploration.controllers import upper_bound

    t2 = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806/specs/h2/T2_MANIFEST.json")
    files = identity.load_training_manifest(t2)
    panel = diagnostics.build_diagnostic_panel(files)
    rows = _load_panel_rows(panel, files)

    audit_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    records = []
    consecutive = 0
    e_star = None

    for epoch in audit_epochs:
        ckpt = RUNS_ROOT / "hnek_full" / f"full_state_e{epoch}.pt"
        if not ckpt.is_file():
            continue
        netG, _ = evaluate._load_netG(ckpt, "hnek_search")
        netG.eval()
        rollout_seed = controllers.controller_bootstrap_seed(run_id, "HNEK", epoch, "C_H_rollout")
        sig = compute_hnek_c_h(
            netG, rows, gamma=0.25, num_timesteps=5, tau=0.01, ngf=64,
            seed=rollout_seed,
        )
        clusters = sig["raw_clusters"]
        repeat_floor = float(sig.get("repeat_floor", 0.0))
        seed = controllers.controller_bootstrap_seed(run_id, "HNEK", epoch, "C_H")
        draws = controllers.cluster_bootstrap_draws(clusters, statistic="mean", n_draws=999, seed=seed)
        point = controllers.point_estimate(clusters, "mean") - repeat_floor
        upper = upper_bound(draws) - repeat_floor
        triggered = upper <= 0.0
        records.append({
            "epoch": epoch,
            "C_H_point": point,
            "C_H_point_unfloored": controllers.point_estimate(clusters, "mean"),
            "C_H_upper": upper,
            "C_H_upper_unfloored": upper_bound(draws),
            "repeat_floor": repeat_floor,
            "triggered": triggered,
        })
        if epoch >= 30:
            if triggered:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 2:
                e_star = epoch
                break

    return {"e_star": e_star, "records": records}


if __name__ == "__main__":
    import json
    raise SystemExit(audit_main())
