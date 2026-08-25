"""Paired-development evaluator, run only after ``TRAINING_FROZEN.ok``."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path("/home/yc/unsb_tired")
CODE_ROOT = REPO_ROOT / "算法设计模块/code"
RUNTIME_ROOT = Path(
    os.environ.get(
        "UNSB_REPAIR_RUNTIME",
        str(REPO_ROOT / "runtime_4090/clean_reexploration_repair_20260825"),
    )
)
RUNS_ROOT = RUNTIME_ROOT / "runs"
AUTHORITY_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")

sys.path.insert(0, str(CODE_ROOT / "baseline"))
sys.path.insert(0, str(CODE_ROOT))

EVAL_DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]


def _img_to_tensor(path: str) -> torch.Tensor:
    t = transforms.Compose(
        [
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return t(Image.open(path).convert("RGB"))


def _load_netG(full_state_path: Path, model_name: str):
    from clean_reexploration import full_state

    state = full_state.load_full_state(full_state_path)
    sd = state["networks"]["netG"]
    if model_name == "hnek_search":
        from models.hnek_search_model import HnekSearchModel
        from clean_reexploration.train_executor import _make_sb_opt
        opt = _make_sb_opt("eval")
        opt.model = "hnek_search"
        opt.hnek_gamma = 0.25
        opt.hnek_coord = "residual"
        opt.hnek_horizon_mode = "physical"
        opt.hnek_partial = "all"
        model = HnekSearchModel(opt)
    else:
        from models.sb_model import SBModel
        from clean_reexploration.train_executor import _make_sb_opt
        model = SBModel(_make_sb_opt("eval"))
    net = model.netG.module if hasattr(model.netG, "module") else model.netG
    net.load_state_dict(sd)
    net.eval()
    return net, model


def rollout_endpoint(netG, A: torch.Tensor, *, num_timesteps: int, tau: float, ngf: int, z: torch.Tensor, bridge_noise: torch.Tensor) -> torch.Tensor:
    """Run the five-step UNSB rollout and return the final endpoint."""
    times = _bridge_schedule(num_timesteps, A.device)
    Xt = A
    Xt_1 = None
    for t in range(num_timesteps):
        if t > 0:
            delta = times[t] - times[t - 1]
            denom = times[-1] - times[t - 1]
            inter = (delta / denom).reshape(-1, 1, 1, 1)
            scale = (delta * (1.0 - delta / denom)).reshape(-1, 1, 1, 1)
            Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * bridge_noise[t - 1]
        time_idx = (t * torch.ones(size=[A.shape[0]], device=A.device)).long()
        Xt_1 = netG(Xt, time_idx, z[t])
    return Xt_1


def _bridge_schedule(num_timesteps: int, device) -> torch.Tensor:
    incs = np.array([0.0] + [1.0 / (i + 1) for i in range(num_timesteps - 1)], dtype=np.float64)
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 + 0.5 * times
    times = np.concatenate([np.zeros(1), times])
    return torch.tensor(times, dtype=torch.float32).to(device)


def _to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(F.mse_loss(a, b).item())
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    from clean_reexploration.diagnostics import _ssim_numpy
    return _ssim_numpy(a, b)


def evaluate_checkpoint(
    *,
    full_state_path: Path,
    model_name: str,
    paired_manifest: list[dict],
    ngf: int,
    num_timesteps: int,
    tau: float,
    replicates: int,
    seed: int,
) -> list[dict]:
    netG, _ = _load_netG(full_state_path, model_name)
    netG.eval()
    device = next(netG.parameters()).device

    # Group the 64 A evaluation images (with target) per domain.
    a_rows = [f for f in paired_manifest if f["role"] == "T3_A"]
    target_by = {f["stem"]: f for f in paired_manifest if f["role"] == "T3_A_TARGET"}
    rng = np.random.default_rng(seed)

    rows = []
    with torch.no_grad():
        for f in a_rows:
            stem = f["stem"]
            domain = f["domain"]
            target_path = target_by[stem]["absolute_path"]
            A = _img_to_tensor(f["absolute_path"]).unsqueeze(0).to(device)
            T = _img_to_tensor(target_path).unsqueeze(0).to(device)
            T_unit = _to_unit(T)
            psnrs = []
            ssims = []
            for r in range(replicates):
                rng_local = np.random.default_rng([seed, int(stem), r])
                gen_z = torch.Generator().manual_seed(int(rng_local.integers(0, 2**31)))
                gen_n = torch.Generator().manual_seed(int(rng_local.integers(0, 2**31)))
                z = torch.randn(
                    size=[num_timesteps, 1, 4 * ngf],
                    generator=gen_z,
                ).to(device)
                noise = torch.randn(
                    size=[num_timesteps - 1, 1, 3, 128, 128],
                    generator=gen_n,
                ).to(device)
                out = rollout_endpoint(netG, A, num_timesteps=num_timesteps, tau=tau, ngf=ngf, z=z, bridge_noise=noise)
                out_unit = _to_unit(out)
                psnrs.append(psnr(out_unit, T_unit))
                ssims.append(ssim(out_unit, T_unit))
            rows.append({
                "domain": domain,
                "stem": stem,
                "psnr": float(np.mean(psnrs)),
                "ssim": float(np.mean(ssims)),
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    per_domain = {}
    for domain in EVAL_DOMAINS:
        dr = [r for r in rows if r["domain"] == domain]
        per_domain[domain] = {
            "psnr": float(np.mean([r["psnr"] for r in dr])) if dr else float("nan"),
            "ssim": float(np.mean([r["ssim"] for r in dr])) if dr else float("nan"),
            "n": len(dr),
        }
    psnr_macro = float(np.mean([per_domain[d]["psnr"] for d in EVAL_DOMAINS]))
    ssim_macro = float(np.mean([per_domain[d]["ssim"] for d in EVAL_DOMAINS]))
    return {"per_domain": per_domain, "psnr_macro": psnr_macro, "ssim_macro": ssim_macro}


def paired_delta(rows: list[dict], plain_rows: list[dict]) -> dict:
    plain_by = {(r["domain"], r["stem"]): r for r in plain_rows}
    diffs = []
    per_domain_diff = {d: [] for d in EVAL_DOMAINS}
    for r in rows:
        p = plain_by.get((r["domain"], r["stem"]))
        if p is not None:
            d = r["psnr"] - p["psnr"]
            diffs.append(d)
            per_domain_diff[r["domain"]].append(d)
    delta = float(np.mean(diffs))
    positive = sum(1 for d in EVAL_DOMAINS if np.mean(per_domain_diff[d]) > 0)
    ci_low = float(np.quantile(diffs, 0.025)) if diffs else float("nan")
    return {
        "delta_psnr": delta,
        "delta_psnr_ci_low": ci_low,
        "positive_domains": positive,
        "per_domain_delta": {d: float(np.mean(per_domain_diff[d])) if per_domain_diff[d] else float("nan") for d in EVAL_DOMAINS},
    }


def evaluate_main() -> int:
    """Run the full paired-development evaluation and write raw evidence."""
    import argparse
    from clean_reexploration import identity

    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=4)
    p.add_argument("--num-timesteps", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--epochs", type=str, default="1,5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200")
    args = p.parse_args()

    t3 = AUTHORITY_ROOT / "specs/h2c/T3_CONFIRMATORY_MANIFEST.json"
    paired = identity.load_paired_development_manifest(t3)
    epochs = [int(e) for e in args.epochs.split(",")]

    lanes = {
        "canonical_plain": "sb",
        "hnek_full": "hnek_search",
        "dt": "sb",
        "hj": "sb",
    }
    evidence = {"canonical_plain": {}, "hnek_full": {}, "dt": {}, "hj": {}}
    for lane, model_name in lanes.items():
        for epoch in epochs:
            ckpt = RUNS_ROOT / lane / f"full_state_e{epoch}.pt"
            if not ckpt.is_file():
                continue
            rows = evaluate_checkpoint(
                full_state_path=ckpt,
                model_name=model_name,
                paired_manifest=paired,
                ngf=args.ngf,
                num_timesteps=args.num_timesteps,
                tau=args.tau,
                replicates=args.replicates,
                seed=2026,
            )
            evidence[lane][str(epoch)] = aggregate(rows)

    # Deltas vs canonical plain at matched epochs.
    for lane in ("hnek_full", "dt", "hj"):
        for epoch in epochs:
            if str(epoch) in evidence[lane] and str(epoch) in evidence["canonical_plain"]:
                evidence[lane][str(epoch)]["delta_vs_plain"] = {
                    "psnr": evidence[lane][str(epoch)]["psnr_macro"]
                    - evidence["canonical_plain"][str(epoch)]["psnr_macro"],
                }

    out = RUNTIME_ROOT / "raw" / "PAIRED_EVAL_EVIDENCE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired_eval": "ok", "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate_main())
