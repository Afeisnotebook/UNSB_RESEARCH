"""Per-step authoritative evaluator trace for one mismatched stem."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


FROZEN_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")
UNSB_TIRED_CODE = Path("/home/yc/unsb_tired/算法设计模块/code")
HIST_RUN = Path("/home/yc/unsb_tired/refactor/_runs/hnek_search/runs/hnek_g0.25")

sys.path.insert(0, str(FROZEN_ROOT))
sys.path.insert(0, str(UNSB_TIRED_CODE))
sys.path.insert(0, str(UNSB_TIRED_CODE / "baseline"))

import scripts.hnek.run_hnek_decisive as RD  # noqa: E402
from scripts.final1 import final1_metrics as F1M  # noqa: E402
from scripts.final1 import final1_common as F1C  # noqa: E402
from models.hnek.hnek_search import HnekSearchConfig, install_hnek_search_model  # noqa: E402


SPEC_SHA = "7a3c135847fe31c736b57d3f3eb8e723d3d8d5649c07d9f9bcc0abfdc22f8573"
RUN_ID = "hnek-search-g0.25_residual_physical_all-seed2026"


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _range(t: torch.Tensor):
    x = t.detach().cpu().float()
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


def install_variant(model):
    return install_hnek_search_model(
        model,
        HnekSearchConfig(gamma=0.25, coord="residual", horizon_mode="physical", partial="all"),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="FoggyCityscapes")
    p.add_argument("--stem", default="0353")
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    F1C.apply_strict_determinism()
    RD.install_hnek_model = install_variant
    plain, method = RD.build_training_pair(Path(tempfile.mkdtemp(prefix="trace_build_")))
    payload = RD.restore_checkpoint(
        HIST_RUN / "checkpoints/full_state_e200.pt", plain, method, run_id=RUN_ID, spec_sha=SPEC_SHA
    )
    method.eval()
    device = method.device
    row = next(r for d in RD.t3_rows().values() for r in d if r["stem"] == args.stem and r["domain"] == args.domain)
    x = RD.load_image(row["absolute_path"], 128, device)
    target = RD.load_image(str(Path(row["absolute_path"]).parent.parent / "target" / f"{args.stem}.png"), 128, device)
    bundle = F1M.build_rollout_bundle(SPEC_SHA, args.domain, args.stem, args.replicate, 4 * 64)

    times = F1M.bridge_times(5)
    Xt = x.to(device)
    prev = None
    steps = []
    for t in range(5):
        z = bundle["z"][t].to(device)
        noise = bundle["noise"][t].to(device)
        before = {"hash": _sha_tensor(Xt), "range": _range(Xt)}
        if t > 0:
            delta = times[t] - times[t - 1]
            denom = times[-1] - times[t - 1]
            inter = delta / denom
            scale = delta * (1 - delta / denom)
            Xt = (1 - inter) * Xt + inter * prev + float((scale * 0.01) ** 0.5) * noise
        t_idx = torch.tensor([t], dtype=torch.long, device=device)
        raw = method.netG(Xt, t_idx, z)
        prev = raw.detach()
        steps.append({
            "t": t,
            "bridge_time": float(times[t]),
            "z_hash": _sha_tensor(z),
            "noise_hash": _sha_tensor(noise),
            "Xt_before": before,
            "Xt_after_bridge_noise": {"hash": _sha_tensor(Xt), "range": _range(Xt)},
            "raw_netG": {"hash": _sha_tensor(raw), "range": _range(raw)},
        })

    out = prev
    out01 = F1M.to_unit_range(out)
    target01 = F1M.to_unit_range(target)
    trace = {
        "domain": args.domain,
        "stem": args.stem,
        "replicate": args.replicate,
        "spec_sha256": SPEC_SHA,
        "run_id": RUN_ID,
        "checkpoint_meta": payload["metadata"],
        "lane": "HNEK_METHOD",
        "netG_state_hash": hashlib.sha256(
            method.netG.module.state_dict() if hasattr(method.netG, "module") else method.netG.state_dict()
        ).hexdigest() if False else _sha_state_dict(method.netG),
        "model_class": type(method).__name__,
        "generator_mode": "hnek_search_gamma_0.25_residual_physical_all",
        "hnek_config": {"gamma": 0.25, "coord": "residual", "horizon_mode": "physical", "partial": "all"},
        "input_tensor": {"hash": _sha_tensor(x), "range": _range(x)},
        "target_tensor": {"hash": _sha_tensor(target), "range": _range(target)},
        "bridge_times": [float(v) for v in times],
        "steps": steps,
        "final_output": {"hash": _sha_tensor(out), "range": _range(out)},
        "metric_input_unit": {"hash": _sha_tensor(out01), "range": _range(out01)},
        "psnr": F1M.psnr_unit(out01, target01),
        "ssim": F1M.ssim_unit(out01, target01),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"psnr": trace["psnr"], "ssim": trace["ssim"], "steps": len(steps)}, indent=2))
    return 0


def _sha_state_dict(net):
    import hashlib as h
    digest = h.sha256()
    sd = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    for k in sorted(sd):
        digest.update(k.encode())
        digest.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    import tempfile
    raise SystemExit(main())
