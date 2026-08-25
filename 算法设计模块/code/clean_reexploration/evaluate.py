"""Paired-development evaluator, run only after ``TRAINING_FROZEN.ok``."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
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
AUTHORITY_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(AUTHORITY_ROOT))
sys.path.insert(0, str(CODE_ROOT / "baseline"))

EVAL_DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]


from scripts.final1 import final1_metrics as F1M  # noqa: E402
from scripts.final1 import final1_common as F1C  # noqa: E402
from scripts.final1 import final1_networks as F1N  # noqa: E402
import scripts.hnek.run_hnek_decisive as RD  # noqa: E402


def _load_netG(full_state_path: Path, model_name: str):
    from clean_reexploration import full_state

    # Support both current clean_reexploration full-state and historical
    # HNEK coupled-lane checkpoints.
    payload = torch.load(full_state_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "lanes" in payload:
        sd = payload["lanes"]["HNEK_METHOD"]["networks"]["G"]
    else:
        state = full_state.load_full_state(full_state_path)
        sd = state["networks"]["netG"]
    attempt = Path(tempfile.mkdtemp(prefix="eval_netg_"))
    model, _ = F1N.build_official_model(attempt, seed=2026)
    if model_name == "hnek_search":
        import importlib
        sys.path.insert(0, str(CODE_ROOT / "baseline"))
        if "models" in sys.modules:
            sys.modules.pop("models", None)
        hnek_mod = importlib.import_module("models.hnek.hnek_search")
        HnekSearchConfig = hnek_mod.HnekSearchConfig
        install_hnek_search_model = hnek_mod.install_hnek_search_model
        install_hnek_search_model(
            model,
            HnekSearchConfig(
                gamma=0.25, coord="residual", horizon_mode="physical", partial="all"
            ),
        )
    net = model.netG.module if hasattr(model.netG, "module") else model.netG
    net.load_state_dict(sd)
    net.eval()
    return model, model


def _img_to_tensor(path: str, device) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((128, 128), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)


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
    spec_sha256: str = "",
) -> list[dict]:
    model, _ = _load_netG(full_state_path, model_name)
    model.eval()
    device = next(model.netG.parameters()).device

    a_rows = [f for f in paired_manifest if f["role"] == "T3_A"]
    target_by = {f["stem"]: f for f in paired_manifest if f["role"] == "T3_A_TARGET"}

    rows = []
    with torch.no_grad():
        for f in a_rows:
            stem = f["stem"]
            domain = f["domain"]
            target_path = target_by[stem]["absolute_path"]
            A = _img_to_tensor(f["absolute_path"], device)
            T = _img_to_tensor(target_path, device)
            T_unit = F1M.to_unit_range(T)
            psnrs = []
            ssims = []
            for r in range(replicates):
                bundle = F1M.build_rollout_bundle(
                    spec_sha256, domain, stem, r, 4 * ngf, 128, 128, num_timesteps
                )
                out = F1M.five_step_rollout(model, A, bundle, device)
                out_unit = F1M.to_unit_range(out)
                psnrs.append(F1M.psnr_unit(out_unit, T_unit))
                ssims.append(F1M.ssim_unit(out_unit, T_unit))
            rows.append({
                "domain": domain,
                "stem": stem,
                "psnr": float(np.mean(psnrs)),
                "ssim": float(np.mean(ssims)),
            })
    return rows


def evaluate_checkpoint_raw(
    *,
    full_state_path: Path,
    model_name: str,
    paired_manifest: list[dict],
    ngf: int,
    num_timesteps: int,
    tau: float,
    replicates: int,
    spec_sha256: str,
) -> list[dict]:
    """Return raw replicate rows (1280 for T3) with canonical bundle hashes."""
    if model_name == "hnek_search":
        from models.hnek.hnek_search import HnekSearchConfig, install_hnek_search_model
        def install_variant(model):
            return install_hnek_search_model(
                model,
                HnekSearchConfig(gamma=0.25, coord="residual", horizon_mode="physical", partial="all"),
            )
        RD.install_hnek_model = install_variant
        import tempfile as _tf
        out_dir = Path(_tf.mkdtemp(prefix="candidate_raw_"))
        plain, method = RD.build_training_pair(out_dir / "build")
        RD.restore_checkpoint(
            full_state_path,
            plain,
            method,
            run_id="hnek-search-g0.25_residual_physical_all-seed2026",
            spec_sha=spec_sha256,
        )
        RD.evaluate_pair(
            plain, method, epoch=200, spec_sha=spec_sha256,
            run_id="hnek-search-g0.25_residual_physical_all-seed2026",
            out_dir=out_dir, purpose="REPAIR_CANDIDATE_ORACLE",
        )
        import csv as _csv
        out = []
        for r in _csv.DictReader((out_dir / "REPLICATE_ROWS.csv").open()):
            if r["variant"] == "HNEK_METHOD":
                out.append({
                    "domain": r["domain"],
                    "stem": r["stem"],
                    "replicate": int(r["replicate"]),
                    "variant": r["variant"],
                    "bundle_hash": r["bundle_hash"],
                    "psnr": float(r["psnr"]),
                    "ssim": float(r["ssim"]),
                })
        return out
    model, _ = _load_netG(full_state_path, model_name)
    model.eval()
    device = next(model.netG.parameters()).device
    a_rows = [f for f in paired_manifest if f["role"] == "T3_A"]
    target_by = {f["stem"]: f for f in paired_manifest if f["role"] == "T3_A_TARGET"}
    rows = []
    with torch.no_grad():
        for f in a_rows:
            stem, domain = f["stem"], f["domain"]
            A = _img_to_tensor(f["absolute_path"], device)
            T = _img_to_tensor(target_by[stem]["absolute_path"], device)
            T_unit = F1M.to_unit_range(T)
            for r in range(replicates):
                bundle = F1M.build_rollout_bundle(
                    spec_sha256, domain, stem, r, 4 * ngf, 128, 128, num_timesteps
                )
                out = F1M.five_step_rollout(model, A, bundle, device)
                out_unit = F1M.to_unit_range(out)
                rows.append({
                    "domain": domain,
                    "stem": stem,
                    "replicate": r,
                    "variant": "HNEK_METHOD",
                    "bundle_hash": F1M.bundle_hash(bundle),
                    "psnr": F1M.psnr_unit(out_unit, T_unit),
                    "ssim": F1M.ssim_unit(out_unit, T_unit),
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
