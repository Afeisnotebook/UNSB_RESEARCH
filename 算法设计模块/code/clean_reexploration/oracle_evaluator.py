"""Authoritative evaluator oracle for the HNEK g0.25 historical checkpoint.

This module is not a hand-written approximation.  It imports and executes the
server's original, hash-locked ``scripts.hnek`` / ``scripts.final1`` evaluator
and the same HNEK search adapter used by the historical ``hnek-search-g0.25``
run, then compares every 320-image / 4-replicate raw row against the historical
raw evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


FROZEN_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")
UNSB_TIRED_CODE = Path("/home/yc/unsb_tired/算法设计模块/code")
HIST_STATE = Path(
    "/home/yc/unsb_tired/refactor/_runs/hnek_search/state/hnek_g0.25"
)
HIST_RUN = Path(
    "/home/yc/unsb_tired/refactor/_runs/hnek_search/runs/hnek_g0.25"
)

sys.path.insert(0, str(FROZEN_ROOT))
sys.path.insert(0, str(UNSB_TIRED_CODE))
sys.path.insert(0, str(UNSB_TIRED_CODE / "baseline"))

import scripts.hnek.run_hnek_decisive as RD  # noqa: E402
from scripts.final1 import final1_metrics as F1M  # noqa: E402
from scripts.final1 import final1_common as F1C  # noqa: E402
from models.hnek.hnek_search import (  # noqa: E402
    HnekSearchConfig,
    install_hnek_search_model,
)


SPEC_SHA = "7a3c135847fe31c736b57d3f3eb8e723d3d8d5649c07d9f9bcc0abfdc22f8573"
RUN_ID = "hnek-search-g0.25_residual_physical_all-seed2026"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_variant(model):
    cfg = HnekSearchConfig(
        gamma=0.25,
        coord="residual",
        horizon_mode="physical",
        partial="all",
    )
    return install_hnek_search_model(model, cfg)


def authoritative_code_identity() -> dict:
    files = [
        FROZEN_ROOT / "scripts/hnek/run_hnek_decisive.py",
        FROZEN_ROOT / "scripts/final1/final1_metrics.py",
        FROZEN_ROOT / "scripts/final1/final1_networks.py",
        FROZEN_ROOT / "scripts/final1/final1_common.py",
        UNSB_TIRED_CODE / "baseline/models/hnek/hnek_search.py",
    ]
    manifest = {str(p): sha256_file(p) for p in files}
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"files": manifest, "code_sha256": digest}


def evaluate_authoritative(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # The historical runner applied strict determinism before building and
    # evaluating.  Reproduce that exact environment.
    F1C.apply_strict_determinism()
    # Build the same shared e0 pair as the historical runner, but install the
    # gamma=0.25 search wrapper instead of the legacy 0.5 adapter.
    RD.install_hnek_model = install_variant
    plain, method = RD.build_training_pair(out_dir / "build")
    payload = RD.restore_checkpoint(
        HIST_RUN / "checkpoints/full_state_e200.pt",
        plain,
        method,
        run_id=RUN_ID,
        spec_sha=SPEC_SHA,
    )
    plain.eval()
    method.eval()
    summary = RD.evaluate_pair(
        plain,
        method,
        epoch=200,
        spec_sha=SPEC_SHA,
        run_id=RUN_ID,
        out_dir=out_dir,
        purpose="AUTHORITATIVE_ORACLE",
    )
    return {
        "summary": summary,
        "checkpoint_payload_meta": payload["metadata"],
        "code_identity": authoritative_code_identity(),
    }


def compare_to_historical(out_dir: Path) -> dict:
    hist_path = HIST_STATE / "eval_e200/REPLICATE_ROWS.csv"
    new_path = out_dir / "REPLICATE_ROWS.csv"
    hist = {r["bundle_hash"]: r for r in csv.DictReader(hist_path.open())}
    new = {r["bundle_hash"]: r for r in csv.DictReader(new_path.open())}
    if set(hist) != set(new):
        return {
            "ok": False,
            "reason": "bundle hash set mismatch",
            "missing": sorted(set(hist) - set(new))[:10],
            "extra": sorted(set(new) - set(hist))[:10],
        }
    psnr_err = 0.0
    ssim_err = 0.0
    macro_new = json.loads((out_dir / "SUMMARY.json").read_text())["macro_psnr"]["HNEK_METHOD"]
    macro_hist = json.loads((HIST_STATE / "eval_e200/SUMMARY.json").read_text())["macro_psnr"]["HNEK_METHOD"]
    for bundle, h in hist.items():
        n = new[bundle]
        psnr_err = max(psnr_err, abs(float(n["psnr"]) - float(h["psnr"])))
        ssim_err = max(ssim_err, abs(float(n["ssim"]) - float(h["ssim"])))
    ok = psnr_err <= 1e-5 and ssim_err <= 1e-6 and abs(macro_new - macro_hist) <= 1e-6
    return {
        "ok": ok,
        "psnr_max_abs_error": psnr_err,
        "ssim_max_abs_error": ssim_err,
        "macro_psnr_abs_error": abs(macro_new - macro_hist),
        "historical_macro": macro_hist,
        "new_macro": macro_new,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    result = evaluate_authoritative(args.out_dir)
    comparison = compare_to_historical(args.out_dir)
    (args.out_dir / "AUTHORITATIVE_ORACLE.json").write_text(
        json.dumps({"evaluation": result, "comparison": comparison}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparison": comparison, "code_identity": result["code_identity"]}, indent=2))
    return 0 if comparison["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
