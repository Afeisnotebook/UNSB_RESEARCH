"""Repair-candidate evaluator oracle.

Runs ``clean_reexploration/evaluate.py`` against the same hash-locked
historical HNEK g0.25 checkpoint and T3 manifest, with the FINAL-1 canonical
CRN bundle, and compares every raw replicate row to the authoritative raw
evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


FROZEN_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")
UNSB_TIRED_CODE = Path("/home/yc/unsb_tired/算法设计模块/code")
HIST_STATE = Path("/home/yc/unsb_tired/refactor/_runs/hnek_search/state/hnek_g0.25")
CHECKPOINT = Path(
    "/home/yc/unsb_tired/refactor/_runs/hnek_search/runs/hnek_g0.25/checkpoints/full_state_e200.pt"
)
SPEC_SHA = "7a3c135847fe31c736b57d3f3eb8e723d3d8d5649c07d9f9bcc0abfdc22f8573"

import sys
sys.path.insert(0, str(UNSB_TIRED_CODE))
sys.path.insert(0, str(UNSB_TIRED_CODE / "baseline"))
sys.path.insert(0, str(FROZEN_ROOT))

from clean_reexploration import evaluate, identity  # noqa: E402
from scripts.final1 import final1_common as F1C  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    F1C.apply_strict_determinism()
    t3 = FROZEN_ROOT / "specs/h2c/T3_CONFIRMATORY_MANIFEST.json"
    paired = identity.load_paired_development_manifest(t3)
    rows = evaluate.evaluate_checkpoint_raw(
        full_state_path=CHECKPOINT,
        model_name="hnek_search",
        paired_manifest=paired,
        ngf=64,
        num_timesteps=5,
        tau=0.01,
        replicates=4,
        spec_sha256=SPEC_SHA,
    )
    out_csv = args.out_dir / "REPAIR_CANDIDATE_REPLICATE_ROWS.csv"
    fields = ["domain", "stem", "replicate", "variant", "bundle_hash", "psnr", "ssim"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    hist = {
        r["bundle_hash"]: r
        for r in csv.DictReader((HIST_STATE / "eval_e200/REPLICATE_ROWS.csv").open())
    }
    new = {r["bundle_hash"]: r for r in rows}
    if set(hist) != set(new):
        comparison = {"ok": False, "reason": "bundle set mismatch"}
    else:
        psnr_err = max(abs(float(new[b]["psnr"]) - float(hist[b]["psnr"])) for b in hist)
        ssim_err = max(abs(float(new[b]["ssim"]) - float(hist[b]["ssim"])) for b in hist)
        macro_new = sum(float(r["psnr"]) for r in rows) / len(rows)
        macro_hist = sum(float(r["psnr"]) for r in hist.values()) / len(hist)
        comparison = {
            "ok": psnr_err <= 1e-5 and ssim_err <= 1e-6 and abs(macro_new - macro_hist) <= 1e-6,
            "psnr_max_abs_error": psnr_err,
            "ssim_max_abs_error": ssim_err,
            "macro_psnr_abs_error": abs(macro_new - macro_hist),
        }
    (args.out_dir / "REPAIR_CANDIDATE_ORACLE.json").write_text(
        json.dumps({"rows": len(rows), "comparison": comparison}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "comparison": comparison}, indent=2))
    return 0 if comparison["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
