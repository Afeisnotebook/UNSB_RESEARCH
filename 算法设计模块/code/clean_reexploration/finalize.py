"""Post-training finalization: evaluate, adjudicate, report and package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/home/yc/unsb_tired")
CODE_ROOT = REPO_ROOT / "算法设计模块/code"
RUNTIME_ROOT = REPO_ROOT / "runtime_4090/clean_reexploration_20260824"
RUNS_ROOT = RUNTIME_ROOT / "runs"
AUTHORITY_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")

sys.path.insert(0, str(CODE_ROOT))

EVAL_DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]


def _paired_delta_bootstrap(rows_a: list[dict], rows_b: list[dict], *, n_draws: int = 999, seed: int) -> dict:
    """Image-cluster (stem nested in domain) paired-delta bootstrap."""
    b_by = {(r["domain"], r["stem"]): r for r in rows_b}
    clusters: dict[str, list[list[float]]] = {d: [] for d in EVAL_DOMAINS}
    for r in rows_a:
        p = b_by.get((r["domain"], r["stem"]))
        if p is None:
            continue
        clusters[r["domain"]].append([r["psnr"] - p["psnr"]])

    from clean_reexploration.controllers import cluster_bootstrap_draws, lower_bound, point_estimate, upper_bound

    draws = cluster_bootstrap_draws(clusters, statistic="mean", n_draws=n_draws, seed=seed)
    per_domain = {}
    positive = 0
    for d in EVAL_DOMAINS:
        units = [v for c in clusters[d] for v in c]
        per_domain[d] = float(np.mean(units)) if units else float("nan")
        if units and per_domain[d] > 0:
            positive += 1
    return {
        "delta_psnr": point_estimate(clusters, "mean"),
        "delta_psnr_ci_low": lower_bound(draws),
        "delta_psnr_ci_high": upper_bound(draws),
        "positive_domains": positive,
        "per_domain_delta": per_domain,
    }


def _evaluate_all(epochs: list[int], replicates: int) -> dict:
    from clean_reexploration import evaluate, identity

    t3 = AUTHORITY_ROOT / "specs/h2c/T3_CONFIRMATORY_MANIFEST.json"
    paired = identity.load_paired_development_manifest(t3)
    lanes = {"canonical_plain": "sb", "hnek_full": "hnek_search", "dt": "sb", "hj": "sb"}
    out: dict = {}
    for lane, model_name in lanes.items():
        out[lane] = {}
        for epoch in epochs:
            ckpt = RUNS_ROOT / lane / f"full_state_e{epoch}.pt"
            if not ckpt.is_file():
                continue
            rows = evaluate.evaluate_checkpoint(
                full_state_path=ckpt,
                model_name=model_name,
                paired_manifest=paired,
                ngf=64,
                num_timesteps=5,
                tau=0.01,
                replicates=replicates,
                seed=2026,
            )
            out[lane][str(epoch)] = {
                "rows": rows,
                "aggregate": evaluate.aggregate(rows),
            }
    return out


def build_evidence(evaluated: dict, *, run_id: str) -> dict:
    """Convert evaluated rows into adjudicator-ready raw evidence."""
    plain = evaluated.get("canonical_plain", {})
    evidence: dict = {
        "schema_version": 1,
        "run_id": run_id,
        "canonical_plain": {},
    }
    e200 = "200"
    if e200 in plain:
        evidence["canonical_plain"]["psnr_macro"] = plain[e200]["aggregate"]["psnr_macro"]

    for lane in ("dt", "hj", "hnek_full"):
        if lane not in evaluated or e200 not in evaluated[lane]:
            continue
        row_a = evaluated[lane][e200]["rows"]
        row_b = plain[e200]["rows"]
        delta = _paired_delta_bootstrap(row_a, row_b, seed=20260824)
        entry = {
            "psnr_macro": evaluated[lane][e200]["aggregate"]["psnr_macro"],
            "ssim_macro": evaluated[lane][e200]["aggregate"]["ssim_macro"],
            **delta,
        }
        evidence[lane] = entry
        if lane == "hnek_full":
            evidence["hnek_full"]["positive_domains_vs_plain"] = delta["positive_domains"]
    return evidence


def build_reports(evidence: dict, *, run_id: str) -> dict:
    from clean_reexploration.adjudicate import adjudicate

    summary = adjudicate(evidence)
    mechanical = {
        "facts": {
            "run_id": run_id,
            "seed": 2026,
            "paired_development": "T3 five-domain, 320 images, single-seed, saturated",
            "canonical_plain_psnr_macro": evidence.get("canonical_plain", {}).get("psnr_macro"),
        },
        "engineering_failures": [],
        "mechanical_results": summary.get("labels", {}),
        "unresolved": [
            "single seed=2026, paired-development, non-confirmatory",
            "no training-seed uncertainty",
            "RainDS-syn has no independent paired-development coverage",
        ],
        "next_action": "等待作者本地审查回传包",
        "author_gate_triggered": False,
    }
    return {"summary": summary, "mechanical": mechanical}


def stage_return(run_id: str, evidence: dict, reports: dict) -> dict:
    from clean_reexploration import identity

    staging = RUNTIME_ROOT / "return_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    def w(rel: str, text: str):
        p = staging / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    w("MECHANICAL_SUMMARY.json", json.dumps(reports["summary"], ensure_ascii=False, indent=2) + "\n")
    w("FINAL_REPORT.md", _final_report_md(run_id, reports))
    w("DECISION_LOG.md", _decision_log_md(reports))
    w("ATTEMPT_LEDGER.json", json.dumps([], ensure_ascii=False, indent=2) + "\n")

    # Checkpoint hash index.
    lines = []
    for lane in ("canonical_plain", "hnek_full", "dt", "hj"):
        d = RUNS_ROOT / lane
        if not d.is_dir():
            continue
        for p in sorted(d.glob("full_state_*.pt")):
            lines.append(f"{identity.sha256_file(p)}  {p.relative_to(RUNTIME_ROOT)}")
    w("CHECKPOINT_SHA256.txt", "\n".join(lines) + "\n")

    # Copy access ledger and prompt.
    ledger = RUNTIME_ROOT / "raw" / "ACCESS_LEDGER.csv"
    if ledger.is_file():
        shutil.copyfile(ledger, staging / "ACCESS_LEDGER.csv")
    prompt = REPO_ROOT / "4090_DTHJ_HNEK_CLEAN_REEXPLORATION_LONG_TASK_PROMPT_CN_20260824.md"
    if prompt.is_file():
        shutil.copyfile(prompt, staging / "TASK_PROMPT_CN_20260824.md")
    spec = CODE_ROOT / "clean_reexploration" / "frozen" / "CLEAN_REEXPLORATION_FROZEN_SPEC.json"
    if spec.is_file():
        dst = staging / "frozen" / "CLEAN_REEXPLORATION_FROZEN_SPEC.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(spec, dst)
    w("CODE_PROVENANCE.json", json.dumps({
        "code_sha256": (RUNTIME_ROOT / "authority" / "CODE_SHA256.txt").read_text().strip(),
        "spec_sha256": (RUNTIME_ROOT / "authority" / "SPEC_CANONICAL_SHA256.txt").read_text().strip(),
        "run_id": run_id,
    }, ensure_ascii=False, indent=2) + "\n")
    w("PAIRED_EVAL_EVIDENCE.json", json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return {"staging": str(staging)}


def _stage_controller_signals(staging: Path) -> None:
    """Compute and stage target-blind controller signals for key checkpoints."""
    from clean_reexploration import controller_audits, diagnostics, identity

    t2 = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806/specs/h2/T2_MANIFEST.json")
    files = identity.load_training_manifest(t2)
    panel = diagnostics.build_diagnostic_panel(files)
    rows = controller_audits._load_panel_rows(panel, files)

    signals = {}
    for lane, model_name in (("hnek_full", "hnek_search"), ("dt", "sb"), ("hj", "sb")):
        for epoch in (20, 50, 100, 200):
            ckpt = RUNS_ROOT / lane / f"full_state_e{epoch}.pt"
            if not ckpt.is_file():
                continue
            from clean_reexploration import evaluate
            netG, _ = evaluate._load_netG(ckpt, model_name)
            netG.eval()
            if lane == "hnek_full":
                sig = controller_audits.compute_hnek_c_h(
                    netG, rows, gamma=0.25, num_timesteps=5, tau=0.01, ngf=64
                )
            else:
                sig = controller_audits.compute_dt_logu(
                    netG, rows, m=4, ngf=64, num_timesteps=5, tau=0.01
                )
            signals[f"{lane}/e{epoch}"] = sig

    p = staging / "CONTROLLER_SIGNALS.json"
    p.write_text(json.dumps(signals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _final_report_md(run_id: str, reports: dict) -> str:
    m = reports["mechanical"]
    labels = m["mechanical_results"]
    return (
        "# FINAL_REPORT\n\n"
        f"run_id: {run_id}\n\n"
        "## 机械结果\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in labels.items())
        + "\n\n## 未决\n\n"
        + "\n".join(f"- {x}" for x in m["unresolved"])
        + "\n\n## 下一动作\n\n等待作者本地审查回传包\n"
    )


def _decision_log_md(reports: dict) -> str:
    m = reports["mechanical"]
    return (
        "# DECISION_LOG\n\n"
        "机械标签仅用于整理，不触发追加实验。\n\n"
        + json.dumps(m["mechanical_results"], ensure_ascii=False, indent=2)
        + "\n"
    )


def finalize_main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=str, default="200")
    p.add_argument("--replicates", type=int, default=4)
    p.add_argument("--run-id", type=str, default="clean-reexploration-s2026-20260824")
    p.add_argument("--zip-name", type=str, default="DTHJ_HNEK_CLEAN_REEXPLORATION_RETURN_20260824.zip")
    args = p.parse_args()

    run_id = (RUNTIME_ROOT / "authority" / "RUN_ID.txt").read_text().strip() or args.run_id
    epochs = [int(e) for e in args.epochs.split(",")]

    evaluated = _evaluate_all(epochs, args.replicates)
    evidence = build_evidence(evaluated, run_id=run_id)
    reports = build_reports(evidence, run_id=run_id)
    stage = stage_return(run_id, evidence, reports)
    _stage_controller_signals(Path(stage["staging"]))

    from clean_reexploration import package_return

    zip_path, sidecar = package_return.package_return(
        staging=Path(stage["staging"]),
        output_dir=RUNTIME_ROOT / "return_staging",
        zip_name=args.zip_name,
    )
    print(json.dumps({
        "finalized": True,
        "zip": str(zip_path),
        "sidecar": str(sidecar),
        "labels": reports["summary"].get("labels", {}),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(finalize_main())
