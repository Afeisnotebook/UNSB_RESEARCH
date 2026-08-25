#!/usr/bin/env python3
"""Orchestrate mechanism screening and render the five required figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    os.environ.get("UNSB_MOTIVATION_ROOT", Path(__file__).resolve().parent.parent)
).expanduser().resolve()
REPO_ROOT = ROOT.parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
PY = os.environ.get("UNSB_PYTHON", sys.executable)
REFACTOR = str(
    Path(
        os.environ.get(
            "UNSB_BASELINE_ROOT", REPO_ROOT / "foundation" / "canonical" / "src"
        )
    ).expanduser().resolve()
)
RUN_ROOT = Path(
    os.environ.get("UNSB_MOTIVATION_RUN_ROOT", REPO_ROOT / "runs" / "MOT-001")
).expanduser().resolve()
REPORT_DIR = RUN_ROOT / "reports" / "mechanism"
FIGURE_DIR = RUN_ROOT / "figures" / "mechanism"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_smoke() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        [
            PY,
            str(SCRIPT_ROOT / "mechanism_direction_rank.py"),
            "--root",
            str(ROOT),
            "--refactor-root",
            REFACTOR,
            "--epochs",
            "1,4",
            "--m",
            "8",
            "--device",
            "cuda",
            "--out-json",
            str(REPORT_DIR / "smoke_direction_rank.json"),
        ]
    )
    _run(
        [
            PY,
            str(SCRIPT_ROOT / "mechanism_gradient.py"),
            "--root",
            str(ROOT),
            "--refactor-root",
            REFACTOR,
            "--epochs",
            "1,4",
            "--m",
            "8",
            "--device",
            "cuda",
            "--out-json",
            str(REPORT_DIR / "smoke_gradient_proxy.json"),
        ]
    )
    _run(
        [
            PY,
            str(SCRIPT_ROOT / "mechanism_feature_alignment.py"),
            "--root",
            str(ROOT),
            "--refactor-root",
            REFACTOR,
            "--epochs",
            "1,4",
            "--z-samples",
            "2",
            "--device",
            "cuda",
            "--out-json",
            str(REPORT_DIR / "smoke_feature_alignment.json"),
        ]
    )
    print("smoke done")


def run_full() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        [
            PY,
            str(SCRIPT_ROOT / "mechanism_direction_rank.py"),
            "--root",
            str(ROOT),
            "--refactor-root",
            REFACTOR,
            "--epochs",
            "1,3,4,5,6,17,20",
            "--m",
            "32",
            "--device",
            "cuda",
            "--out-json",
            str(REPORT_DIR / "direction_rank.json"),
        ]
    )
    _run(
        [
            PY,
            str(SCRIPT_ROOT / "mechanism_gradient.py"),
            "--root",
            str(ROOT),
            "--refactor-root",
            REFACTOR,
            "--epochs",
            "1,3,4,5,6,17,20",
            "--m",
            "32",
            "--device",
            "cuda",
            "--out-json",
            str(REPORT_DIR / "gradient_proxy.json"),
        ]
    )
    _run(
        [
            PY,
            str(SCRIPT_ROOT / "mechanism_feature_alignment.py"),
            "--root",
            str(ROOT),
            "--refactor-root",
            REFACTOR,
            "--epochs",
            "1,3,4,5,6,17,20",
            "--z-samples",
            "8",
            "--device",
            "cuda",
            "--out-json",
            str(REPORT_DIR / "feature_alignment.json"),
        ]
    )
    _run(
        [
            PY,
            str(SCRIPT_ROOT / "mechanism_compression_window.py"),
            "--raw-dir",
            str(ROOT / "raw"),
            "--direction-rank-json",
            str(REPORT_DIR / "direction_rank.json"),
            "--gradient-proxy-json",
            str(REPORT_DIR / "gradient_proxy.json"),
            "--feature-alignment-json",
            str(REPORT_DIR / "feature_alignment.json"),
            "--out-json",
            str(REPORT_DIR / "compression_window.json"),
        ]
    )
    print("full mechanism run done")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grad = _load_json(REPORT_DIR / "gradient_proxy.json")
    rank = _load_json(REPORT_DIR / "direction_rank.json")
    feat = _load_json(REPORT_DIR / "feature_alignment.json")
    comp = _load_json(REPORT_DIR / "compression_window.json")

    epochs = [int(e) for e in sorted(grad["epochs"], key=int)]

    # Figure 1: gradient proxy
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for t in ["t1", "t2", "t3"]:
        y = [grad["epochs"][str(e)].get(t, {}).get("mean_cosine", float("nan")) for e in epochs]
        ax.plot(epochs, y, marker="o", label=t)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-domain direction cosine proxy")
    ax.set_title("gradient conflict proxy (higher cosine = lower conflict)")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.savefig(FIGURE_DIR / "gradient_conflict.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: direction rank trajectory
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    aio_rank = [rank["aio"][str(e)].get("overall", {}).get("effective_rank", float("nan")) for e in epochs]
    single_rank = []
    for e in epochs:
        vals = [
            rank["single"][d][str(e)].get("effective_rank", float("nan"))
            for d in rank["single"]
            if str(e) in rank["single"][d]
        ]
        single_rank.append(float(np.nanmedian(vals)) if vals else float("nan"))
    axes[0].plot(epochs, aio_rank, marker="o", label="AIO")
    axes[0].plot(epochs, single_rank, marker="s", label="Single median")
    axes[0].set_title("effective rank")
    axes[0].legend()

    for ax, key, title in [
        (axes[1], "top3_energy", "top-3 energy"),
        (axes[2], "spectral_entropy", "spectral entropy"),
    ]:
        y = [rank["aio"][str(e)].get("overall", {}).get(key, float("nan")) for e in epochs]
        ax.plot(epochs, y, marker="o")
        ax.set_title(title)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.18)
    fig.suptitle("direction-field rank/spectrum (AIO)")
    fig.savefig(FIGURE_DIR / "direction_rank_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: feature alignment
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    layers = feat.get("layers", [])
    for layer in layers:
        y = [
            feat["epochs"][str(e)].get(layer, {}).get("mean_cka", float("nan"))
            for e in epochs
            if str(e) in feat["epochs"] and isinstance(feat["epochs"][str(e)], dict)
        ]
        ax.plot(epochs[: len(y)], y, marker="o", label=layer)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean pairwise domain CKA")
    ax.set_title("feature alignment proxy")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.18)
    fig.savefig(FIGURE_DIR / "feature_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: relation to compression window
    comp_rows = {int(r["epoch"]): r for r in comp["epochs"]}
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)
    keys = ["aio_logU_z_compression", "effective_rank_z_compression", "feature_cka_z_compression", "gradient_proxy_cosine_z_compression"]
    for key in keys:
        y = [comp_rows[e].get(key, float("nan")) for e in epochs]
        axes[0].plot(epochs, y, marker="o", label=key.replace("_z_compression", ""))
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_title("component z-scores (positive = more compressed)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.18)
    y = [comp_rows[e].get("compression_score", float("nan")) for e in epochs]
    axes[1].plot(epochs, y, marker="o", color="#111111")
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_title("multimetric compression score")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.18)
    fig.savefig(FIGURE_DIR / "compression_window_relation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: compression score multimetric
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    y = [comp_rows[e].get("compression_score", float("nan")) for e in epochs]
    axes[0].bar([str(e) for e in epochs], y, color=["#d62728" if v < 0 else "#2ca02c" for v in y])
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_title("compression score by epoch")
    axes[0].grid(axis="y", alpha=0.18)
    for key in keys:
        y = [comp_rows[e].get(key, float("nan")) for e in epochs]
        axes[1].plot(epochs, y, marker="o", label=key.replace("_z_compression", ""))
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("epoch")
    axes[1].set_title("component contributions")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.18)
    fig.savefig(FIGURE_DIR / "compression_score_multimetric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figures written")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "figures"], required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke()
    elif args.mode == "full":
        run_full()
        make_figures()
    else:
        make_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
