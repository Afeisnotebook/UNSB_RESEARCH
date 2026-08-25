#!/usr/bin/env python3
"""Render motivation evidence panels b/c/d/e from raw measurement artifacts.

Inputs are the raw JSONL rows and panel_b NPZ/PCA files produced by
``run_measure.py`` (and the optional ``MOTIVATION_SUMMARY.json`` produced by
``summarize_motivation.py``).  The script intentionally plots evidence rather
than interpreting it; all labels are kept descriptive.

Panels:
  panel_b -- joint PCA projection and per-arm covariance ellipses
  panel_c -- median(log U) over epochs with domain-balanced bootstrap CIs
  panel_d -- per-region U_reg heatmaps, averaged over images and domains
  panel_e -- image-level paired AIO - Single U difference distributions
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from measure_path_geometry import paired_bootstrap


EPS = 1e-8
DEFAULT_DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
ARM_COLORS = {
    "single_FoggyCityscapes_s2026": "#4C72B0",
    "single_LowLightTrafficData_s2026": "#DD8452",
    "single_RainCityscapes_s2026": "#55A868",
    "single_RSCityscapes_s2026": "#C44E52",
    "single_SnowTrafficData_s2026": "#8172B3",
    "aio_plain": "#111111",
    "aio_dt": "#FF8C00",
}


def short_label(method: str) -> str:
    if method == "aio_plain":
        return "AIO plain"
    if method == "aio_dt":
        return "AIO DT (sanity)"
    for domain in DEFAULT_DOMAINS:
        if method == f"single_{domain}_s2026":
            return domain.replace("TrafficData", "").replace("Cityscapes", "")
    return method


def load_rows(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def domain_balanced_logu_series(rows: list[dict], method: str, t: int) -> list[dict]:
    """Return epoch -> median and bootstrap CI for one method/bridge time."""
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("method") != method:
            continue
        if int(r.get("bridge_time_index", -1)) != t:
            continue
        by_epoch[int(r["epoch"])].append(r)

    out = []
    for epoch in sorted(by_epoch):
        sub = by_epoch[epoch]
        by_domain: dict[str, list[float]] = defaultdict(list)
        for r in sub:
            by_domain[r.get("domain") or "unknown"].append(float(r["log_U"]))
        medians = [float(np.median(v)) for v in by_domain.values()]
        median = float(np.median(medians)) if medians else float("nan")

        # Domain-balanced bootstrap CI (same logic as summarize_motivation.py).
        domains = sorted(by_domain)
        rng = np.random.default_rng(2026)
        stats = []
        for _ in range(2000):
            dom_meds = []
            for d in domains:
                vals = np.asarray(by_domain[d], dtype=np.float64)
                idx = rng.integers(0, len(vals), size=len(vals))
                dom_meds.append(float(np.median(vals[idx])))
            stats.append(float(np.median(dom_meds)))
        stats = np.asarray(stats)
        out.append(
            {
                "epoch": epoch,
                "median_log_U": median,
                "ci_low": float(np.quantile(stats, 0.025)),
                "ci_high": float(np.quantile(stats, 0.975)),
            }
        )
    return out


def method_order(rows: list[dict]) -> list[str]:
    methods = sorted({r["method"] for r in rows if "method" in r})
    singles = [m for m in methods if m.startswith("single_")]
    singles = sorted(singles, key=lambda m: DEFAULT_DOMAINS.index(m.removeprefix("single_").removesuffix("_s2026")) if m.removeprefix("single_").removesuffix("_s2026") in DEFAULT_DOMAINS else 999)
    others = [m for m in methods if m not in singles]
    return singles + sorted(others)


def _scatter_or_placeholder(ax, title: str) -> None:
    ax.text(
        0.5,
        0.5,
        "raw evidence missing",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.45",
    )
    ax.set_title(title)


def plot_panel_b(raw_dir: Path, out_path: Path) -> None:
    pca_path = raw_dir / "panel_b_pca.json"
    if not pca_path.exists():
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_or_placeholder(ax, "panel_b: joint PCA")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    pca = json.loads(pca_path.read_text(encoding="utf-8"))
    proj = np.asarray(pca["proj"], dtype=np.float64)
    lengths = pca.get("method_lengths", {})
    singular_values = np.asarray(pca.get("singular_values", []), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    start = 0
    plotted = False
    for method, n in lengths.items():
        end = start + int(n)
        block = proj[start:end]
        if block.size:
            plotted = True
            color = ARM_COLORS.get(method, "#888888")
            ax.scatter(
                block[:, 0],
                block[:, 1],
                s=14,
                alpha=0.42,
                color=color,
                label=short_label(method),
                edgecolors="none",
            )
            if len(block) >= 2:
                cov = np.cov(block.T)
                _draw_cov_ellipse(ax, block.mean(axis=0), cov, color=color, n_std=1.0)
        start = end

    if plotted:
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("joint PCA component 1")
        ax.set_ylabel("joint PCA component 2")
        if singular_values.size:
            ax.set_title(
                "panel_b: joint unit-direction PCA\n"
                + ", ".join(f"PC{i+1} singular value {v:.2e}" for i, v in enumerate(singular_values))
            )
        else:
            ax.set_title("panel_b: joint unit-direction PCA")
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.set_aspect("auto")
    else:
        _scatter_or_placeholder(ax, "panel_b: joint PCA")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_cov_ellipse(ax, mean, cov, *, color, n_std: float = 1.0) -> None:
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0.0))
    from matplotlib.patches import Ellipse

    ax.add_patch(
        Ellipse(
            mean,
            width,
            height,
            angle=theta,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            alpha=0.8,
        )
    )


def plot_panel_c(rows: list[dict], out_path: Path, bridge_times: list[int]) -> None:
    methods = method_order(rows)
    if not methods:
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_or_placeholder(ax, "panel_c: log U trajectory")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, len(bridge_times), figsize=(5.2 * len(bridge_times), 5.0), sharey=False)
    if len(bridge_times) == 1:
        axes = [axes]

    for ax, t in zip(axes, bridge_times):
        for method in methods:
            series = domain_balanced_logu_series(rows, method, t)
            if not series:
                continue
            epochs = [s["epoch"] for s in series]
            y = [s["median_log_U"] for s in series]
            lo = [s["ci_low"] for s in series]
            hi = [s["ci_high"] for s in series]
            color = ARM_COLORS.get(method, "#888888")
            lw = 2.4 if method == "aio_plain" else 1.4
            ls = "--" if method == "aio_dt" else "-"
            ax.plot(epochs, y, color=color, lw=lw, ls=ls, label=short_label(method))
            ax.fill_between(epochs, lo, hi, color=color, alpha=0.12, lw=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel("median log U")
        ax.set_title(f"bridge time index t={t}")
        ax.grid(alpha=0.18)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("panel_c: median(log U) trajectory (domain-balanced, bootstrap 95% CI)", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _average_u_maps(rows: list[dict], method: str, t: int) -> np.ndarray | None:
    maps = []
    for r in rows:
        if r.get("method") != method or int(r.get("bridge_time_index", -1)) != t:
            continue
        if "u_map" not in r:
            continue
        maps.append(np.asarray(r["u_map"], dtype=np.float64))
    if not maps:
        return None
    stacked = np.stack(maps, axis=0)
    return float(np.mean(stacked, axis=0).sum())


def plot_panel_d(rows: list[dict], out_path: Path, bridge_times: list[int]) -> None:
    methods = method_order(rows)
    if not methods:
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_or_placeholder(ax, "panel_d: region U heatmap")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(methods), len(bridge_times), figsize=(3.4 * len(bridge_times), 2.5 * len(methods)), squeeze=False)
    all_vals = []
    per_cell = {}
    for i, method in enumerate(methods):
        for j, t in enumerate(bridge_times):
            maps = []
            for r in rows:
                if r.get("method") != method or int(r.get("bridge_time_index", -1)) != t:
                    continue
                if "u_map" in r:
                    maps.append(np.asarray(r["u_map"], dtype=np.float64))
            if maps:
                arr = np.mean(np.stack(maps, axis=0), axis=0)
            else:
                arr = None
            per_cell[(i, j)] = arr
            if arr is not None:
                all_vals.append(arr.ravel())

    vmin = float(np.percentile(np.concatenate(all_vals), 5)) if all_vals else 0.0
    vmax = float(np.percentile(np.concatenate(all_vals), 95)) if all_vals else 1.0

    for i, method in enumerate(methods):
        axes[i, 0].set_ylabel(short_label(method), fontsize=9)
        for j, t in enumerate(bridge_times):
            ax = axes[i, j]
            arr = per_cell[(i, j)]
            if arr is None:
                _scatter_or_placeholder(ax, "")
            else:
                im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={t}", fontsize=9)

    if all_vals:
        fig.colorbar(im, ax=axes, location="right", shrink=0.85, label="mean U_reg")
    fig.suptitle("panel_d: per-region U_reg heatmaps (image/domain averaged)", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rows_for_method_epoch(rows: list[dict], method_prefix: str, t: int) -> dict[tuple[str, str], dict]:
    out = {}
    for r in rows:
        method = r.get("method", "")
        if method_prefix == "aio_plain":
            if method != "aio_plain":
                continue
        elif not method.startswith(method_prefix):
            continue
        if int(r.get("bridge_time_index", -1)) != t:
            continue
        key = (r.get("domain") or "", r.get("stem") or "")
        out[key] = r
    return out


def plot_panel_e(rows: list[dict], out_path: Path, bridge_times: list[int], domains: list[str]) -> None:
    fig, axes = plt.subplots(1, len(bridge_times), figsize=(4.8 * len(bridge_times), 4.5), sharey=False)
    if len(bridge_times) == 1:
        axes = [axes]

    for ax, t in zip(axes, bridge_times):
        aio_rows = _rows_for_method_epoch(rows, "aio_plain", t)
        diffs = []
        per_domain = {}
        for domain in domains:
            single_rows = _rows_for_method_epoch(rows, f"single_{domain}_s2026", t)
            vals = []
            for key, aio_row in aio_rows.items():
                if key[0] != domain or key not in single_rows:
                    continue
                vals.append(float(aio_row["U"]) - float(single_rows[key]["U"]))
            if vals:
                per_domain[domain] = np.asarray(vals, dtype=np.float64)
                diffs.extend(vals)

        if diffs:
            diffs = np.asarray(diffs, dtype=np.float64)
            boot = paired_bootstrap(
                diffs + 0.0,  # paired_bootstrap expects two arrays; passing same array computes CI on the values
                np.zeros_like(diffs),
                n_boot=5000,
                seed=2026,
            )
            ax.hist(diffs, bins=max(8, int(math.sqrt(len(diffs)))), color="#4C72B0", alpha=0.75, density=True)
            ax.axvline(float(np.mean(diffs)), color="#111111", lw=1.8)
            ax.axvspan(boot["ci_low"], boot["ci_high"], color="#111111", alpha=0.12)
            ax.set_xlabel("AIO U - Single U")
            ax.set_ylabel("density")
            ax.set_title(f"t={t}\nmean {float(np.mean(diffs)):.3f}, 95% CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]", fontsize=8)
        else:
            _scatter_or_placeholder(ax, f"t={t}: no paired rows")

    fig.suptitle("panel_e: paired image-level U difference (AIO plain - Single)", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--domains",
        default="FoggyCityscapes,LowLightTrafficData,RainCityscapes,RSCityscapes,SnowTrafficData",
    )
    parser.add_argument("--bridge-times", default="1,2,3")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_times = [int(x) for x in args.bridge_times.split(",") if x.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    rows = load_rows(raw_dir)

    plot_panel_b(raw_dir, out_dir / "panel_b.png")
    plot_panel_c(rows, out_dir / "panel_c.png", bridge_times)
    plot_panel_d(rows, out_dir / "panel_d.png", bridge_times)
    plot_panel_e(rows, out_dir / "panel_e.png", bridge_times, domains)
    print(json.dumps({"n_rows": len(rows), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
