#!/usr/bin/env python3
"""Stage-window visualisation for the UNSB motivation bypath task.

This script does NOT load checkpoints or train models.  It reads the existing
``raw/*.jsonl`` rows and draws four stage-window figures into
``figures/window/``:

  * window_panel_c_phases.png
  * window_panel_e_stages.png
  * window_panel_d_stages.png
  * window_consensus_score.png

Only Single-task and Plain All-in-One arms are considered.  DT/HJ are excluded.
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
DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
BRIDGE_TIMES = [1, 2, 3]
ARM_COLORS = {
    "single_FoggyCityscapes_s2026": "#4C72B0",
    "single_LowLightTrafficData_s2026": "#DD8452",
    "single_RainCityscapes_s2026": "#C44E52",
    "single_RSCityscapes_s2026": "#55A868",
    "single_SnowTrafficData_s2026": "#8172B3",
    "aio_plain": "#111111",
}
WINDOWS = {
    "initial": (1, 1),
    "compression": (4, 5),
    "redivergence": (20, 20),
}


def short_label(method: str) -> str:
    if method == "aio_plain":
        return "AIO plain"
    for domain in DOMAINS:
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
    """Median log U trajectory with a domain-balanced bootstrap CI."""
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

        domains = sorted(by_domain)
        rng = np.random.default_rng(2026)
        stats = []
        for _ in range(1500):
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
    methods = {r["method"] for r in rows if "method" in r}
    singles = sorted(
        [m for m in methods if m.startswith("single_")],
        key=lambda m: (
            DOMAINS.index(m.removeprefix("single_").removesuffix("_s2026"))
            if m.removeprefix("single_").removesuffix("_s2026") in DOMAINS
            else 999
        ),
    )
    return singles + ["aio_plain"]


def _shade_windows(ax) -> None:
    ax.axvspan(0.5, 1.5, color="#2ca02c", alpha=0.08)
    ax.axvspan(3.5, 5.5, color="#d62728", alpha=0.10)
    ax.axvspan(19.5, 20.5, color="#1f77b4", alpha=0.10)


def plot_panel_c(rows: list[dict], out_path: Path) -> None:
    methods = [m for m in method_order(rows) if m != "aio_dt"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=False)

    for ax, t in zip(axes, BRIDGE_TIMES):
        for method in methods:
            series = domain_balanced_logu_series(rows, method, t)
            if not series:
                continue
            epochs = [s["epoch"] for s in series]
            y = [s["median_log_U"] for s in series]
            lo = [s["ci_low"] for s in series]
            hi = [s["ci_high"] for s in series]
            color = ARM_COLORS.get(method, "#888888")
            lw = 2.6 if method == "aio_plain" else 1.4
            ax.plot(epochs, y, color=color, lw=lw, label=short_label(method))
            ax.fill_between(epochs, lo, hi, color=color, alpha=0.12, lw=0)
        _shade_windows(ax)
        ax.set_xlabel("epoch")
        ax.set_ylabel("median log U")
        ax.set_title(f"bridge time t={t}")
        ax.grid(alpha=0.18)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "panel_c stage-window view: log U trajectories (epoch aligned, bootstrap CI)",
        y=1.02,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _window_diffs_by_domain(
    rows: list[dict],
    window: tuple[int, int],
    t: int,
    metric: str = "U",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Paired AIO-Single differences for one stage and bridge time.

    For every (domain, stem) we average the metric over the window epochs, then
    subtract the corresponding Single value from the AIO value.  The returned
    pooled array contains ten image-stem pairs per domain (domain-equal weight).
    """
    lo_epoch, hi_epoch = window
    aio: dict[tuple[str, str], list[float]] = defaultdict(list)
    single: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        if int(r.get("bridge_time_index", -1)) != t:
            continue
        epoch = int(r.get("epoch", -1))
        if not (lo_epoch <= epoch <= hi_epoch):
            continue
        method = r.get("method")
        key = (r.get("domain") or "", r.get("stem") or "")
        val = float(r.get(metric))
        if method == "aio_plain":
            aio[key].append(val)
        elif method and method.startswith("single_"):
            single[r.get("domain")][key].append(val)

    per_domain: dict[str, np.ndarray] = {}
    pooled = []
    for domain in DOMAINS:
        domain_pairs = []
        for key, aio_vals in aio.items():
            if key[0] != domain or key not in single[domain]:
                continue
            aio_mean = float(np.mean(aio_vals))
            single_mean = float(np.mean(single[domain][key]))
            domain_pairs.append(aio_mean - single_mean)
        if domain_pairs:
            arr = np.asarray(domain_pairs, dtype=np.float64)
            per_domain[domain] = arr
            pooled.extend(arr.tolist())
    return per_domain, np.asarray(pooled, dtype=np.float64)


def _sign_string(domain: str, arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"{domain}: empty"
    mean = float(np.mean(arr))
    boot = paired_bootstrap(arr, np.zeros_like(arr), n_boot=3000, seed=2026)
    if boot["ci_low"] is not None and boot["ci_low"] > 0:
        sign = "+"
    elif boot["ci_high"] is not None and boot["ci_high"] < 0:
        sign = "-"
    else:
        sign = "~"
    return f"{domain} {sign} ({mean:+.2e})"


def plot_panel_e(rows: list[dict], out_path: Path) -> None:
    stage_names = list(WINDOWS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.5), squeeze=False)
    vmin = float("inf")
    vmax = float("-inf")
    cell_data = {}

    for si, stage_name in enumerate(stage_names):
        window = WINDOWS[stage_name]
        for ti, t in enumerate(BRIDGE_TIMES):
            per_domain, pooled = _window_diffs_by_domain(rows, window, t, metric="U")
            cell_data[(si, ti)] = (per_domain, pooled)
            if pooled.size:
                vmin = min(vmin, float(np.min(pooled)))
                vmax = max(vmax, float(np.max(pooled)))

    if not np.isfinite(vmin):
        vmin, vmax = -1.0, 1.0

    for si, stage_name in enumerate(stage_names):
        for ti, t in enumerate(BRIDGE_TIMES):
            ax = axes[si, ti]
            per_domain, pooled = cell_data[(si, ti)]
            if pooled.size:
                boot = paired_bootstrap(pooled, np.zeros_like(pooled), n_boot=5000, seed=2026)
                ax.hist(pooled, bins=12, color="#4C72B0", alpha=0.72, density=True)
                ax.axvline(float(np.mean(pooled)), color="#111111", lw=1.8)
                if boot["ci_low"] is not None:
                    ax.axvspan(boot["ci_low"], boot["ci_high"], color="#111111", alpha=0.12)
                annotation = "\n".join([_sign_string(d, per_domain[d]) for d in DOMAINS if d in per_domain])
                ax.text(
                    0.02,
                    0.97,
                    annotation,
                    va="top",
                    ha="left",
                    transform=ax.transAxes,
                    fontsize=6,
                    color="#333333",
                )
                ax.set_title(
                    f"{stage_name} / t={t}\nmean {float(np.mean(pooled)):.3e}, "
                    f"CI [{boot['ci_low']:.3e}, {boot['ci_high']:.3e}]",
                    fontsize=8,
                )
            else:
                ax.text(0.5, 0.5, "no pairs", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{stage_name} / t={t}", fontsize=8)
            ax.set_xlabel("AIO U - Single U")
            ax.set_ylabel("density")
            ax.set_xlim(vmin, vmax)

    fig.suptitle("panel_e stage-window paired differences (AIO plain - Single)", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _average_u_map(
    rows: list[dict],
    method: str | None,
    window: tuple[int, int],
    t: int,
    domain: str | None = None,
) -> np.ndarray | None:
    maps = []
    lo, hi = window
    for r in rows:
        if int(r.get("bridge_time_index", -1)) != t:
            continue
        epoch = int(r.get("epoch", -1))
        if not (lo <= epoch <= hi):
            continue
        if method is not None and r.get("method") != method:
            continue
        if domain is not None and r.get("domain") != domain:
            continue
        if method is None and not r.get("method", "").startswith("single_"):
            continue
        if "u_map" not in r:
            continue
        maps.append(np.asarray(r["u_map"], dtype=np.float64))
    if not maps:
        return None
    return np.mean(np.stack(maps, axis=0), axis=0)


def plot_panel_d(rows: list[dict], out_path: Path) -> None:
    stages = [("compression", WINDOWS["compression"]), ("redivergence", WINDOWS["redivergence"])]
    methods = ["aio_plain"] + [f"single_{d}_s2026" for d in DOMAINS]
    n_methods = len(methods)
    fig = plt.figure(figsize=(15.0, 7.2))
    gs = fig.add_gridspec(2, n_methods * 3, hspace=0.32, wspace=0.18)

    all_vals = []
    cell = {}
    for si, (stage_name, window) in enumerate(stages):
        for mi, method in enumerate(methods):
            for ti, t in enumerate(BRIDGE_TIMES):
                if method == "aio_plain":
                    arr = _average_u_map(rows, "aio_plain", window, t)
                else:
                    domain = method.removeprefix("single_").removesuffix("_s2026")
                    arr = _average_u_map(rows, method, window, t, domain=domain)
                cell[(si, mi, ti)] = arr
                if arr is not None:
                    all_vals.append(arr)

    vmin = float(np.percentile(np.asarray(all_vals), 5)) if all_vals else 0.0
    vmax = float(np.percentile(np.asarray(all_vals), 95)) if all_vals else 1.0

    for si, (stage_name, window) in enumerate(stages):
        for mi, method in enumerate(methods):
            label = short_label(method)
            special = label in {"Rain", "RSCityscapes"}
            for ti, t in enumerate(BRIDGE_TIMES):
                col = mi * 3 + ti
                ax = fig.add_subplot(gs[si, col])
                arr = cell[(si, mi, ti)]
                if arr is None:
                    ax.text(0.5, 0.5, "NA", ha="center", va="center", transform=ax.transAxes)
                else:
                    im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if si == 0 and mi == 0:
                    ax.set_title(f"t={t}", fontsize=8)
                if ti == 0:
                    ax.set_ylabel(label, fontsize=8, color="#C44E52" if special else "#333333")

    for si, (stage_name, _) in enumerate(stages):
        ax0 = fig.add_subplot(gs[si, 0])
        pos = ax0.get_position()
        fig.text(
            pos.x0 - 0.08,
            pos.y0 + pos.height / 2,
            "Epoch 4-5" if si == 0 else "Epoch 20",
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )

    fig.colorbar(im, ax=fig.axes, location="right", shrink=0.72, label="mean U_reg")
    fig.suptitle(
        "panel_d stage-window spatial U_reg (Rain and RSCityscapes marked red)",
        y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cell_metrics(rows: list[dict]) -> dict:
    """Per-epoch paired log U differences across domain x bridge-time cells."""
    out = {}
    epochs = sorted({int(r["epoch"]) for r in rows if "epoch" in r})
    for epoch in epochs:
        cell_means = []
        cell_ci_sep = 0
        domain_cell_means = defaultdict(list)
        for domain in DOMAINS:
            single_method = f"single_{domain}_s2026"
            domain_cells = []
            for t in BRIDGE_TIMES:
                aio_vals: dict[str, list[float]] = defaultdict(list)
                single_vals: dict[str, list[float]] = defaultdict(list)
                for r in rows:
                    if int(r.get("epoch", -1)) != epoch or int(r.get("bridge_time_index", -1)) != t:
                        continue
                    method = r.get("method")
                    key = r.get("stem")
                    val = float(r.get("log_U"))
                    if method == "aio_plain" and r.get("domain") == domain:
                        aio_vals[key].append(val)
                    elif method == single_method:
                        single_vals[key].append(val)
                diffs = []
                for key, aio_list in aio_vals.items():
                    if key in single_vals:
                        diffs.append(float(np.mean(aio_list)) - float(np.mean(single_vals[key])))
                if diffs:
                    arr = np.asarray(diffs, dtype=np.float64)
                    mean = float(np.mean(arr))
                    cell_means.append(mean)
                    domain_cells.append(mean)
                    boot = paired_bootstrap(arr, np.zeros_like(arr), n_boot=1500, seed=2026)
                    if boot["ci_low"] is not None and (boot["ci_low"] > 0 or boot["ci_high"] < 0):
                        cell_ci_sep += 1
            if domain_cells:
                domain_cell_means[domain] = float(np.mean(domain_cells))

        if not cell_means:
            continue
        dom_vals = np.asarray([domain_cell_means[d] for d in DOMAINS if d in domain_cell_means])
        majority_sign = float(np.sign(np.median(dom_vals))) if dom_vals.size else 0.0
        agree = float(np.mean(np.sign(dom_vals) == majority_sign)) if dom_vals.size else float("nan")
        signed_consensus = majority_sign * agree if dom_vals.size else float("nan")
        out[epoch] = {
            "signed_consensus": signed_consensus,
            "n_ci_separated": cell_ci_sep,
            "pooled_effect": float(np.mean(cell_means)),
            "domain_signs": domain_cell_means,
        }
    return out


def plot_consensus(rows: list[dict], out_path: Path) -> None:
    metrics = _cell_metrics(rows)
    epochs = sorted(metrics)
    signed = [metrics[e]["signed_consensus"] for e in epochs]
    n_sep = [metrics[e]["n_ci_separated"] for e in epochs]
    effect = [metrics[e]["pooled_effect"] for e in epochs]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    ax1, ax2, ax3 = axes

    _shade_windows(ax1)
    ax1.plot(epochs, signed, marker="o", color="#111111")
    ax1.axhline(0, color="0.55", lw=0.8)
    ax1.set_ylabel("signed cross-domain consensus\n(majority sign x agreement)")
    ax1.set_title("window consensus score: Epoch 1 positive, Epoch 4-5 negative, Epoch 20 positive")

    _shade_windows(ax2)
    ax2.plot(epochs, n_sep, marker="s", color="#d62728")
    ax2.set_ylabel("domain-bridge cells with\nbootstrap CI separated from zero")
    ax2.set_ylim(0, 15.5)

    _shade_windows(ax3)
    ax3.plot(epochs, effect, marker="o", color="#1f77b4")
    ax3.axhline(0, color="0.55", lw=0.8)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("pooled effect size\n(mean AIO-Single log U)")

    for ax in axes:
        ax.grid(alpha=0.18)
        ax.set_xlim(0.5, 20.5)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(raw_dir)

    plot_panel_c(rows, out_dir / "window_panel_c_phases.png")
    plot_panel_e(rows, out_dir / "window_panel_e_stages.png")
    plot_panel_d(rows, out_dir / "window_panel_d_stages.png")
    plot_consensus(rows, out_dir / "window_consensus_score.png")

    print(json.dumps({"n_rows": len(rows), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
