#!/usr/bin/env python3
"""Window audit for the UNSB motivation bypath task.

This script reads existing ``raw/*.jsonl`` rows and computes, for a set of
candidate epoch windows, paired AIO-Single differences at both image level (U)
and spatial level (U_reg).  It does not load checkpoints or train models.

Output:
  * a JSON audit summary printed to stdout (and optionally written to disk);
  * three PNG figures for the current raw directory/seed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from measure_path_geometry import paired_bootstrap


DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
CANDIDATE_WINDOWS = [
    ("3-4", 3, 4),
    ("3-5", 3, 5),
    ("3-6", 3, 6),
    ("4-5", 4, 5),
    ("4-6", 4, 6),
    ("5-6", 5, 6),
    ("4-7", 4, 7),
    ("1-1", 1, 1),
    ("20-20", 20, 20),
]


def load_rows(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def detect_methods(rows: list[dict]) -> tuple[list[str], str]:
    methods = {r["method"] for r in rows if "method" in r}
    singles = sorted([m for m in methods if m.startswith("single_")])
    aio_methods = [m for m in methods if m == "aio_plain" or m.startswith("aio_plain")]
    if not aio_methods:
        raise ValueError("no AIO plain method found in raw rows")
    aio = "aio_plain" if "aio_plain" in methods else aio_methods[0]
    return singles, aio


def _paired_image_diffs(
    rows: list[dict],
    *,
    aio_method: str,
    single_method: str,
    domain: str,
    bridge_time: int,
    lo_epoch: int,
    hi_epoch: int,
    metric: str,
) -> np.ndarray:
    aio: dict[str, list[float]] = defaultdict(list)
    single: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if int(r.get("bridge_time_index", -1)) != bridge_time:
            continue
        epoch = int(r.get("epoch", -1))
        if not (lo_epoch <= epoch <= hi_epoch):
            continue
        if metric not in r:
            continue
        method = r.get("method")
        if method == aio_method and r.get("domain") == domain:
            aio[r.get("stem")].append(float(r[metric]))
        elif method == single_method:
            single[r.get("stem")].append(float(r[metric]))
    out = []
    for stem, aio_vals in aio.items():
        if stem in single:
            out.append(float(np.mean(aio_vals)) - float(np.mean(single[stem])))
    return np.asarray(out, dtype=np.float64)


def _bootstrap_diff(diff: np.ndarray) -> dict:
    if diff.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    return paired_bootstrap(diff, np.zeros_like(diff), n_boot=3000, seed=2026)


def image_window_audit(
    rows: list[dict],
    *,
    aio_method: str,
    single_methods: list[str],
    lo_epoch: int,
    hi_epoch: int,
) -> dict:
    """Paired AIO-Single U differences aggregated over a candidate window."""
    domain_signs = {}
    per_domain = {}
    per_t = {}
    cell_means = []

    for domain in DOMAINS:
        sm = next((m for m in single_methods if f"_{domain}_s" in m or m.endswith(f"_{domain}_s")), None)
        if sm is None:
            continue
        dom_t_means = []
        for t in (1, 2, 3):
            diff = _paired_image_diffs(
                rows,
                aio_method=aio_method,
                single_method=sm,
                domain=domain,
                bridge_time=t,
                lo_epoch=lo_epoch,
                hi_epoch=hi_epoch,
                metric="U",
            )
            mean = float(np.mean(diff)) if diff.size else 0.0
            dom_t_means.append(mean)
            cell_means.append(mean)
            per_domain.setdefault(domain, []).append(diff)
        domain_signs[domain] = float(np.sign(np.mean(dom_t_means))) if dom_t_means else 0.0

    # Per-bridge-time pooled across five domains (equal domain weight).
    for t in (1, 2, 3):
        pooled = []
        for domain in DOMAINS:
            if domain in per_domain and len(per_domain[domain]) >= t:
                diff = per_domain[domain][t - 1]
                if diff.size:
                    pooled.extend(diff.tolist())
        arr = np.asarray(pooled, dtype=np.float64)
        per_t[t] = _bootstrap_diff(arr)

    domain_values = [domain_signs[d] for d in DOMAINS if d in domain_signs]
    if domain_values:
        majority_sign = float(np.sign(np.median(domain_values)))
        agree_count = int(np.sum(np.sign(np.asarray(domain_values)) == majority_sign))
        agree_fraction = agree_count / len(domain_values)
        signed_consensus = majority_sign * agree_fraction
    else:
        majority_sign = 0.0
        agree_count = 0
        agree_fraction = float("nan")
        signed_consensus = float("nan")

    return {
        "window": [lo_epoch, hi_epoch],
        "domain_signs": domain_signs,
        "n_domains": len(domain_values),
        "agree_count": agree_count,
        "agree_fraction": agree_fraction,
        "majority_sign": int(majority_sign),
        "signed_consensus": signed_consensus,
        "per_bridge_time": per_t,
        "pooled_cell_mean": float(np.mean(cell_means)) if cell_means else float("nan"),
    }


def ureg_window_audit(
    rows: list[dict],
    *,
    aio_method: str,
    single_methods: list[str],
    lo_epoch: int,
    hi_epoch: int,
) -> dict:
    """Spatial U_reg audit: AIO minus Single mean map sum per domain."""
    per_domain = {}
    for domain in DOMAINS:
        sm = next((m for m in single_methods if f"_{domain}_s" in m), None)
        if sm is None:
            continue
        diffs = []
        for t in (1, 2, 3):
            aio_vals = []
            single_vals = []
            for r in rows:
                if int(r.get("bridge_time_index", -1)) != t:
                    continue
                epoch = int(r.get("epoch", -1))
                if not (lo_epoch <= epoch <= hi_epoch):
                    continue
                if "u_map" not in r:
                    continue
                val = float(np.asarray(r["u_map"], dtype=np.float64).sum())
                method = r.get("method")
                if method == aio_method and r.get("domain") == domain:
                    aio_vals.append(val)
                elif method == sm:
                    single_vals.append(val)
            if aio_vals and single_vals:
                diffs.append(float(np.mean(aio_vals)) - float(np.mean(single_vals)))
        per_domain[domain] = float(np.mean(diffs)) if diffs else 0.0

    vals = list(per_domain.values())
    majority_sign = int(np.sign(np.median(vals))) if vals else 0
    agree_count = int(np.sum(np.sign(np.asarray(vals)) == majority_sign)) if vals else 0
    return {
        "window": [lo_epoch, hi_epoch],
        "per_domain": per_domain,
        "majority_sign": majority_sign,
        "agree_count": agree_count,
        "pooled_mean": float(np.mean(vals)) if vals else float("nan"),
    }


def _plot_window_score(audits: dict, out_path: Path) -> None:
    names = [name for name, _, _ in CANDIDATE_WINDOWS]
    signed = [audits[name]["signed_consensus"] for name in names]
    n_t_same = []
    for name in names:
        per_t = audits[name]["per_bridge_time"]
        signs = []
        for t in (1, 2, 3):
            d = per_t[t]
            if d["mean"] is None or d["ci_low"] is None or d["ci_high"] is None:
                signs.append(0.0)
            elif d["ci_low"] > 0:
                signs.append(1.0)
            elif d["ci_high"] < 0:
                signs.append(-1.0)
            else:
                signs.append(0.0)
        med = float(np.median(signs))
        n_t_same.append(int(np.sum(np.sign(np.asarray(signs)) == np.sign(med))) if med != 0 else 0)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    colors = ["#d62728" if v < 0 else ("#2ca02c" if v > 0 else "#999999") for v in signed]
    axes[0].bar(names, signed, color=colors)
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].axhline(0.8, color="#2ca02c", lw=0.7, ls="--", alpha=0.7)
    axes[0].axhline(-0.8, color="#d62728", lw=0.7, ls="--", alpha=0.7)
    axes[0].set_ylabel("signed consensus\n(majority sign x agreement)")
    axes[0].set_title("seed=2026 candidate-window score")
    axes[1].bar(names, n_t_same, color="#1f77b4")
    axes[1].set_ylabel("bridge times with pooled CI\non same side of zero (max 3)")
    axes[1].set_ylim(0, 3.5)
    axes[1].set_xlabel("candidate epoch window")
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_panel_e(audits: dict, out_path: Path) -> None:
    names = [name for name, _, _ in CANDIDATE_WINDOWS]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=False)
    for ax, t in zip(axes, (1, 2, 3)):
        means = []
        lows = []
        highs = []
        for name in names:
            d = audits[name]["per_bridge_time"][t]
            means.append(d["mean"] if d["mean"] is not None else 0.0)
            lows.append(d["ci_low"] if d["ci_low"] is not None else 0.0)
            highs.append(d["ci_high"] if d["ci_high"] is not None else 0.0)
        x = np.arange(len(names))
        yerr = [
            [m - lo for m, lo in zip(means, lows)],
            [hi - m for m, hi in zip(means, highs)],
        ]
        colors = ["#d62728" if m < 0 else ("#2ca02c" if m > 0 else "#999999") for m in means]
        ax.bar(x, means, yerr=yerr, capsize=3, color=colors, alpha=0.82)
        ax.axhline(0, color="0.4", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"t={t}")
        ax.set_ylabel("pooled AIO-Single U diff")
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("seed=2026 panel_e stage audit: pooled paired U differences by bridge time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_panel_d(audits_ureg: dict, out_path: Path) -> None:
    names = [name for name, _, _ in CANDIDATE_WINDOWS]
    mat = np.array([[audits_ureg[name]["per_domain"].get(d, 0.0) for d in DOMAINS] for name in names])
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    vmax = float(np.max(np.abs(mat))) if mat.size else 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels([d.replace("TrafficData", "").replace("Cityscapes", "") for d in DOMAINS], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(DOMAINS)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=6, color="#111111")
    fig.colorbar(im, ax=ax, shrink=0.8, label="AIO - Single mean U_reg map sum")
    ax.set_title("seed=2026 panel_d stage audit: U_reg spatial-direction summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--audit-json", default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(raw_dir)
    single_methods, aio_method = detect_methods(rows)

    audits = {}
    audits_ureg = {}
    for name, lo, hi in CANDIDATE_WINDOWS:
        audits[name] = image_window_audit(
            rows,
            aio_method=aio_method,
            single_methods=single_methods,
            lo_epoch=lo,
            hi_epoch=hi,
        )
        audits_ureg[name] = ureg_window_audit(
            rows,
            aio_method=aio_method,
            single_methods=single_methods,
            lo_epoch=lo,
            hi_epoch=hi,
        )

    _plot_window_score(audits, out_dir / f"seed{args.seed}_window_score.png")
    _plot_panel_e(audits, out_dir / f"seed{args.seed}_panel_e_stages.png")
    _plot_panel_d(audits_ureg, out_dir / f"seed{args.seed}_panel_d_stages.png")

    result = {
        "seed": args.seed,
        "n_rows": len(rows),
        "aio_method": aio_method,
        "single_methods": single_methods,
        "candidate_windows": audits,
        "ureg_audit": audits_ureg,
    }
    if args.audit_json:
        Path(args.audit_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
