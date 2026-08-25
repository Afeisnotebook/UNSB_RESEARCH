from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RainDS-syn",
    "RSCityscapes",
    "SnowTrafficData",
]
LABELS = ["Fog", "Low-light", "Rain", "RainDS", "Rain-streak", "Snow"]
COLORS = ["#3B6FB6", "#D89B2B", "#2F9E73", "#7B61A8", "#D45D4C", "#5A8F29"]
AGE_COLORS = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.10,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    raw = pd.read_csv(root / "raw" / "RECIPROCAL_KERNEL_BY_AGE.csv")
    stats = json.loads((root / "reports" / "PHASE_STATISTICS.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(root / "reports" / "PHASE_CELL_SUMMARY.csv")
    phase_draws = pd.read_csv(root / "reports" / "PHASE_BOOTSTRAP_DRAWS.csv")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(15.5, 9.2), constrained_layout=False, facecolor="white")
    grid = figure.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        top=0.86,
        bottom=0.095,
        wspace=0.28,
        hspace=0.38,
    )
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    representative_time = 2
    representative = next(
        item for item in stats["time_results"] if item["bridge_time_index"] == representative_time
    )
    domain_ages = [representative["domain_effective_ages"][domain] for domain in DOMAINS]
    common = representative["best_common_age"]

    # A: one shared checkpoint projected onto six task-specific clocks.
    axis = axes[0]
    for row, (domain, label, color, age) in enumerate(
        zip(DOMAINS, LABELS, COLORS, domain_ages, strict=True)
    ):
        y = len(DOMAINS) - 1 - row
        axis.plot(range(1, 7), [y] * 6, color="#D5D8DC", lw=1.6, zorder=1)
        axis.scatter(range(1, 7), [y] * 6, s=18, facecolor="white", edgecolor="#B0B5BA", zorder=2)
        axis.scatter(age, y, s=70, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        axis.text(0.75, y, label, ha="right", va="center", color=color, fontsize=8.5)
        axis.text(age, y + 0.30, f"e{age}", ha="center", va="bottom", color=color, fontsize=7.5)
    axis.axvline(common, color="#202124", ls="--", lw=1.3, zorder=0)
    axis.text(common, 5.55, f"best single clock: e{common}", ha="center", va="bottom", fontsize=8)
    axis.set_xlim(0.65, 6.35)
    axis.set_ylim(-0.6, 5.9)
    axis.set_xticks(range(1, 7))
    axis.set_yticks([])
    axis.set_xlabel("task-specific training phase (epoch)")
    axis.set_title("One AIO checkpoint occupies different task phases")
    axis.spines[["left", "right", "top"]].set_visible(False)
    panel_label(axis, "A")

    # B: regret profiles at t=0.74.
    axis = axes[1]
    block = raw[raw["bridge_time_index"] == representative_time]
    for domain, label, color in zip(DOMAINS, LABELS, COLORS, strict=True):
        profile = (
            block[block["domain"] == domain]
            .groupby("single_epoch")["reciprocal_KDD"]
            .mean()
            .sort_index()
        )
        regret = profile - profile.min()
        axis.plot(profile.index, regret, marker="o", ms=4.2, lw=1.8, color=color, label=label)
    axis.axvline(common, color="#202124", ls="--", lw=1.0)
    axis.set_xticks(range(1, 7))
    axis.set_xlabel("task-specific epoch")
    axis.set_ylabel("within-domain KDD regret")
    axis.set_title("No common phase minimizes all six kernel profiles (t=0.74)")
    axis.grid(axis="y", color="#E6E8EA", lw=0.7)
    axis.legend(ncol=2, frameon=False, loc="upper center")
    panel_label(axis, "B")

    # C: domain x bridge-time phase field.
    axis = axes[2]
    time_indices = sorted(cells["bridge_time_index"].unique())
    matrix = np.empty((len(DOMAINS), len(time_indices)))
    shares = np.empty_like(matrix)
    for row, domain in enumerate(DOMAINS):
        for column, time_index in enumerate(time_indices):
            cell = cells[(cells["domain"] == domain) & (cells["bridge_time_index"] == time_index)].iloc[0]
            matrix[row, column] = cell["effective_age"]
            shares[row, column] = cell["bootstrap_modal_share"]
    cmap = ListedColormap(AGE_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 7.5, 1), cmap.N)
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"e{int(matrix[row, column])}\n{shares[row, column] * 100:.0f}%",
                ha="center",
                va="center",
                color="white" if matrix[row, column] >= 4 else "#202124",
                fontsize=7.5,
            )
    time_values = [
        next(item["bridge_time_value"] for item in stats["time_results"] if item["bridge_time_index"] == index)
        for index in time_indices
    ]
    axis.set_xticks(range(len(time_indices)), [f"t={value:.2f}" for value in time_values])
    axis.set_yticks(range(len(DOMAINS)), LABELS)
    axis.set_title("Six-domain bridge-time phase field (mode share)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.025, ticks=range(1, 7))
    colorbar.set_label("effective phase")
    panel_label(axis, "C")

    # D: primary cross-fitted shared-clock regret.
    axis = axes[3]
    values = np.array([item["crossfit_sync_regret"] for item in stats["time_results"]])
    lows = np.array([item["bootstrap_ci_95"][0] for item in stats["time_results"]])
    highs = np.array([item["bootstrap_ci_95"][1] for item in stats["time_results"]])
    times = np.array([item["bridge_time_value"] for item in stats["time_results"]])
    axis.errorbar(
        times,
        values,
        yerr=np.vstack([values - lows, highs - values]),
        color="#D14A32",
        marker="o",
        ms=6,
        lw=2,
        capsize=4,
    )
    axis.axhline(0, color="#202124", lw=0.9)
    for x, y, item in zip(times, values, stats["time_results"], strict=True):
        axis.annotate(
            f"{item['crossfit_normalized_regret'] * 100:.1f}% of profile range",
            (x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
        )
    axis.set_xticks(times, [f"{value:.2f}" for value in times])
    axis.set_xlabel("bridge time")
    axis.set_ylabel("held-out reciprocal KDD regret")
    axis.set_title("A shared clock pays positive held-out kernel regret")
    axis.grid(axis="y", color="#E6E8EA", lw=0.7)
    panel_label(axis, "D")

    # E: full bootstrap phase distributions, with no soft-min temperature.
    axis = axes[4]
    phase_block = phase_draws[phase_draws["bridge_time_index"] == representative_time]
    probability = np.zeros((len(DOMAINS), 6), dtype=np.float64)
    for row, domain in enumerate(DOMAINS):
        counts = phase_block.loc[phase_block["domain"] == domain, "effective_age"].value_counts()
        for age in range(1, 7):
            probability[row, age - 1] = counts.get(age, 0) / max(counts.sum(), 1)
    probability_image = axis.imshow(probability, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(probability.shape[0]):
        for column in range(probability.shape[1]):
            value = probability[row, column]
            if value >= 0.01:
                axis.text(
                    column,
                    row,
                    f"{value * 100:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#202124",
                    fontsize=7.5,
                )
    axis.set_xticks(range(6), [f"e{age}" for age in range(1, 7)])
    axis.set_yticks(range(len(DOMAINS)), LABELS)
    axis.set_xlabel("bootstrap effective phase")
    axis.set_title(
        "Phase distributions remain separated "
        f"(W₂ energy={representative['wasserstein_barycenter_energy']:.2f})"
    )
    colorbar = figure.colorbar(probability_image, ax=axis, fraction=0.04, pad=0.025)
    colorbar.set_label("bootstrap probability")
    panel_label(axis, "E")

    # F: domain contributions to the representative shared-clock regret.
    axis = axes[5]
    representative_cells = cells[cells["bridge_time_index"] == representative_time].set_index("domain")
    contributions = np.array(
        [representative_cells.loc[domain, "crossfit_domain_regret"] for domain in DOMAINS]
    )
    positions = np.arange(len(DOMAINS))
    axis.barh(positions, contributions, color=COLORS, alpha=0.88)
    axis.axvline(0, color="#202124", lw=0.9)
    axis.axvline(values[1], color="#D14A32", ls="--", lw=1.2)
    for y, value in zip(positions, contributions, strict=True):
        axis.text(
            value + (0.002 if value >= 0 else -0.002),
            y,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7.5,
        )
    axis.set_yticks(positions, LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("held-out KDD regret at t=0.74")
    axis.set_title("Domain contributions to the shared-clock cost")
    axis.grid(axis="x", color="#E6E8EA", lw=0.7)
    panel_label(axis, "F")

    verdict = stats["verdict"]
    if verdict == "SUPPORTED_SIXDOMAIN_SHARED_CLOCK_REGRET":
        title = "A shared UNSB clock incurs measurable domain-phase regret"
    elif verdict == "PARTIAL_SIXDOMAIN_PHASE_STRUCTURE":
        title = "Six-domain UNSB shows partial domain-phase structure"
    else:
        title = "Six-domain replication does not support shared-clock phase regret"
    figure.suptitle(title, fontsize=18, fontweight="bold", y=0.965)
    figure.text(
        0.5,
        0.91,
        "Fresh plain UNSB · six domains · 120 train + 80 held-out images/domain · seed 2051 · "
        "cross-fitted clocks · no paired target · no candidate method",
        ha="center",
        va="center",
        fontsize=10,
        color="#4A4D50",
    )
    figure.text(
        0.5,
        0.025,
        "KDD measures reciprocal angular mismatch between conditional mean endpoint directions. "
        "The figure supports a process-level phase incompatibility, not causal restoration harm or method effectiveness.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#55585B",
    )
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"UNSB_SIXDOMAIN_PHASE_HEADFIGURE.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
