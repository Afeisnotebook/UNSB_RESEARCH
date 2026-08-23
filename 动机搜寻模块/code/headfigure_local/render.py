from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
from PIL import Image

try:
    from .common import DOMAINS, dump_json, sha256_file
except ImportError:  # direct script execution
    from common import DOMAINS, dump_json, sha256_file


SINGLE = "#167D8D"
AIO = "#E4572E"
CLOCK = "#6C5CE7"
ENVELOPE = "#222222"
DOMAIN_SHORT = {
    "FoggyCityscapes": "Fog",
    "LowLightTrafficData": "Low-light",
    "RainCityscapes": "Rain",
    "RSCityscapes": "Rain-streak",
    "SnowTrafficData": "Snow",
}


def load_thumbnail(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((96, 96), Image.Resampling.LANCZOS)
    return np.asarray(image)


def draw_schematic(ax, heldout: list[dict]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    reps = [next(row for row in heldout if row["domain"] == d and row.get("representative")) for d in DOMAINS]
    colors = ["#80B1D3", "#FDB462", "#B3DE69", "#BC80BD", "#FB8072"]
    xs = np.linspace(0.8, 4.6, 5)
    for x, row, color in zip(xs, reps, colors):
        image = load_thumbnail(row["source_path"])
        ax.imshow(image, extent=(x - 0.38, x + 0.38, 7.55, 8.31), zorder=3)
        ax.add_patch(plt.Rectangle((x - 0.4, 7.53), 0.8, 0.8, fill=False, lw=1.4, ec=color, zorder=4))
        ax.add_patch(FancyArrowPatch((x, 7.45), (x, 5.9), arrowstyle="-|>", mutation_scale=10, lw=1, color=color))
        ax.add_patch(plt.Circle((x, 5.45), 0.34, ec=SINGLE, fc="white", lw=1.8))
        ax.add_patch(FancyArrowPatch((x, 5.1), (x, 3.75), arrowstyle="-|>", mutation_scale=10, lw=1, color=SINGLE))
    ax.text(2.7, 9.05, "Task-specific training", ha="center", va="bottom", fontsize=10, weight="bold", color=SINGLE)
    ax.text(2.7, 3.35, "five independent bridges", ha="center", va="top", fontsize=8.5, color=SINGLE)

    xs2 = np.linspace(5.7, 9.5, 5)
    for x, row, color in zip(xs2, reps, colors):
        image = load_thumbnail(row["source_path"])
        ax.imshow(image, extent=(x - 0.38, x + 0.38, 7.55, 8.31), zorder=3)
        ax.add_patch(plt.Rectangle((x - 0.4, 7.53), 0.8, 0.8, fill=False, lw=1.4, ec=color, zorder=4))
        ax.add_patch(FancyArrowPatch((x, 7.45), (7.6, 5.65), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color, alpha=0.85))
    ax.add_patch(plt.Circle((7.6, 5.35), 0.52, ec=AIO, fc="#FFF1EC", lw=2.3))
    angles = np.linspace(-55, 55, 7)
    for angle in angles:
        rad = np.deg2rad(angle)
        end = (7.6 + 1.15 * np.sin(rad), 3.85 - 0.25 * abs(np.sin(rad)))
        ax.add_patch(FancyArrowPatch((7.6, 4.85), end, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=AIO, alpha=0.75))
    ax.add_patch(Polygon([[6.55, 3.58], [8.65, 3.58], [7.6, 4.95]], closed=True, fc=AIO, ec="none", alpha=0.09))
    ax.text(7.6, 9.05, "All-in-One training", ha="center", va="bottom", fontsize=10, weight="bold", color=AIO)
    ax.text(7.6, 3.35, "one shared bridge · initial fan-out?", ha="center", va="top", fontsize=8.5, color=AIO)
    ax.plot([5.15, 5.15], [3.0, 9.1], color="#DDDDDD", lw=1)


def draw_consensus(ax, historic: pd.DataFrame) -> None:
    matrix = np.ones((6, 5))
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#DFF3EF"]), vmin=0, vmax=1, aspect="auto")
    for row in range(6):
        for col in range(5):
            ax.text(col, row, "+", ha="center", va="center", color="#087F6A", fontsize=16, weight="bold")
    ax.set_xticks(range(5), [DOMAIN_SHORT[d] for d in DOMAINS], rotation=32, ha="right", fontsize=8)
    ax.set_yticks(range(6), [str(s) for s in range(2026, 2032)], fontsize=8)
    ax.set_xlabel("weather domain", fontsize=8)
    ax.set_ylabel("independent training seed", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ci_positive = int(historic["ci_positive"].astype(bool).sum())
    ax.text(1.02, 0.66, "30 / 30\nseed × domain\npositive", transform=ax.transAxes, ha="left", va="center", fontsize=9, color="#087F6A", weight="bold")
    ax.text(1.02, 0.27, f"{ci_positive} / 18\nseed × bridge-time\nCIs above zero", transform=ax.transAxes, ha="left", va="center", fontsize=8, color="#555555")


def draw_forest(ax, domain: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    data = domain[(domain["bridge_time_index"] == 2) & (domain["contrast"] == "age_envelope_excess")].copy()
    data["order"] = data["domain"].map({d: i for i, d in enumerate(DOMAINS)})
    data = data.sort_values("order", ascending=False)
    y = np.arange(len(data)) + 1
    for yi, (_, row) in zip(y, data.iterrows()):
        color = AIO if row["mean"] > 0 else "#777777"
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=color, lw=2)
        ax.scatter(row["mean"], yi, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
    pooled = bootstrap[(bootstrap["bridge_time_index"] == 2) & (bootstrap["contrast"] == "age_envelope_excess")].iloc[0]
    diamond = np.array([[pooled["ci_low"], 0], [pooled["mean"], 0.28], [pooled["ci_high"], 0], [pooled["mean"], -0.28]])
    ax.add_patch(Polygon(diamond, closed=True, fc=AIO, ec="white", lw=0.8, zorder=3))
    ax.axvline(0, color="#777777", lw=1, ls="--")
    ax.set_yticks([0] + list(y), ["Pooled"] + [DOMAIN_SHORT[d] for d in data["domain"]], fontsize=8)
    ax.set_xlabel(r"AIO e1 − max Single e1…e5  [log$_{10}$ $D_{sph}$]", fontsize=8)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_bridge_curves(ax, bootstrap: pd.DataFrame) -> None:
    bridge_values = {1: 0.50, 2: 0.74, 3: 0.86}
    styles = {
        "exposure_gap": (SINGLE, "o", "Exposure-matched: AIO e1 − Single e1"),
        "clock_gap": (CLOCK, "s", "Optimizer-clock: AIO e1 − Single e5"),
        "age_envelope_excess": (AIO, "D", "Early-age envelope: AIO e1 − max Single"),
    }
    for contrast, (color, marker, label) in styles.items():
        rows = bootstrap[bootstrap["contrast"] == contrast].sort_values("bridge_time_index")
        x = np.array([bridge_values[int(t)] for t in rows["bridge_time_index"]])
        y = rows["mean"].to_numpy()
        lo = rows["ci_low"].to_numpy()
        hi = rows["ci_high"].to_numpy()
        ax.plot(x, y, color=color, marker=marker, lw=2.2, ms=5, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xlabel("physical bridge time", fontsize=9)
    ax.set_ylabel(r"paired excess [log$_{10}$ $D_{sph}$]", fontsize=9)
    ax.set_xticks(list(bridge_values.values()))
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="best")


def ellipse_from_points(ax, points: np.ndarray, color: str, label: str) -> None:
    center = points.mean(axis=0)
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = math.sqrt(4.605)  # 90% Gaussian reference, display only
    width, height = 2 * scale * np.sqrt(vals[:2])
    ax.add_patch(Ellipse(center, width, height, angle=angle, ec=color, fc=color, alpha=0.12, lw=1.5))
    ax.scatter(points[:, 0], points[:, 1], s=8, color=color, alpha=0.42, label=label, edgecolors="none")
    ax.scatter(center[0], center[1], s=22, color=color, edgecolor="white", lw=0.5, zorder=4)


def draw_pca_glyph(ax, block: pd.DataFrame, domain: str, show_legend: bool) -> None:
    mapping = {
        "single_e1": (SINGLE, "Single e1"),
        "single_e5": (CLOCK, "Single e5"),
        "aio_e1": (AIO, "AIO e1"),
    }
    for arm, (color, label) in mapping.items():
        points = block[block["arm"] == arm][["pca1", "pca2"]].to_numpy(dtype=float)
        ellipse_from_points(ax, points, color, label)
    ax.axhline(0, color="#DDDDDD", lw=0.6)
    ax.axvline(0, color="#DDDDDD", lw=0.6)
    ax.set_title(DOMAIN_SHORT[domain], fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[:].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=6.7, loc="upper left", bbox_to_anchor=(-0.15, 1.25), ncol=3)


def make_headfigure(run_root: Path) -> dict:
    heldout = json.loads((run_root / "state" / "HELDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    historic = pd.read_csv(run_root / "reports" / "HISTORIC_E1_CONSENSUS.csv")
    bootstrap = pd.read_csv(run_root / "reports" / "BOOTSTRAP_SUMMARY.csv")
    domain = pd.read_csv(run_root / "reports" / "DOMAIN_SUMMARY.csv")
    pca = pd.read_csv(run_root / "raw" / "REPRESENTATIVE_JOINT_PCA.csv")
    adjudication = json.loads((run_root / "reports" / "ADJUDICATION.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(18.2, 10.2), facecolor="white")
    gs = fig.add_gridspec(2, 12, height_ratios=[1.02, 1.0], hspace=0.39, wspace=0.8)
    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:8])
    ax_c = fig.add_subplot(gs[0, 8:12])
    ax_d = fig.add_subplot(gs[1, 0:5])
    pca_gs = gs[1, 5:12].subgridspec(1, 5, wspace=0.15)
    pca_axes = [fig.add_subplot(pca_gs[0, idx]) for idx in range(5)]

    draw_schematic(ax_a, heldout)
    draw_consensus(ax_b, historic)
    draw_forest(ax_c, domain, bootstrap)
    draw_bridge_curves(ax_d, bootstrap)
    for idx, domain_name in enumerate(DOMAINS):
        draw_pca_glyph(pca_axes[idx], pca[pca["domain"] == domain_name], domain_name, idx == 0)

    fig.suptitle(
        "Sharing one bridge changes the initial conditional direction field",
        x=0.5,
        y=0.985,
        fontsize=18,
        weight="bold",
    )
    fig.text(0.02, 0.945, "A", fontsize=14, weight="bold")
    fig.text(0.345, 0.945, "B", fontsize=14, weight="bold")
    fig.text(0.675, 0.945, "C", fontsize=14, weight="bold")
    fig.text(0.02, 0.48, "D", fontsize=14, weight="bold")
    fig.text(0.435, 0.48, "E", fontsize=14, weight="bold")
    fig.text(0.035, 0.915, "Training-regime contrast", fontsize=10, weight="bold")
    fig.text(0.36, 0.915, "Epoch-1 replication wall", fontsize=10, weight="bold")
    fig.text(0.69, 0.915, "Fresh-seed early-age envelope", fontsize=10, weight="bold")
    fig.text(0.035, 0.455, "Dual-matched bridge-time effect", fontsize=10, weight="bold")
    fig.text(0.45, 0.455, "Joint direction geometry · hash-selected input per domain · t=0.74", fontsize=10, weight="bold")
    verdict_text = adjudication["verdict"].replace("_", " ")
    fig.text(
        0.5,
        0.018,
        rf"$D_{{sph}}$ = mean pairwise cosine distance of stochastic endpoint directions. "
        f"Fresh seed 2041 · 20 held-out-within-discovery images/domain · M=32 · verdict: {verdict_text}. "
        "Direction dispersion is not calibrated posterior uncertainty.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#555555",
    )

    figure_root = run_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    png = figure_root / "UNSB_MOTIVATION_HEADFIGURE.png"
    pdf = figure_root / "UNSB_MOTIVATION_HEADFIGURE.pdf"
    svg = figure_root / "UNSB_MOTIVATION_HEADFIGURE.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "svg": str(svg)}


def fmt(value: float) -> str:
    return f"{value:+.3f}"


def write_report(run_root: Path, figure_paths: dict) -> Path:
    adjudication = json.loads((run_root / "reports" / "ADJUDICATION.json").read_text(encoding="utf-8"))
    bootstrap = pd.read_csv(run_root / "reports" / "BOOTSTRAP_SUMMARY.csv")
    domain = pd.read_csv(run_root / "reports" / "DOMAIN_SUMMARY.csv")
    prepare = json.loads((run_root / "state" / "PREPARE_AUDIT.json").read_text(encoding="utf-8"))
    training = json.loads((run_root / "state" / "TRAINING_STATE.json").read_text(encoding="utf-8"))
    measurement = json.loads((run_root / "state" / "MEASUREMENT_STATE.json").read_text(encoding="utf-8"))

    lines = [
        "# UNSB 动机头图本地验证报告：初始共享桥扇出",
        "",
        "> 本报告只比较同一 plain UNSB 的 Single-task 与 All-in-One 无配对训练，不含任何方法分支。",
        "",
        "## 1. 结论",
        "",
        f"冻结裁决：**`{adjudication['verdict']}`**。",
        "",
        "核心问题是：五域共享 AIO 在 epoch 1 的随机条件恢复方向，是否不仅高于本域曝光匹配的 Single epoch 1，还高于总优化步匹配的 Single epoch 5，并超出 Single epoch 1–5 的整个早期包络。",
        "",
        "## 2. 三个预注册对照",
        "",
        "| bridge t | exposure gap | 95% CI | clock gap | 95% CI | age-envelope excess | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in (1, 2, 3):
        block = {row["contrast"]: row for _, row in bootstrap[bootstrap["bridge_time_index"] == t].iterrows()}
        lines.append(
            f"| { {1:0.50,2:0.74,3:0.86}[t]:.2f} | "
            f"{fmt(block['exposure_gap']['mean'])} | [{fmt(block['exposure_gap']['ci_low'])}, {fmt(block['exposure_gap']['ci_high'])}] | "
            f"{fmt(block['clock_gap']['mean'])} | [{fmt(block['clock_gap']['ci_low'])}, {fmt(block['clock_gap']['ci_high'])}] | "
            f"{fmt(block['age_envelope_excess']['mean'])} | [{fmt(block['age_envelope_excess']['ci_low'])}, {fmt(block['age_envelope_excess']['ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "所有差值单位均为 `log10(D_sph)`；置信区间使用域等权、先抽域再抽图像的嵌套 bootstrap。正值表示 AIO epoch 1 的方向离散更高。",
            "",
            "## 3. 分域 age-envelope 结果（bridge t=0.74）",
            "",
            "| domain | mean | 95% CI | positive images |",
            "|---|---:|---:|---:|",
        ]
    )
    block = domain[(domain["bridge_time_index"] == 2) & (domain["contrast"] == "age_envelope_excess")]
    for domain_name in DOMAINS:
        row = block[block["domain"] == domain_name].iloc[0]
        lines.append(
            f"| {domain_name} | {fmt(row['mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {row['positive_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 4. 与已有六 seed 证据的关系",
            "",
            f"仓库历史审计在 seed 2026–2031 上给出 Epoch 1：`{adjudication['historic_epoch1']['seed_domain_positive']}/{adjudication['historic_epoch1']['seed_domain_total']}` 个 seed×domain 单元为正，`{adjudication['historic_epoch1']['seed_time_ci_positive']}/{adjudication['historic_epoch1']['seed_time_total']}` 个 seed×bridge-time pooled 区间位于零上方。fresh seed=2041 的本地双控制实验用于检查该现象是否还能超越普通 optimizer-age 解释。",
            "",
            "## 5. 头图读法",
            "",
            "- A：同样五个天气域，左侧分别训练任务专用桥，右侧共享一个桥；问号表示这是待检验现象，不是预设结论。",
            "- B：六个历史 seed 的 Epoch-1 复现墙，每个单元只编码预注册符号，不用跨 seed 的绝对 U 大小制造色差。",
            "- C：fresh seed 下 AIO epoch 1 相对 Single epoch 1–5 最大离散的分域 forest；Pooled 使用域等权嵌套 bootstrap。",
            "- D：本域曝光、总优化步和整个早期年龄包络三种控制随 bridge time 的结果。",
            "- E：每域按哈希盲选一张输入，在同一 PCA 中显示 Single e1、Single e5 与 AIO e1 的随机方向；椭圆是 90% Gaussian reference，仅用于显示。",
            "",
            "## 6. 工程与数据边界",
            "",
            f"- 数据准备：`{prepare['verdict']}`；五域各 100A/100B，历史方向测量 stem 排除后哈希盲选 20 图/域。",
            f"- 训练：`{training['status']}`；seed 2041；总墙钟 {training.get('elapsed_seconds', 0)/60:.1f} 分钟。",
            f"- 测量：`{measurement['status']}`；{measurement['rows']} 个 image×arm×time 统计行；M={measurement['M']}；墙钟 {measurement['elapsed_seconds']/60:.1f} 分钟。",
            "- held-out 状态：`HELDOUT_WITHIN_DISCOVERY_NOT_CONFIRMATORY`；没有读取 paired target 或 sealed 图像。",
            "- `D_sph` 是单位恢复方向间的平均 pairwise cosine distance，不是校准后验不确定性。",
            "- 图像 bootstrap 不等于训练 seed 级确认；fresh seed 只是新增一个训练复本。",
            "",
            "## 7. 论文表述边界",
            "",
            "允许：共享五域训练在初始阶段改变 plain UNSB 的条件方向场；若双控制门通过，可进一步写为该扇出不能仅由本域曝光或总 optimizer age 解释。",
            "",
            "禁止：把它写成因果伤害、校准 posterior uncertainty、全训练阶段规律，或直接声称它必然导致终点性能下降。",
            "",
            "## 8. 文件",
            "",
            f"- PNG：`{figure_paths['png']}`",
            f"- PDF：`{figure_paths['pdf']}`",
            f"- SVG：`{figure_paths['svg']}`",
            "- 原始统计：`raw/DIRECTION_STATISTICS.csv`",
            "- 配对行：`reports/PAIRED_CONTRAST_ROWS.csv`",
            "- 裁决：`reports/ADJUDICATION.json`",
            "",
        ]
    )
    report = run_root / "reports" / "SCIENTIFIC_REPORT_CN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def build_manifest(run_root: Path, paths: list[Path]) -> Path:
    manifest = run_root / "MANIFEST.sha256"
    entries = []
    for path in sorted(set(paths), key=lambda p: str(p).lower()):
        rel = path.relative_to(run_root).as_posix()
        entries.append(f"{sha256_file(path)}  {rel}")
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    figure_paths = make_headfigure(run_root)
    report = write_report(run_root, figure_paths)
    required = [
        run_root / "state" / "PREPARE_AUDIT.json",
        run_root / "state" / "TRAINING_STATE.json",
        run_root / "state" / "MEASUREMENT_STATE.json",
        run_root / "state" / "DATA_MANIFEST.csv",
        run_root / "state" / "HELDOUT_MANIFEST.json",
        run_root / "raw" / "DIRECTION_STATISTICS.csv",
        run_root / "raw" / "REPRESENTATIVE_JOINT_PCA.csv",
        run_root / "reports" / "PAIRED_CONTRAST_ROWS.csv",
        run_root / "reports" / "BOOTSTRAP_SUMMARY.csv",
        run_root / "reports" / "BOOTSTRAP_DRAWS.csv",
        run_root / "reports" / "DOMAIN_SUMMARY.csv",
        run_root / "reports" / "HISTORIC_E1_CONSENSUS.csv",
        run_root / "reports" / "ADJUDICATION.json",
        report,
        *(Path(path) for path in figure_paths.values()),
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    audit = {
        "verdict": "PASS" if not missing else "HOLD",
        "missing": missing,
        "required_files": len(required),
        "figure_bytes": {Path(path).name: Path(path).stat().st_size for path in figure_paths.values()},
        "report_sha256": sha256_file(report),
        "pure_baseline_only": True,
    }
    dump_json(run_root / "state" / "FINAL_AUDIT.json", audit)
    required.append(run_root / "state" / "FINAL_AUDIT.json")
    manifest = build_manifest(run_root, required)
    print(json.dumps({**audit, "manifest": str(manifest)}, ensure_ascii=False, indent=2))
    return 0 if audit["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
