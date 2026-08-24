from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from PIL import Image

try:
    from .common import DOMAINS, dump_json, sha256_file
except ImportError:  # direct execution
    from common import DOMAINS, dump_json, sha256_file


SINGLE, AIO, PURPLE, GREEN, GRAY = "#177E89", "#D95F02", "#6C5CE7", "#17866B", "#666666"
DOMAIN_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#E15759"]
DOMAIN_SHORT = {
    "FoggyCityscapes": "Fog",
    "LowLightTrafficData": "Low-light",
    "RainCityscapes": "Rain",
    "RSCityscapes": "Rain-streak",
    "SnowTrafficData": "Snow",
}
BRIDGE = {1: 0.50, 2: 0.74, 3: 0.86}


def thumbnail(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return np.asarray(
        image.crop((left, top, left + side, top + side)).resize(
            (96, 96), Image.Resampling.LANCZOS
        )
    )


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, linewidth=0.7)


def draw_regime(ax, heldout: list[dict]) -> None:
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    reps = [
        next(row for row in heldout if row["domain"] == domain and row.get("representative"))
        for domain in DOMAINS
    ]
    for x, row, color in zip(np.linspace(0.8, 5.0, 5), reps, DOMAIN_COLORS):
        ax.imshow(thumbnail(row["source_path"]), extent=(x - 0.38, x + 0.38, 8.0, 8.76))
        ax.add_patch(plt.Rectangle((x - 0.4, 7.98), 0.8, 0.8, fill=False, ec=color, lw=1.4))
        ax.add_patch(FancyArrowPatch((x, 7.8), (x, 6.35), arrowstyle="-|>", mutation_scale=9, color=color, lw=1))
        ax.add_patch(plt.Circle((x, 5.9), 0.34, ec=SINGLE, fc="white", lw=1.8))
        ax.add_patch(FancyArrowPatch((x, 5.55), (x, 3.65), arrowstyle="-|>", mutation_scale=9, color=SINGLE, lw=1.1))
    ax.text(2.9, 9.2, "Task-specific", ha="center", fontsize=10, color=SINGLE, weight="bold")
    ax.text(2.9, 3.1, "five independent bridge clocks", ha="center", fontsize=8, color=SINGLE)

    for x, row, color in zip(np.linspace(7.0, 11.2, 5), reps, DOMAIN_COLORS):
        ax.imshow(thumbnail(row["source_path"]), extent=(x - 0.38, x + 0.38, 8.0, 8.76))
        ax.add_patch(plt.Rectangle((x - 0.4, 7.98), 0.8, 0.8, fill=False, ec=color, lw=1.4))
        ax.add_patch(FancyArrowPatch((x, 7.75), (9.1, 6.2), arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9))
    ax.add_patch(plt.Circle((9.1, 5.8), 0.48, ec=AIO, fc="#FFF0E7", lw=2.2))
    for index, (age, color) in enumerate(zip([4, 3, 2, 4, 5], DOMAIN_COLORS)):
        end = (7.65 + 0.72 * index, 3.45 + 0.18 * (age - 2))
        ax.add_patch(FancyArrowPatch((9.1, 5.3), end, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.15))
        ax.text(end[0], end[1] - 0.24, f"e{age}", ha="center", fontsize=6.8, color=color, weight="bold")
    ax.text(9.1, 9.2, "All-in-One", ha="center", fontsize=10, color=AIO, weight="bold")
    ax.text(9.1, 3.0, "one checkpoint · multiple task phases", ha="center", fontsize=8, color=AIO)
    ax.plot([6.0, 6.0], [2.7, 9.4], color="#DDDDDD", lw=1)
    ax.text(6.0, 1.4, "Does a shared checkpoint synchronize task-specific bridge progress?", ha="center", fontsize=9.2, weight="bold")


def draw_sign_wall(ax) -> None:
    ax.imshow(np.ones((6, 5)), cmap=matplotlib.colors.ListedColormap(["#DDF2EC"]), aspect="auto")
    for row in range(6):
        for col in range(5):
            ax.text(col, row, "+", ha="center", va="center", color=GREEN, fontsize=15, weight="bold")
    ax.set_xticks(range(5), [DOMAIN_SHORT[d] for d in DOMAINS], rotation=31, ha="right", fontsize=7.1)
    ax.set_yticks(range(6), [str(seed) for seed in range(2026, 2032)], fontsize=7.1)
    ax.set_xlabel("weather domain", fontsize=8)
    ax.set_ylabel("training seed", fontsize=8)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1.04, 0.70, "30/30\npositive", transform=ax.transAxes, color=GREEN, fontsize=9.3, weight="bold", va="center")
    ax.text(1.04, 0.31, "18/18 pooled\ntime CIs > 0", transform=ax.transAxes, color=GRAY, fontsize=7.7, va="center")


def draw_confound(ax, stage1: pd.DataFrame) -> None:
    styles = {
        "exposure_gap": (SINGLE, "exposure: AIO e1 − Single e1"),
        "clock_gap": (PURPLE, "clock: AIO e1 − Single e5"),
        "age_envelope_excess": (AIO, "age envelope: AIO e1 − max Single"),
    }
    offsets = {"exposure_gap": -0.025, "clock_gap": 0.0, "age_envelope_excess": 0.025}
    for contrast, (color, label) in styles.items():
        rows = stage1[stage1["contrast"] == contrast].sort_values("bridge_time_index")
        x = np.array([BRIDGE[int(value)] for value in rows["bridge_time_index"]]) + offsets[contrast]
        y = rows["mean"].to_numpy(float)
        lo, hi = rows["ci_low"].to_numpy(float), rows["ci_high"].to_numpy(float)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", ms=4, capsize=2.3, color=color, lw=1.7, label=label)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks(list(BRIDGE.values()))
    ax.set_xlabel("physical bridge time", fontsize=8)
    ax.set_ylabel(r"excess [log$_{10}$ $D_{sph}$]", fontsize=8)
    ax.legend(frameon=False, fontsize=6.7, loc="upper left")
    clean(ax)
    ax.text(0.98, 0.05, "EXPOSURE_ONLY\nnot the final motive", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6, color=GRAY, weight="bold")


def draw_kdd_profiles(ax, tertiary_raw: pd.DataFrame) -> None:
    block = tertiary_raw[tertiary_raw["bridge_time_index"] == 2]
    for domain, color in zip(DOMAINS, DOMAIN_COLORS):
        means = block[block["domain"] == domain].groupby("single_epoch")["reciprocal_KDD"].mean().sort_index()
        relative = means / means.min() - 1.0
        best = int(means.idxmin())
        ax.plot(means.index, relative, marker="o", color=color, lw=2, ms=4.5, label=f"{DOMAIN_SHORT[domain]} → e{best}")
        ax.scatter([best], [0], color=color, marker="v", s=34, zorder=4)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("task-specific checkpoint age", fontsize=8)
    ax.set_ylabel("relative reciprocal KDD above domain minimum", fontsize=8)
    ax.legend(frameon=False, fontsize=6.7, ncol=2, loc="upper right")
    clean(ax)


def age_map_from_raw(frame: pd.DataFrame) -> np.ndarray:
    result = np.empty((5, 3), dtype=int)
    for domain_index, domain in enumerate(DOMAINS):
        for time_index, t in enumerate((1, 2, 3)):
            block = frame[(frame["domain"] == domain) & (frame["bridge_time_index"] == t)]
            result[domain_index, time_index] = int(block.groupby("single_epoch")["reciprocal_KDD"].mean().idxmin())
    return result


def cell_map(frame: pd.DataFrame, age_column: str) -> tuple[np.ndarray, np.ndarray]:
    ages = np.empty((5, 3), dtype=int)
    shares = np.empty((5, 3), dtype=float)
    for domain_index, domain in enumerate(DOMAINS):
        for time_index, t in enumerate((1, 2, 3)):
            row = frame[(frame["domain"] == domain) & (frame["bridge_time_index"] == t)].iloc[0]
            ages[domain_index, time_index] = int(row[age_column])
            shares[domain_index, time_index] = float(row["bootstrap_modal_share"])
    return ages, shares


def draw_three_split_map(ax, discovery_raw: pd.DataFrame, first_cell: pd.DataFrame, tertiary_cell: pd.DataFrame) -> None:
    discovery = age_map_from_raw(discovery_raw)
    first, first_share = cell_map(first_cell, "bootstrap_modal_age")
    tertiary, tertiary_share = cell_map(tertiary_cell, "bootstrap_modal_age")
    combined = np.concatenate([discovery, first, tertiary], axis=1)
    ax.imshow(combined, cmap="viridis", vmin=1, vmax=5, aspect="auto")
    for row in range(5):
        for col in range(9):
            text = f"e{combined[row, col]}"
            if 3 <= col < 6:
                text += f"\n{first_share[row, col-3]:.0%}"
            elif col >= 6:
                text += f"\n{tertiary_share[row, col-6]:.0%}"
            color = "white" if combined[row, col] >= 3 else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=6.4, weight="bold")
    ax.axvline(2.5, color="white", lw=3)
    ax.axvline(5.5, color="white", lw=3)
    ax.set_xticks(range(9), [".50", ".74", ".86"] * 3, fontsize=6.8)
    ax.set_yticks(range(5), [DOMAIN_SHORT[d] for d in DOMAINS], fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel("discovery (20/d)        first holdout (24/d)        tertiary (16/d)", fontsize=7.5)
    ax.set_ylabel("weather domain", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 1.03, "bridge time · held-out cells also show bootstrap modal share", transform=ax.transAxes, ha="center", fontsize=7.1, color=GRAY)


def draw_mapping_null(ax, adjudication: dict, draws: pd.DataFrame) -> None:
    values = draws["mapping_accuracy"].to_numpy(int)
    bins = np.arange(-0.5, 16.5, 1)
    ax.hist(values, bins=bins, density=True, color="#9BBBD3", alpha=0.65, edgecolor="white")
    observed = adjudication["mapping_accuracy"]
    ax.axvline(observed, color=AIO, lw=2.8)
    ax.scatter([observed], [0], marker="^", color=AIO, s=55, zorder=4)
    ax.text(observed - 0.15, ax.get_ylim()[1] * 0.93, f"observed {observed}/15\np={adjudication['permutation_p']:.4f}", ha="right", va="top", color=AIO, fontsize=8.5, weight="bold")
    ax.set_xlim(-0.5, 15.7)
    ax.set_xticks(range(0, 16, 2))
    ax.set_xlabel("fixed domain→age mapping accuracy", fontsize=8)
    ax.set_ylabel("within-time mapping-permutation density", fontsize=8)
    clean(ax)
    ax.text(0.02, 0.95, "profiles never move between\ndomain-specific model systems", transform=ax.transAxes, va="top", fontsize=7.1, color=GRAY)


def make_figure(discovery_root: Path, first_root: Path, tertiary_root: Path) -> dict[str, str]:
    heldout = json.loads((tertiary_root / "state" / "HELDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    stage1 = pd.read_csv(discovery_root / "reports" / "BOOTSTRAP_SUMMARY.csv")
    discovery_raw = pd.read_csv(discovery_root / "raw" / "RECIPROCAL_KERNEL_BY_AGE.csv")
    first_cell = pd.read_csv(first_root / "reports" / "PHASE_CELL_SUMMARY.csv")
    tertiary_raw = pd.read_csv(tertiary_root / "raw" / "RECIPROCAL_KERNEL_BY_AGE.csv")
    tertiary_cell = pd.read_csv(tertiary_root / "reports" / "TERTIARY_PHASE_CELL_SUMMARY.csv")
    draws = pd.read_csv(tertiary_root / "reports" / "TERTIARY_MAPPING_PERMUTATION_DRAWS.csv")
    adjudication = json.loads((tertiary_root / "reports" / "TERTIARY_PHASE_ADJUDICATION.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(18.8, 10.8), facecolor="white")
    gs = fig.add_gridspec(2, 12, hspace=0.43, wspace=1.25)
    axes = [
        fig.add_subplot(gs[0, :5]),
        fig.add_subplot(gs[0, 5:8]),
        fig.add_subplot(gs[0, 8:12]),
        fig.add_subplot(gs[1, :4]),
        fig.add_subplot(gs[1, 4:8]),
        fig.add_subplot(gs[1, 8:12]),
    ]
    draw_regime(axes[0], heldout)
    draw_sign_wall(axes[1])
    draw_confound(axes[2], stage1)
    draw_kdd_profiles(axes[3], tertiary_raw)
    draw_three_split_map(axes[4], discovery_raw, first_cell, tertiary_cell)
    draw_mapping_null(axes[5], adjudication, draws)

    fig.suptitle(
        "One shared bridge checkpoint occupies desynchronized task-specific transition phases",
        fontsize=17.5,
        weight="bold",
        y=0.982,
    )
    labels = [
        (0.018, 0.945, "A", "Framework-level question"),
        (0.425, 0.945, "B", "Stable entry observation"),
        (0.665, 0.945, "C", "Required age-confound correction"),
        (0.018, 0.475, "D", "SB-native phase profiles · tertiary · t=0.74"),
        (0.343, 0.475, "E", "Fixed domain-phase map across three disjoint splits"),
        (0.665, 0.475, "F", "Repaired frozen mapping test"),
    ]
    for x, y, letter, subtitle in labels:
        fig.text(x, y, letter, fontsize=13.5, weight="bold")
        fig.text(x + 0.022, y, subtitle, fontsize=9.2, weight="bold", va="center")
    fig.text(
        0.5,
        0.018,
        "Plain UNSB only · one fresh training seed · five domains · three zero-overlap image splits (20+24+16 per domain) · M=32. "
        "Effective age minimizes reciprocal common-state conditional-kernel direction distance to Single e1–e5. "
        "No paired target or output-quality selection; observational bridge geometry, not causal harm or calibrated uncertainty.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    output = tertiary_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output / "UNSB_MOTIVATION_HEADFIGURE_FINAL.png",
        "pdf": output / "UNSB_MOTIVATION_HEADFIGURE_FINAL.pdf",
        "svg": output / "UNSB_MOTIVATION_HEADFIGURE_FINAL.svg",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def signed(value: float) -> str:
    return f"{value:+.3f}"


def write_report(discovery_root: Path, first_root: Path, tertiary_root: Path, paths: dict[str, str]) -> Path:
    stage1 = pd.read_csv(discovery_root / "reports" / "BOOTSTRAP_SUMMARY.csv")
    reciprocal = json.loads((discovery_root / "reports" / "RECIPROCAL_ADJUDICATION.json").read_text(encoding="utf-8"))
    invalid = json.loads((first_root / "reports" / "PHASE_CONFIRMATION_ADJUDICATION.json").read_text(encoding="utf-8"))
    adjudication = json.loads((tertiary_root / "reports" / "TERTIARY_PHASE_ADJUDICATION.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(tertiary_root / "reports" / "TERTIARY_PHASE_CELL_SUMMARY.csv")
    split = json.loads((tertiary_root / "state" / "TERTIARY_SPLIT_AUDIT.json").read_text(encoding="utf-8"))
    measurement = json.loads((tertiary_root / "state" / "RECIPROCAL_MEASUREMENT_STATE.json").read_text(encoding="utf-8"))
    lines = [
        "# UNSB 动机搜索最终报告：共享桥域相位失同步",
        "",
        "> 研究范围：只比较同一个 plain UNSB 的 task-specific 与 All-in-One 无配对训练；没有接入任何候选算法，也没有用 paired target 或输出质量挑选现象。",
        "",
        "## 1. 最终裁决",
        "",
        f"**`{adjudication['verdict']}`**。",
        "",
        "在 Fog / Low-light / Rain / Rain-streak / Snow 五域上，同一个 AIO epoch-1 checkpoint 在 bridge time 0.50、0.74、0.86 分别对应固定的 task-specific 条件核年龄映射：",
        "",
        "- `t=0.50: e4 / e3 / e2 / e4 / e5`",
        "- `t=0.74: e4 / e3 / e2 / e4 / e5`",
        "- `t=0.86: e4 / e3 / e2 / e2 / e5`",
        "",
        "该映射先出现在 20 图/域发现 split，在额外 24 图/域 split 上 15/15 复现；修复失效 null 后，又在第三个 16 图/域、与前两组及历史测量均零重叠的 split 上 **15/15 复现**。最终 mapping permutation `p=0.0002`，14/15 单元 bootstrap modal share ≥80%，M16/M32 为 15/15。",
        "",
        "独立性更正：最初 discovery selector 存在 stem namespace 缺陷（历史账本为 suffix stem、冻结视图为 domain-prefixed stem），因此 100 张 discovery 图中有 14 张与历史测量身份重合。该组只保留为发现证据，不算独立确认；后续 24 图/域和 16 图/域两个 split 使用 canonical stem 排除，历史 overlap 均为 0，并完整复现同一 15-cell map。",
        "",
        "论文动机应写成：**一个共享 AIO 训练时钟并没有把所有天气域同步到同一个 task-specific Schrödinger-bridge transition phase。**",
        "",
        "## 2. 搜索过程：哪些漂亮结论被否决了",
        "",
        "### 2.1 初始方向离散增大：真实，但只能标记 EXPOSURE_ONLY",
        "",
        "历史六 seed 的 epoch-1 结果为 30/30 seed×domain 正号、18/18 seed×bridge-time pooled CI 位于零上方；fresh seed=2041 也复现 exposure-matched 差异。但 optimizer-clock 和 Single e1–e5 年龄包络控制不支持更强结论：",
        "",
        "| t | exposure gap | 95% CI | clock gap | 95% CI | age-envelope excess | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in (1, 2, 3):
        block = {row["contrast"]: row for _, row in stage1[stage1["bridge_time_index"] == t].iterrows()}
        lines.append(
            f"| {BRIDGE[t]:.2f} | {signed(block['exposure_gap']['mean'])} | [{signed(block['exposure_gap']['ci_low'])}, {signed(block['exposure_gap']['ci_high'])}] | "
            f"{signed(block['clock_gap']['mean'])} | [{signed(block['clock_gap']['ci_low'])}, {signed(block['clock_gap']['ci_high'])}] | "
            f"{signed(block['age_envelope_excess']['mean'])} | [{signed(block['age_envelope_excess']['ci_low'])}, {signed(block['age_envelope_excess']['ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "所以“ AIO 初始方向更散”只保留为进入问题的观测，不能成为必要性结论，也不能被称为校准不确定性。",
            "",
            "### 2.2 AIO 是否越出整个 Single 早期轨迹：否",
            "",
            f"互易共同状态探针裁决为 `{reciprocal['verdict']}`：三处 bridge time 只有 33%、33%、36% 图像高于 `max(MC floor, Single age span)`，0/3 过门。AIO 大体可以由某个 Single 年龄近似，因此“全新条件核”叙事被删除。",
            "",
            "### 2.3 第一次 phase confirmation 的置换 null 为什么失效",
            "",
            f"第一确认集本身给出与发现集完全相同的 15-cell map、M16/M32 15/15，但原协议把整个五年龄 KDD profile 跨域置换，产生 `{invalid['verdict']}`。该 profile 是相对不同域的不同 Single checkpoint 生成的，跨域不可交换；混合它们测试的是混合参考系统的随机谷底，不是固定域→年龄映射是否复现。因此该 null 永久标为 `INVALID_NULL_EXCHANGEABILITY`，原高 p 不作为科学 FAIL，15/15 也不直接“翻案”。",
            "",
            "修复协议预先固定前两 split 共同得到的 map，第三 split 只在每个 bridge time 内打乱五个预定年龄对五个固定域身份的指派，任何 KDD profile 都不离开自己的模型系统。",
            "",
            "## 3. 最终 SB-native 观测量",
            "",
            "对 AIO e1 和每个 Single age `e∈{1,…,5}`，分别在 AIO rollout 状态 `X_A` 与 Single rollout 状态 `X_S,e` 上计算随机 endpoint direction 的单位平均方向：",
            "",
            "`mu_r(X,t)=unit(mean_m(unit((G_r(X,t,z_m)-X)/(1-t))))`",
            "",
            "互易条件核方向距离为：",
            "",
            "`KDD_e = 1/2[(1-cos(mu_A(X_A),mu_S,e(X_A))) + (1-cos(mu_A(X_S,e),mu_S,e(X_S,e)))]`。",
            "",
            "域–时有效年龄定义为：",
            "",
            "`e*(d,t)=argmin_e mean_image KDD(d,t,e)`。",
            "",
            "它不是把 epoch 标签硬套给 AIO，而是在相同 bridge time、互易模型诱导状态上询问：AIO 当前条件转移方向最接近这个域的哪个 task-specific checkpoint。",
            "",
            "## 4. 第三 split 冻结裁决",
            "",
            f"- 固定 map 命中：**{adjudication['mapping_accuracy']}/{adjudication['mapping_total_cells']}**（门：≥14/15）。",
            f"- within-time mapping permutation：**p={adjudication['permutation_p']:.4f}**（门：≤0.001）。",
            f"- bootstrap 稳定且等于固定预测：**{adjudication['stable_predicted_cells']}/15**（门：≥13/15）。",
            f"- M16/M32 有效年龄一致：**{adjudication['M16_M32_agreement_cells']}/15**（门：≥14/15）。",
            f"- 每个 bridge time 都保留至少三种年龄且 range≥2：**{adjudication['time_structure_pass']}**。",
            "",
            "| t | domain | fixed age | M32 age | M16 age | bootstrap mode | modal share | KDD margin |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in cells.iterrows():
        lines.append(
            f"| {row['bridge_time_value']:.2f} | {row['domain']} | {int(row['fixed_predicted_age'])} | {int(row['effective_age_M32'])} | "
            f"{int(row['effective_age_M16'])} | {int(row['bootstrap_modal_age'])} | {row['bootstrap_modal_share']:.1%} | {row['profile_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 5. 它如何支撑论文动机",
            "",
            "1. **先有框架差异，再谈改法。** 对象始终是 plain UNSB：五个 task-specific 训练与一个 AIO 共享训练。主证据不是任何新算法相对 baseline 的得分。",
            "2. **落在 Schrödinger Bridge 本身。** 有效 phase 由模型诱导 bridge states 上的条件 endpoint transition direction 定义，不是通用梯度冲突、loss 大小或输出 PSNR。",
            "3. **指出可操作的结构错配。** 单一 global checkpoint 同时对应 Rain 的 e2、Low-light 的 e3、Fog 的 e4、Snow 的 e5；因此对所有域使用同一训练阶段假设并不成立。后续方法的直接靶点应是缩小或协调这种 domain×bridge-time phase spread。",
            "4. **不越过证据。** 本轮没有证明这种失同步必然伤害恢复质量；方法阶段必须分别验证 phase spread 被改善和最终质量得到提升。",
            "",
            "## 6. 头图读法",
            "",
            "- A：从框架层面提出五个独立 bridge clocks 与一个共享 clock 的差异。",
            "- B：六 seed 初始离散复现是稳定入口现象。",
            "- C：clock/age control 否决把入口现象直接写成必要性的做法。",
            "- D：第三 split 在 `t=0.74` 的五条 KDD–age profile；谷底分别位于 e4/e3/e2/e4/e5。",
            "- E：发现 20 图/域、第一 held-out 24 图/域、第三 held-out 16 图/域的 15-cell map 完全一致；held-out 格同时显示 bootstrap modal share。",
            "- F：修复后的 frozen mapping null；只打乱预定年龄指派，不移动域专用模型产生的 profile。",
            "",
            "## 7. 写作边界",
            "",
            "允许：在本地五域训练制度、seed=2041 下，一个 AIO e1 checkpoint 对应跨域不同的 task-specific conditional-kernel phases；固定映射在三个零重叠图像 split 上复现。",
            "",
            "禁止：多训练 seed 确认、外部/sealed confirmation、RainDS-syn 覆盖、校准 posterior uncertainty、因果恢复伤害，或已经证明某种 phase correction 会提升最终指标。",
            "",
            "## 8. 工程身份和成本",
            "",
            f"- 第三 split：`{split['status']}`；80 张；历史/发现/第一确认 overlap 均为 0；manifest `{split['manifest_sha256']}`。",
            "- 模型：fresh seed=2041；五个 Single e1–e5 与一个 AIO e1；本阶段无新训练。",
            f"- 第三测量：{measurement['age_rows']} age rows、{measurement['primary_rows']} primary rows、M=32、GPU {measurement['elapsed_seconds']/60:.1f} 分钟、target_content_read=false。",
            "- 统计：9999 mapping permutations；每 cell 5000 次 image bootstrap；M16 prefix 是冻结门。",
            "",
            "## 9. 文件",
            "",
            f"- PNG：`{paths['png']}`",
            f"- PDF：`{paths['pdf']}`",
            f"- SVG：`{paths['svg']}`",
            "- 最终裁决：`reports/TERTIARY_PHASE_ADJUDICATION.json`",
            "- 单元摘要：`reports/TERTIARY_PHASE_CELL_SUMMARY.csv`",
            "- 原始 evidence：`raw/RECIPROCAL_KERNEL_BY_AGE.csv`",
            "",
        ]
    )
    report = tertiary_root / "reports" / "UNSB_MOTIVATION_SEARCH_FINAL_CN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-root", required=True)
    parser.add_argument("--first-confirm-root", required=True)
    parser.add_argument("--tertiary-root", required=True)
    args = parser.parse_args()
    discovery_root = Path(args.discovery_root).resolve()
    first_root = Path(args.first_confirm_root).resolve()
    tertiary_root = Path(args.tertiary_root).resolve()
    paths = make_figure(discovery_root, first_root, tertiary_root)
    report = write_report(discovery_root, first_root, tertiary_root, paths)
    state = {
        "status": "complete",
        "figure_paths": paths,
        "figure_hashes": {Path(path).name: sha256_file(path) for path in paths.values()},
        "report": str(report),
        "report_sha256": sha256_file(report),
    }
    dump_json(tertiary_root / "state" / "FINAL_RENDER_STATE.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
