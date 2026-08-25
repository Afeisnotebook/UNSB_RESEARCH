#!/usr/bin/env python3
"""Produce a draft adjudication ledger and Chinese report from evidence.

This script is deliberately conservative.  The experiment is single-seed and the
frozen expression policy forbids several strong claims, so the generated report
is a *draft for human confirmation*, not a final scientific verdict.  Claims
that can only be weakly supported are marked SUPPORTED_SCREEN and always carry
``needs_human_confirmation=true``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np


DEFAULT_DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
ALLOWED_TERMS = {
    "SUPPORTED",
    "SUPPORTED_SCREEN",
    "UNRESOLVED",
    "NOT_SUPPORTED",
    "FAILED",
}


def _series(summary: dict, method: str, t: int) -> list[dict]:
    return (
        summary.get("panel_c", {})
        .get(method, {})
        .get(f"t{t}", [])
    )


def _median_series_value(series: list[dict]) -> float | None:
    vals = [float(x.get("median_log_U")) for x in series if "median_log_U" in x]
    return float(np.median(vals)) if vals else None


def _ci_overlap(a: dict, b: dict) -> bool:
    if not a or not b:
        return True
    return float(a.get("ci_low", float("-inf"))) <= float(b.get("ci_high", float("inf"))) and float(
        b.get("ci_low", float("-inf"))
    ) <= float(a.get("ci_high", float("inf")))


def evidence_status(rows_count: int, summary_exists: bool) -> str:
    if not summary_exists or rows_count == 0:
        return "FAILED"
    return "SUPPORTED_SCREEN" if rows_count > 0 else "UNRESOLVED"


def assess_stage_dependent_geometry(summary: dict) -> dict:
    """Compare AIO plain against each single domain across epochs/t.

    A claim is only marked SUPPORTED_SCREEN when the direction is not constant
    across stages (sign changes) or CIs separate at some stage, while remaining
    single-seed and therefore human-confirmation-only.
    """
    t_keys = [f"t{t}" for t in (1, 2, 3)]
    details = {}
    any_separation = False
    any_sign_change = False
    any_data = False

    for domain in DEFAULT_DOMAINS:
        single_method = f"single_{domain}_s2026"
        domain_detail = {}
        for t in (1, 2, 3):
            aio = _series(summary, "aio_plain", t)
            single = _series(summary, single_method, t)
            if not aio or not single:
                continue
            by_epoch_a = {int(x["epoch"]): x for x in aio}
            by_epoch_s = {int(x["epoch"]): x for x in single}
            common = sorted(set(by_epoch_a) & set(by_epoch_s))
            if not common:
                continue
            any_data = True
            diffs = []
            separated = False
            for epoch in common:
                a = by_epoch_a[epoch]
                s = by_epoch_s[epoch]
                diff = float(a["median_log_U"]) - float(s["median_log_U"])
                diffs.append(diff)
                if not _ci_overlap(a, s):
                    separated = True
            if separated:
                any_separation = True
            if diffs and (min(diffs) < 0 < max(diffs)):
                any_sign_change = True
            domain_detail[f"t{t}"] = {
                "n_common_epochs": len(common),
                "median_diff": float(np.median(diffs)) if diffs else None,
                "min_diff": float(min(diffs)) if diffs else None,
                "max_diff": float(max(diffs)) if diffs else None,
                "sign_changes": bool(diffs and min(diffs) < 0 < max(diffs)),
                "ci_separation_in_any_epoch": bool(separated),
            }
        details[domain] = domain_detail

    if not any_data:
        status = "FAILED"
    elif any_sign_change or any_separation:
        status = "SUPPORTED_SCREEN"
    else:
        status = "UNRESOLVED"

    return {
        "status": status,
        "sign_changes": any_sign_change,
        "ci_separation": any_separation,
        "detail": details,
    }


def assess_dt_sanity(summary: dict) -> dict:
    """Compare AIO DT against AIO plain for the post-hoc sanity check."""
    wins = 0
    total = 0
    details = {}
    for t in (1, 2, 3):
        aio = _series(summary, "aio_plain", t)
        dt_rows = _series(summary, "aio_dt", t)
        if not aio or not dt_rows:
            continue
        aio_by = {int(x["epoch"]): x for x in aio}
        dt_by = {int(x["epoch"]): x for x in dt_rows}
        common = sorted(set(aio_by) & set(dt_by))
        diffs = []
        for epoch in common:
            diff = float(dt_by[epoch]["median_log_U"]) - float(aio_by[epoch]["median_log_U"])
            diffs.append(diff)
            if diff < 0:
                wins += 1
            total += 1
        details[f"t{t}"] = {
            "n_common_epochs": len(common),
            "median_dt_minus_aio": float(np.median(diffs)) if diffs else None,
        }
    if total == 0:
        return {"status": "UNRESOLVED", "detail": details}
    ratio = wins / total
    if ratio >= 0.6:
        status = "SUPPORTED_SCREEN"
    elif ratio <= 0.4:
        status = "NOT_SUPPORTED"
    else:
        status = "UNRESOLVED"
    return {"status": status, "fraction_dt_lower": ratio, "detail": details}


def build_claims(summary: dict, rows_count: int, summary_exists: bool) -> list[dict]:
    base_status = evidence_status(rows_count, summary_exists)
    stage = assess_stage_dependent_geometry(summary)
    dt_sanity = assess_dt_sanity(summary)

    claims = []
    if not summary_exists or rows_count == 0:
        claims.append(
            {
                "claim_id": "pipeline_complete",
                "text_cn": "训练/测量/汇总链路是否产生了完整 raw 证据与汇总结果",
                "status": "FAILED",
                "evidence_files": [],
                "needs_human_confirmation": True,
                "human_confirmation_items": ["确认训练日志、raw JSONL/NPZ 与 summary 是否缺失；缺失则不应继续裁决。"],
            }
        )
        return claims

    claims.append(
        {
            "claim_id": "geometry_different_stage_dependent",
            "text_cn": "Single 与 Plain All-in-One 的条件方向几何不同，且差异具有阶段依赖",
            "status": stage["status"] if stage["status"] in ALLOWED_TERMS else "UNRESOLVED",
            "evidence_files": ["raw/*.jsonl", "reports/MOTIVATION_SUMMARY.json", "figures/panel_c.png"],
            "needs_human_confirmation": True,
            "human_confirmation_items": [
                "确认 panel_c 的跨域加权口径、epoch 对齐口径与 bootstrap CI 是否恰当。",
                "确认差异方向是否在阶段间反转，以及单 seed 下是否只能写 SUPPORTED_SCREEN。",
                "不得升级为“AIO 全程更分散”或“U 是校准不确定性”。",
            ],
            "evidence_detail": stage,
        }
    )
    claims.append(
        {
            "claim_id": "rain_and_multidomain_nonuniform",
            "text_cn": "Rain/多域下路径几何发生改变，但并非全程同号",
            "status": stage["status"] if stage["status"] in ALLOWED_TERMS else "UNRESOLVED",
            "evidence_files": ["raw/*.jsonl", "reports/MOTIVATION_SUMMARY.json", "figures/panel_c.png"],
            "needs_human_confirmation": True,
            "human_confirmation_items": [
                "分别核对 RainCityscapes 及其他域的 t=1/2/3 轨迹和符号。",
                "若仅部分域非全程同号，结论应限定到对应域，不做全称断言。",
            ],
            "evidence_detail": stage,
        }
    )
    claims.append(
        {
            "claim_id": "dt_sanity_lower_U",
            "text_cn": "DT 作为路径尺度干预 sanity check，相对 Plain AIO 降低 U",
            "status": dt_sanity["status"],
            "evidence_files": ["raw/*.jsonl", "reports/MOTIVATION_SUMMARY.json", "figures/panel_c.png"],
            "needs_human_confirmation": True,
            "human_confirmation_items": [
                "确认 DT 只作为 post-hoc sanity check，不进入主对照。",
                "确认激活窗口长度、warmup 与 bridge time 对齐是否影响该比较。",
            ],
            "evidence_detail": dt_sanity,
        }
    )
    claims.append(
        {
            "claim_id": "single_seed_scope",
            "text_cn": "本旁路只允许 SUPPORTED / SUPPORTED_SCREEN 级别方向性措辞，禁止全称因果或校准结论",
            "status": "SUPPORTED",
            "evidence_files": ["MOTIVATION_FROZEN_SPEC.json", "reports/MOTIVATION_FIGURE_REPORT_CN.md"],
            "needs_human_confirmation": False,
            "human_confirmation_items": [],
            "evidence_detail": {"seed": 2026, "arms": ["single x5", "aio_plain", "aio_dt sanity"]},
        }
    )
    return claims


def _cn_status(status: str) -> str:
    return {
        "SUPPORTED": "支持",
        "SUPPORTED_SCREEN": "支持性筛查（单 seed/方向性，需人工确认）",
        "UNRESOLVED": "未决",
        "NOT_SUPPORTED": "不支持",
        "FAILED": "链路失败/证据缺失",
    }.get(status, status)


def render_report(
    summary_path: Path,
    raw_dir: Path,
    claims: list[dict],
    summary_exists: bool,
    rows_count: int,
    out_path: Path,
) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = []
    lines.append("# UNSB 动机图纯基线重启 — 自动裁决草稿")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：`{now}`")
    lines.append(f"- 汇总文件：`{summary_path}`")
    lines.append(f"- raw 行数：`{rows_count}`")
    lines.append(f"- 汇总存在：`{summary_exists}`")
    lines.append("")
    lines.append("> 这是**草稿**，不是最终科学结论。单 seed=2026，所有方向性结论必须先人工确认。")
    lines.append("")
    lines.append("## 自动裁决结果")
    lines.append("")
    lines.append("| claim_id | 状态 | 需人工确认 | 文本 |")
    lines.append("|---|---|---|---|")
    for c in claims:
        lines.append(
            f"| {c['claim_id']} | {_cn_status(c['status'])} | "
            f"{'是' if c['needs_human_confirmation'] else '否'} | {c['text_cn']} |"
        )
    lines.append("")
    lines.append("## 自动结果要点（仍需人工核对）")
    lines.append("")
    for c in claims:
        if c["claim_id"] == "geometry_different_stage_dependent":
            d = c.get("evidence_detail", {})
            lines.append(
                "- 几何阶段依赖：状态 "
                f"{_cn_status(c['status'])}；sign_changes={d.get('sign_changes')}；"
                f"ci_separation={d.get('ci_separation')}。"
            )
        elif c["claim_id"] == "rain_and_multidomain_nonuniform":
            d = c.get("evidence_detail", {})
            lines.append(
                "- Rain/多域非全程同号：状态 "
                f"{_cn_status(c['status'])}；sign_changes={d.get('sign_changes')}；"
                f"ci_separation={d.get('ci_separation')}。"
            )
        elif c["claim_id"] == "dt_sanity_lower_U":
            d = c.get("evidence_detail", {})
            lines.append(
                "- DT sanity check：状态 "
                f"{_cn_status(c['status'])}；fraction_dt_lower={d.get('fraction_dt_lower')}。"
            )
    lines.append("")
    lines.append("## 证据文件")
    lines.append("")
    if summary_exists and rows_count:
        lines.append("- `raw/*.jsonl`：每 image / bridge-time 的 U、log U、u_map。")
        lines.append("- `raw/panel_b_directions.npz` 与 `raw/panel_b_pca.json`：panel_b 联合 PCA。")
        lines.append("- `reports/MOTIVATION_SUMMARY.json`：panel_c/d/e 汇总。")
        lines.append("- `figures/panel_b.png`、`panel_c.png`、`panel_d.png`、`panel_e.png`。")
    else:
        lines.append("- 当前证据缺失或汇总不存在。")
    lines.append("")
    lines.append("## 作者睡醒后需要人工确认什么")
    lines.append("")
    lines.append("1. 检查 `WAIT_MAIN`、GPU 占用与训练日志，确认没有抢占主线或静默降级。")
    lines.append("2. 检查单 seed 轨迹：panel_c 中哪些域/哪个 bridge time 出现阶段反转或 CI 分离。")
    lines.append("3. 检查 panel_e 的 paired bootstrap 是否只是少数图像/单一 bridge time 的结果。")
    lines.append("4. 检查 panel_d 的 32px 区域图是否符合物理预期，且不是由显示尺度造成的伪影。")
    lines.append("5. 检查 DT sanity check 是否只被用作后置机制一致性，而非主对照结论。")
    lines.append("6. 逐条核对 `CLAIM_LEDGER.json` 中每个 `SUPPORTED_SCREEN`/`UNRESOLVED`/`NOT_SUPPORTED`/`FAILED` 项。")
    lines.append("7. 禁止将结果改写成“AIO 全程更分散”“U 是校准不确定性”“HJ 修复局部冲突”。")
    lines.append("")
    lines.append("## 结论保留边界")
    lines.append("")
    lines.append("- 允许：Single 与 AIO 条件方向几何不同（stage-dependent）。")
    lines.append("- 允许：Rain/多域下路径几何改变，但非全程同号。")
    lines.append("- 允许：DT 作为路径尺度干预降低 U（仅机制一致性 sanity check）。")
    lines.append("- 禁止：任何全称因果、校准 posterior/epistemic、AIO 全程更分散、HJ 修复局部冲突的表述。")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    raw_dir = Path(args.raw_dir)
    report_path = Path(args.report)
    ledger_path = Path(args.ledger)

    summary_exists = summary_path.exists()
    summary = {}
    if summary_exists:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows_count = 0
    if raw_dir.exists():
        for p in raw_dir.glob("*.jsonl"):
            rows_count += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())

    claims = build_claims(summary, rows_count, summary_exists)
    ledger = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "single_seed": 2026,
        "adjudication_policy": "draft_only_until_human_confirmation",
        "claims": claims,
    }
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(summary_path, raw_dir, claims, summary_exists, rows_count, report_path)
    print(json.dumps({"claims": len(claims), "ledger": str(ledger_path), "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
