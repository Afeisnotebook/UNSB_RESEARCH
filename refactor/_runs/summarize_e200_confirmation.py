#!/usr/bin/env python3
"""Summarize the two-variant e200 confirmation into human-readable evidence."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path("/home/yc/unsb_tired/refactor/_runs/hnek_search")
OUT_MD = BASE / "E200_CONFIRMATION.md"
OUT_JSON = BASE / "E200_CONFIRMATION.json"

VARIANTS = [
    {
        "name": "hnek_coord_y",
        "label": "Variant A: endpoint coordinate (X_t,Y)",
        "args": "--gamma 0.5 --coord endpoint --horizon-mode physical --partial all",
    },
    {
        "name": "hnek_g0.25",
        "label": "Variant B: gamma=0.25 residual coordinate",
        "args": "--gamma 0.25 --coord residual --horizon-mode physical --partial all",
    },
]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> int:
    rows = []
    for variant in VARIANTS:
        name = variant["name"]
        e50_summary = load(BASE / "state" / name / "eval_e50" / "SUMMARY.json")
        e200_summary = load(BASE / "state" / name / "eval_e200" / "SUMMARY.json")
        e50_decision = load(BASE / "state" / name / "E50_ADJUDICATION.json")
        e200_decision = load(BASE / "state" / name / "E200_ADJUDICATION.json")

        e50_delta = (
            float(e50_summary["macro_psnr_delta_db"])
            if e50_summary
            else float("nan")
        )
        e200_delta = (
            float(e200_summary["macro_psnr_delta_db"])
            if e200_summary
            else float("nan")
        )
        e200_ci = (
            [float(x) for x in e200_summary["psnr_ci95"]]
            if e200_summary and "psnr_ci95" in e200_summary
            else [float("nan"), float("nan")]
        )
        e200_pos = (
            int(e200_summary["positive_domains"])
            if e200_summary
            else None
        )
        e200_verdict = (
            e200_decision.get("verdict") if e200_decision else "MISSING"
        )
        change = e200_delta - e50_delta if e200_summary and e50_summary else float("nan")
        rows.append({
            "name": name,
            "label": variant["label"],
            "args": variant["args"],
            "e50_delta_db": e50_delta,
            "e200_delta_db": e200_delta,
            "e200_ci95": e200_ci,
            "e200_positive_domains": e200_pos,
            "e200_verdict": e200_verdict,
            "change_e200_minus_e50_db": change,
        })

    lines = [
        "# HNEK e200 confirmation",
        "",
        "> 由 `summarize_e200_confirmation.py` 自动生成。只做单 seed=2026 的 paired-development 确认，不构成 confirmatory 结论。",
        "",
        "| variant | e50 delta | e200 delta | e200 95% CI | e200 positive domains | e200 verdict | change (e200−e50) |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        ci = r["e200_ci95"]
        lines.append(
            "| {name} | {e50:.4f} | {e200:.4f} | [{lo:.4f}, {hi:.4f}] | {pos} | {verdict} | {change:+.4f} |".format(
                name=r["name"],
                e50=r["e50_delta_db"],
                e200=r["e200_delta_db"],
                lo=ci[0],
                hi=ci[1],
                pos=r["e200_positive_domains"],
                verdict=r["e200_verdict"],
                change=r["change_e200_minus_e50_db"],
            )
        )

    lines.extend([
        "",
        "## 判定",
        "",
    ])
    passed = [
        r for r in rows
        if r["e200_delta_db"] > 0
        and r["e200_ci95"][0] > 0
        and (r["e200_positive_domains"] or 0) >= 3
    ]
    if passed:
        lines.append(
            "e200 后仍保持正向且 CI 不含 0、positive domains ≥ 3/5 的变体："
            + ", ".join(r["name"] for r in passed)
            + "。"
        )
    else:
        lines.append("没有变体同时满足 delta > 0、CI 不含 0、positive domains ≥ 3/5。")

    lines.extend([
        "",
        "## 下一步建议（本机不做多 seed）",
        "",
        "- 若至少一个变体通过上述判定，下一步应在更好的服务器上以同 seed=2026 复现，再补 2~4 个独立 seed；",
        "- 优先验证最强且实现解释最清晰的变体；",
        "- 多 seed 报告使用 mean±std/CI，并继续注明 reflection/adaptive pool 的 limitation 口径。",
        "",
    ])

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "design": "single_seed_e200_confirmation",
                "variants": rows,
                "passed": [r["name"] for r in passed],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
