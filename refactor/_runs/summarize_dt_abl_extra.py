#!/usr/bin/env python3
"""Summarize the three late DT knock-out ablations against the clean plain baseline.

Read-only with respect to training artifacts; writes one new markdown file under
``refactor/_runs/`` and prints the same table to stdout.
"""

from __future__ import annotations

import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
PLAIN = os.path.join(
    ROOT, "metrics", "dtcov_clean_plain_e200", "metrics_summary.json"
)

ABLATIONS = [
    ("A2 frozen teacher -> self", "dt_abl_a2_self", "--dtcov_teacher self"),
    ("A3 domain x time EMA -> global", "dt_abl_a3_global", "--dtcov_norm_mode global"),
    ("A5 signal norm -> raw U", "dt_abl_a5_nonorm", "--dtcov_signal_norm off"),
]


def load_psnr(path: str) -> float:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return float(data["summary"]["overall"]["psnr"])


def main() -> None:
    plain = load_psnr(PLAIN)
    rows = []
    for label, name, flag in ABLATIONS:
        path = os.path.join(ROOT, "metrics_abl", name, "metrics_summary.json")
        if not os.path.exists(path):
            rows.append((label, flag, None))
            continue
        psnr = load_psnr(path)
        rows.append((label, flag, psnr))

    lines = [
        "# DT 补充 knock-out 结果（单 seed=2026，test40）",
        "",
        f"- 干净 plain PSNR：{plain:.4f}",
        "- 基线（grouped_domain, ramp-hold-decay）：18.8453（+0.8875）",
        "",
        "| 改动 | 开关 | PSNR | delta vs plain |",
        "|---|---|---:|---:|",
    ]
    for label, flag, psnr in rows:
        if psnr is None:
            lines.append(f"| {label} | `{flag}` | 待评测 | — |")
        else:
            lines.append(
                f"| {label} | `{flag}` | {psnr:.4f} | {psnr - plain:+.4f} |"
            )
    lines.append("")

    text = "\n".join(lines)
    print(text)
    out = os.path.join(ROOT, "DT_ABL_EXTRA_RESULT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
