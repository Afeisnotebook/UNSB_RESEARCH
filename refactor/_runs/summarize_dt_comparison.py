"""Summarize clean DT vs clean plain after both evals finish."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import metrics  # noqa: E402


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    base = Path("/home/yc/unsb_tired/refactor/_runs/metrics")
    plain_dir = base / "dtcov_clean_plain_e200"
    dt_dir = base / "dtcov_clean_best_e200"
    plain = load_csv(plain_dir / "metrics_per_image.csv")
    dt = load_csv(dt_dir / "metrics_per_image.csv")

    out = {}
    for metric_name in ("psnr", "ssim", "lpips", "niqe"):
        a, b = metrics.align_pairs(dt, plain, key="filename", metric=metric_name)
        boot = metrics.paired_bootstrap(a, b, n_bootstrap=50000, seed=2026)
        out[metric_name] = {
            "n": boot["n"],
            "mean_delta": round(boot["mean"], 4) if boot["mean"] is not None else None,
            "ci_low": round(boot["ci_low"], 4) if boot["ci_low"] is not None else None,
            "ci_high": round(boot["ci_high"], 4) if boot["ci_high"] is not None else None,
        }

    report = {
        "comparison": "dtcov_clean_best_e200 - dtcov_clean_plain_e200",
        "target": "clean-framework relative gain (DT vs clean plain); baseline is +0.8875 dB (see refactor/BASELINE_DECISION.md)",
        "metrics": out,
    }
    dest = base / "dt_clean_comparison.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
