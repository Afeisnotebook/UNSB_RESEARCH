#!/usr/bin/env python3
"""Aggregate extracted path-geometry rows into c/d/e summaries (and adjudication inputs)."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from measure_path_geometry import paired_bootstrap


TAG_RE = re.compile(r"^(?P<method>.+)__e(?P<epoch>\d+)$")


def load_rows(raw_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def domain_balanced_median_logu(rows: list[dict]) -> float:
    by_domain: dict[str, list[float]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(float(r["log_U"]))
    if not by_domain:
        return float("nan")
    return float(np.median([np.median(v) for v in by_domain.values()]))


def domain_balanced_ci_logu(rows: list[dict], n_boot: int = 10000, seed: int = 2026) -> dict:
    by_domain: dict[str, list[float]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(float(r["log_U"]))
    domains = sorted(by_domain)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        meds = []
        for d in domains:
            vals = np.asarray(by_domain[d])
            idx = rng.integers(0, len(vals), size=len(vals))
            meds.append(float(np.median(vals[idx])))
        stats.append(float(np.median(meds)))
    stats = np.asarray(stats)
    return {
        "median": float(np.median(stats)),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
    }


def panel_c(rows: list[dict], bridge_times: list[int]) -> dict:
    out = {}
    methods = sorted({r["method"] for r in rows if "method" in r})
    for method in methods:
        epochs = sorted({int(r["epoch"]) for r in rows if r["method"] == method})
        for t in bridge_times:
            series = []
            for epoch in epochs:
                sub = [
                    r
                    for r in rows
                    if r["method"] == method
                    and int(r["epoch"]) == epoch
                    and int(r["bridge_time_index"]) == t
                ]
                if not sub:
                    continue
                ci = domain_balanced_ci_logu(sub)
                series.append(
                    {
                        "epoch": epoch,
                        "median_log_U": domain_balanced_median_logu(sub),
                        **ci,
                    }
                )
            out.setdefault(method, {})[f"t{t}"] = series
    return out


def panel_e(rows: list[dict], bridge_times: list[int], domains: list[str]) -> dict:
    out = {}
    for t in bridge_times:
        aio = {
            (r["domain"], r["stem"]): r
            for r in rows
            if r.get("method") == "aio_plain" and int(r["bridge_time_index"]) == t
        }
        singles = {}
        for r in rows:
            m = r.get("method", "")
            if m.startswith("single_") and int(r["bridge_time_index"]) == t:
                singles.setdefault(r["domain"], {})[(r["domain"], r["stem"])] = r

        per_domain = {}
        pooled_a = []
        pooled_b = []
        for domain in domains:
            sd = singles.get(domain, {})
            a_vals, b_vals = [], []
            for key, aio_row in aio.items():
                if key[0] != domain or key not in sd:
                    continue
                a_vals.append(float(aio_row["U"]))
                b_vals.append(float(sd[key]["U"]))
            if a_vals:
                pooled_a.extend(a_vals)
                pooled_b.extend(b_vals)
                per_domain[domain] = paired_bootstrap(
                    np.asarray(a_vals), np.asarray(b_vals)
                )
        pooled = paired_bootstrap(np.asarray(pooled_a), np.asarray(pooled_b))
        out[f"t{t}"] = {"per_domain": per_domain, "pooled": pooled}
    return out


def panel_d(rows: list[dict], bridge_times: list[int]) -> dict:
    out = {}
    methods = sorted({r["method"] for r in rows if "method" in r})
    for method in methods:
        for t in bridge_times:
            by_domain: dict[str, list[np.ndarray]] = {}
            for r in rows:
                if (
                    r.get("method") == method
                    and int(r["bridge_time_index"]) == t
                    and "u_map" in r
                ):
                    by_domain.setdefault(r["domain"], []).append(
                        np.asarray(r["u_map"], dtype=np.float64)
                    )
            if not by_domain:
                continue
            domain_means = [np.mean(np.stack(v), axis=0) for v in by_domain.values()]
            out.setdefault(method, {})[f"t{t}"] = float(
                np.mean(np.stack(domain_means), axis=0).sum()
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--domains", default="FoggyCityscapes,LowLightTrafficData,RainCityscapes,RSCityscapes,SnowTrafficData")
    parser.add_argument("--bridge-times", default="1,2,3")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    bridge_times = [int(x) for x in args.bridge_times.split(",") if x.strip()]
    rows = load_rows(Path(args.raw_dir))
    summary = {
        "panel_c": panel_c(rows, bridge_times),
        "panel_e": panel_e(rows, bridge_times, domains),
        "panel_d": panel_d(rows, bridge_times),
        "n_rows": len(rows),
    }
    Path(args.out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"n_rows": len(rows), "out": args.out}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
