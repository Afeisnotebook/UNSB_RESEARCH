#!/usr/bin/env python3
"""Combine mechanism indicators into a transparent compression-score table."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mechanism_common import BRIDGE_TIMES, DOMAINS, EPOCHS


def load_raw_indicators(raw_dir: Path) -> dict[int, dict[str, float]]:
    rows = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by_epoch = defaultdict(list)
    for r in rows:
        if r.get("method") != "aio_plain":
            continue
        epoch = int(r.get("epoch", -1))
        if epoch in EPOCHS:
            by_epoch[epoch].append(r)
    out = {}
    for epoch, rr in by_epoch.items():
        logu = [float(r["log_U"]) for r in rr if "log_U" in r]
        ureg = []
        for r in rr:
            if "u_map" in r:
                ureg.append(float(np.asarray(r["u_map"]).sum()))
        out[epoch] = {
            "aio_logU": float(np.median(logu)) if logu else float("nan"),
            "aio_ureg": float(np.median(ureg)) if ureg else float("nan"),
        }
    return out


def load_direction_rank(path: Path) -> dict[int, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for epoch_str, v in data.get("aio", {}).items():
        overall = v.get("overall", v)
        out[int(epoch_str)] = {k: overall.get(k, float("nan")) for k in ["effective_rank", "mean_energy", "spectral_entropy"]}
    return out


def load_gradient_proxy(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for epoch_str, v in data.get("epochs", {}).items():
        vals = []
        for t in ["t1", "t2", "t3"]:
            block = v.get(t, {})
            if "mean_cosine" in block and np.isfinite(block["mean_cosine"]):
                vals.append(float(block["mean_cosine"]))
        out[int(epoch_str)] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_feature_alignment(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for epoch_str, v in data.get("epochs", {}).items():
        vals = []
        for layer, block in v.items():
            if isinstance(block, dict) and "mean_cka" in block and np.isfinite(block["mean_cka"]):
                vals.append(float(block["mean_cka"]))
        out[int(epoch_str)] = float(np.mean(vals)) if vals else float("nan")
    return out


def zscore(vals: dict[int, float]) -> dict[int, float]:
    epochs = sorted(vals)
    arr = np.asarray([vals[e] for e in epochs if np.isfinite(vals[e])], dtype=np.float64)
    if arr.size < 2 or np.std(arr) < 1e-12:
        return {e: 0.0 for e in epochs}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {e: (float(vals[e]) - mean) / std if np.isfinite(vals[e]) else float("nan") for e in epochs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--direction-rank-json", required=True)
    parser.add_argument("--gradient-proxy-json", required=True)
    parser.add_argument("--feature-alignment-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    raw_ind = load_raw_indicators(Path(args.raw_dir))
    rank_ind = load_direction_rank(Path(args.direction_rank_json))
    grad_ind = load_gradient_proxy(Path(args.gradient_proxy_json))
    feat_ind = load_feature_alignment(Path(args.feature_alignment_json))

    epochs = [e for e in EPOCHS if e in raw_ind]
    table = []
    components = {}

    # Positive z means more compressed; invert indicators where larger means less compressed.
    comp_inputs = {
        "aio_logU": {e: raw_ind[e]["aio_logU"] for e in epochs},
        "aio_ureg": {e: raw_ind[e]["aio_ureg"] for e in epochs},
        "effective_rank": {e: rank_ind.get(e, {}).get("effective_rank", float("nan")) for e in epochs},
        "mean_energy": {e: rank_ind.get(e, {}).get("mean_energy", float("nan")) for e in epochs},
        "gradient_proxy_cosine": {e: grad_ind.get(e, float("nan")) for e in epochs},
        "feature_cka": {e: feat_ind.get(e, float("nan")) for e in epochs},
    }
    sign = {
        "aio_logU": -1.0,
        "aio_ureg": -1.0,
        "effective_rank": -1.0,
        "mean_energy": 1.0,
        "gradient_proxy_cosine": 1.0,
        "feature_cka": 1.0,
    }
    for name, vals in comp_inputs.items():
        z = zscore(vals)
        components[name] = {str(e): {"value": vals[e], "z_compression": sign[name] * z[e] if np.isfinite(z[e]) else None} for e in epochs}

    for e in epochs:
        scores = []
        row = {"epoch": e}
        for name in comp_inputs:
            v = components[name][str(e)]["value"]
            zc = components[name][str(e)]["z_compression"]
            row[name] = v
            row[f"{name}_z_compression"] = zc
            if zc is not None and np.isfinite(zc):
                scores.append(float(zc))
        row["compression_score"] = float(np.mean(scores)) if scores else float("nan")
        row["n_components"] = len(scores)
        table.append(row)

    out = {"epochs": table, "components": components}
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out_json, "n_epochs": len(table)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
