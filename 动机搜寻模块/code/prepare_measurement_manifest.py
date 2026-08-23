#!/usr/bin/env python3
"""Build the effect-blind measurement image manifest from DATA_MANIFEST.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def rgb_mean(path: str) -> float:
    img = Image.open(path).convert("RGB").resize((64, 64), Image.BILINEAR)
    return float(np.asarray(img, dtype=np.float32).mean())


def main() -> int:
    data = json.loads((PROJECT_ROOT / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    discovery = [
        r for r in data["files"] if r["role"] == "discovery" and r["side"] == "input"
    ]
    by_domain: dict[str, list[dict]] = {}
    for r in discovery:
        by_domain.setdefault(r["domain"], []).append(r)

    medoids = {}
    c_subset = {}
    d_subset = {}
    for domain, rows in sorted(by_domain.items()):
        rows = sorted(rows, key=lambda r: r["stem"])
        means = np.array([rgb_mean(r["source_path"]) for r in rows])
        medoid_idx = int(np.argmin(np.abs(means - np.median(means))))
        medoids[domain] = rows[medoid_idx]
        c_subset[domain] = rows[:10]
        d_subset[domain] = rows[:10]

    manifest = {
        "discovery": discovery,
        "b_medoids": list(medoids.values()),
        "c_subset": [r for rows in c_subset.values() for r in rows],
        "d_subset": [r for rows in d_subset.values() for r in rows],
    }
    (PROJECT_ROOT / "MEASUREMENT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "discovery": len(discovery),
                "b_medoids": len(manifest["b_medoids"]),
                "c_subset": len(manifest["c_subset"]),
                "d_subset": len(manifest["d_subset"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
