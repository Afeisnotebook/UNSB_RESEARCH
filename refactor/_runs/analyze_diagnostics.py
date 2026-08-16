#!/usr/bin/env python3
"""Read-only audit of the intervention diagnostics JSONLs.

Prints, from raw diagnostic lines, the first epoch where each mechanism becomes
active and its steady-state level, so the "effective intervention point" claim
is traceable to the logged evidence rather than a prose note.
"""

from __future__ import annotations

import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(ROOT, "diagnostics")


def load(name: str) -> list[dict]:
    path = os.path.join(DIAG, name)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_epoch(rows: list[dict], key: str, thresh: float = 0.0) -> int | None:
    for row in rows:
        if float(row.get(key, 0.0)) > thresh:
            return int(row["epoch"])
    return None


def mean_last(rows: list[dict], key: str, k: int = 10) -> float:
    tail = [float(r.get(key, 0.0)) for r in rows[-k:]]
    return sum(tail) / len(tail) if tail else 0.0


def main() -> int:
    dt = load("dtcov_adaptive.jsonl")
    hj = load("hj_adaptive.jsonl")

    print("DT (dtcov_adaptive.jsonl)")
    print(f"  first drift>0 epoch: {first_epoch(dt, 'drift', 1e-6)}")
    print(f"  first lambda>=0.001 epoch: {first_epoch(dt, 'lambda_value', 0.00099)}")
    print(f"  last-10 drift mean: {mean_last(dt, 'drift'):.4f}")
    print(f"  last-10 SB entropy grad norm mean: {mean_last(dt, 'sb_entropy_grad_norm'):.4f}")

    print("HJ (hj_adaptive.jsonl)")
    print(f"  first gate_hit_rate>0 epoch: {first_epoch(hj, 'gate_hit_rate', 1e-6)}")
    print(f"  last-10 gate_hit_rate mean: {mean_last(hj, 'gate_hit_rate'):.4f}")
    print(f"  last-10 risk_mean mean: {mean_last(hj, 'risk_mean'):.4f}")
    print(f"  last-10 risk_positive mean: {mean_last(hj, 'risk_positive'):.4f}")
    print(f"  last-10 SB entropy grad norm mean: {mean_last(hj, 'sb_entropy_grad_norm'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
