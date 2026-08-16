#!/usr/bin/env python3
"""Aggregate per-seed deltas into mean +/- CI across seeds.

Usage:
    python aggregate_multiseed.py 0.8875 0.74 1.05

Prints n / mean / std / sem / 95% CI (Student-t for n>=2, normal fallback).
"""

from __future__ import annotations

import math
import sys


# Two-sided t critical values for 95% CI, indexed by degrees of freedom (n-1).
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: aggregate_multiseed.py delta1 delta2 ...", file=sys.stderr)
        return 2
    try:
        vals = [float(x) for x in argv[1:]]
    except ValueError:
        print("all arguments must be numeric", file=sys.stderr)
        return 2

    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / max(n - 1, 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    t = _T95.get(n - 1, 1.96)
    half = t * sem
    print(f"n={n} mean={mean:.4f} std={std:.4f} sem={sem:.4f} "
          f"ci95=[{mean-half:.4f}, {mean+half:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
