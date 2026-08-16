"""Paired-image metric comparison with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def align_pairs(
    rows_a: list[dict],
    rows_b: list[dict],
    *,
    key: str = "filename",
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align two per-image rows by identity key, returning paired arrays."""
    b_by_key = {row[key]: row for row in rows_b}
    a_vals = []
    b_vals = []
    for row in rows_a:
        other = b_by_key.get(row[key])
        if other is None or row.get(metric) is None or other.get(metric) is None:
            continue
        a_vals.append(float(row[metric]))
        b_vals.append(float(other[metric]))
    return np.asarray(a_vals, dtype=np.float64), np.asarray(b_vals, dtype=np.float64)


def paired_bootstrap(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    *,
    n_bootstrap: int = 50000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> dict:
    """Mean of (a - b) with percentile bootstrap CI over paired images."""
    a_vals = np.asarray(a_vals, dtype=np.float64)
    b_vals = np.asarray(b_vals, dtype=np.float64)
    if a_vals.shape != b_vals.shape or a_vals.size == 0:
        return {
            "n": 0,
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
        }

    diff = a_vals - b_vals
    rng = np.random.default_rng(seed)
    n = diff.size
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = diff[indices].mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {
        "n": int(n),
        "mean": float(diff.mean()),
        "ci_low": low,
        "ci_high": high,
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
    }
