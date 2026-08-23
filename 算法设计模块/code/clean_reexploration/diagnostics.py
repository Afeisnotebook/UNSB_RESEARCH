"""Target-blind diagnostic panel and pure statistic helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _stem_key(stem: str) -> str:
    return hashlib.sha256(stem.encode("utf-8")).hexdigest()


def build_diagnostic_panel(
    training_manifest: list[dict],
    *,
    per_domain_a: int = 16,
    per_domain_b: int = 16,
    seed: int = 20260824,
) -> dict:
    """Select a frozen, domain-stratified, target-blind diagnostic panel.

    The panel is selected deterministically from the T2 unpaired A/B identities
    (stem SHA-256 sorted), never reading pixels or paired targets.
    """
    domains = sorted({f["domain"] for f in training_manifest})
    rng = np.random.default_rng(seed)
    panel: dict[str, dict[str, list[str]]] = {}
    for domain in domains:
        a_stems = sorted(
            {f["stem"] for f in training_manifest if f["domain"] == domain and f["side"] == "A"}
        )
        b_stems = sorted(
            {f["stem"] for f in training_manifest if f["domain"] == domain and f["side"] == "B"}
        )
        a_stems.sort(key=_stem_key)
        b_stems.sort(key=_stem_key)
        panel[domain] = {
            "A": a_stems[: int(per_domain_a)],
            "B": b_stems[: int(per_domain_b)],
        }
    return panel


def panel_to_manifest_rows(panel: dict, training_manifest: list[dict]) -> list[dict]:
    """Map a selected panel back to full manifest file rows (unpaired A/B only)."""
    by_key = {
        (f["domain"], f["side"], f["stem"]): f for f in training_manifest
        if f["side"] in ("A", "B")
    }
    rows = []
    for domain, sides in panel.items():
        for side in ("A", "B"):
            for stem in sides[side]:
                rows.append(by_key[(domain, side, stem)])
    return rows


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Unbiased energy distance between two samples (flattened feature vectors)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    def d(x, y):
        return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)

    aa = d(a, a)
    bb = d(b, b)
    ab = d(a, b)
    n, m = a.shape[0], b.shape[0]
    if n < 2 or m < 2:
        return float(np.mean(ab))
    term1 = 2.0 * np.mean(ab)
    term2 = np.sum(aa - np.diag(np.diag(aa))) / (n * (n - 1))
    term3 = np.sum(bb - np.diag(np.diag(bb))) / (m * (m - 1))
    return float(max(0.0, term1 - term2 - term3))


def repeat_floor(distances: list[float], *, quantile: float = 0.99) -> float:
    arr = np.asarray([d for d in distances if np.isfinite(d)], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, quantile))


def stable_manifest_hash(rows: list[dict]) -> str:
    text = "\n".join(
        f"{r['domain']}\t{r['side'] if 'side' in r else r['role']}\t{r['stem']}"
        for r in sorted(rows, key=lambda r: (r["domain"], r.get("side", r.get("role")), r["stem"]))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
