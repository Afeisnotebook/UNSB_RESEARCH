from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def dump_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bridge_times(num_timesteps: int = 5) -> np.ndarray:
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be >= 2")
    increments = np.array([0.0] + [1.0 / (i + 1) for i in range(num_timesteps - 1)])
    times = np.cumsum(increments)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    return np.concatenate([np.zeros(1), times])


def spherical_dispersion(unit_directions: np.ndarray) -> dict[str, float]:
    """Return exact pairwise spherical dispersion and related quantities.

    ``unit_directions`` has shape [M, D] and every row must have unit norm.
    D_sph is both the mean pairwise cosine distance and the sample covariance
    trace of the unit vectors.  It is bounded and has no small-denominator
    amplification.
    """
    x = np.asarray(unit_directions, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("expected [M,D] with M>=2")
    norms = np.linalg.norm(x, axis=1)
    if not np.all(np.isfinite(x)) or not np.allclose(norms, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("directions must be finite unit vectors")
    # The tensors arrive as float32-normalized rows.  Re-normalize in float64
    # before applying the exact pairwise identity; otherwise a nearly
    # identical direction panel can have R^2=1+O(1e-8) and a tiny negative
    # direct cosine distance solely from roundoff.
    x = x / norms[:, None]
    m = x.shape[0]
    mean = x.mean(axis=0)
    r2 = float(np.dot(mean, mean))
    d_sph = float(m / (m - 1.0) * max(0.0, 1.0 - r2))
    gram = x @ x.T
    iu = np.triu_indices(m, k=1)
    direct = max(0.0, float(np.mean(1.0 - gram[iu])))
    if not np.isclose(d_sph, direct, atol=2e-10, rtol=2e-8):
        raise AssertionError(f"pairwise identity mismatch: {d_sph} vs {direct}")
    mean_cos = float(np.clip(1.0 - d_sph, -1.0, 1.0))
    return {
        "D_sph": d_sph,
        "R2": r2,
        "legacy_U": float(d_sph / (r2 + 1e-8)),
        "mean_pair_cos": mean_cos,
        "mean_pair_angle_deg": float(np.degrees(np.arccos(mean_cos))),
    }


def nested_domain_image_bootstrap(
    rows: Iterable[dict],
    value_key: str,
    *,
    domains: list[str] | None = None,
    draws: int = 5000,
    seed: int = 20410824,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Equal-domain estimator with nested domain/image percentile bootstrap."""
    domains = domains or DOMAINS
    grouped: dict[str, np.ndarray] = {}
    rows = list(rows)
    for domain in domains:
        values = np.asarray(
            [float(r[value_key]) for r in rows if r["domain"] == domain], dtype=np.float64
        )
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"missing or non-finite values for {domain}")
        grouped[domain] = values

    domain_means = np.array([grouped[d].mean() for d in domains], dtype=np.float64)
    observed = float(domain_means.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        selected = rng.integers(0, len(domains), size=len(domains))
        means = []
        for idx in selected:
            values = grouped[domains[int(idx)]]
            image_idx = rng.integers(0, values.size, size=values.size)
            means.append(float(values[image_idx].mean()))
        boot[b] = float(np.mean(means))
    result = {
        "n_domains": len(domains),
        "n_images": int(sum(len(v) for v in grouped.values())),
        "mean": observed,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "positive_domain_count": int(np.sum(domain_means > 0)),
    }
    return result, boot
