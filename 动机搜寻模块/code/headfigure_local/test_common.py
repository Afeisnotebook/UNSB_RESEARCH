from __future__ import annotations

import numpy as np

from .common import nested_domain_image_bootstrap, spherical_dispersion, stable_seed


def test_spherical_dispersion_identical_is_zero():
    x = np.zeros((8, 16), dtype=np.float64)
    x[:, 0] = 1.0
    result = spherical_dispersion(x)
    assert result["D_sph"] == 0.0
    assert np.isclose(result["R2"], 1.0)


def test_spherical_dispersion_orthogonal_pair():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = spherical_dispersion(x)
    assert np.isclose(result["D_sph"], 1.0)
    assert np.isclose(result["mean_pair_angle_deg"], 90.0)


def test_stable_seed_identity():
    assert stable_seed("a", 1, "b") == stable_seed("a", 1, "b")
    assert stable_seed("a", 1, "b") != stable_seed("a", 2, "b")


def test_nested_bootstrap_positive():
    rows = []
    domains = ["a", "b", "c"]
    for domain_idx, domain in enumerate(domains):
        for image in range(10):
            rows.append({"domain": domain, "effect": 1.0 + 0.01 * domain_idx + 0.001 * image})
    summary, draws = nested_domain_image_bootstrap(
        rows, "effect", domains=domains, draws=1000, seed=7
    )
    assert summary["ci_low"] > 0
    assert summary["positive_domain_count"] == 3
    assert draws.shape == (1000,)
