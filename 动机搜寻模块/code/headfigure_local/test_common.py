from __future__ import annotations

import numpy as np

from .common import nested_domain_image_bootstrap, spherical_dispersion, stable_seed
from .measure import joint_pca_rows
from .measure_reciprocal import cosine_distance, mean_direction_and_floor, single_age_span
from .adjudicate_phase_confirmation import effective_ages, phase_variance


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


def test_float32_near_identical_panel_has_no_negative_roundoff():
    rng = np.random.default_rng(9)
    base = rng.normal(size=(64,)).astype(np.float32)
    base /= np.linalg.norm(base)
    x = np.stack([base + 1e-7 * rng.normal(size=64).astype(np.float32) for _ in range(32)])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    result = spherical_dispersion(x)
    assert result["D_sph"] >= 0
    assert np.isfinite(result["legacy_U"])


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


def test_joint_pca_uses_all_three_arms():
    rng = np.random.default_rng(11)
    groups = {}
    for index, arm in enumerate(("single_e1", "single_e5", "aio_e1")):
        x = rng.normal(size=(8, 32)) + index * 0.05
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        groups[arm] = x.astype(np.float32)
    rows = joint_pca_rows(groups, "FoggyCityscapes", "fixture")
    assert len(rows) == 24
    assert {row["arm"] for row in rows} == set(groups)
    assert all(np.isfinite(row["pca1"]) and np.isfinite(row["pca2"]) for row in rows)


def test_reciprocal_mean_direction_floor_identical_panel():
    import torch

    panel = torch.zeros(8, 3, 2, 2)
    panel[:, 0] = 1.0
    mean, floor = mean_direction_and_floor(panel)
    assert np.isclose(float(mean.norm()), 1.0)
    assert floor == 0.0


def test_reciprocal_cosine_distance_and_age_span():
    import torch

    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    z = torch.tensor([-1.0, 0.0, 0.0])
    assert np.isclose(cosine_distance(x, y), 1.0)
    assert np.isclose(cosine_distance(x, z), 2.0)
    assert np.isclose(single_age_span([x, y, z]), 2.0)


def test_phase_effective_age_and_variance():
    first = np.array([[3.0, 2.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 2.0, 3.0]])
    second = np.array([[3.0, 2.0, 2.0, 1.0, 3.0], [3.0, 2.0, 2.0, 1.0, 3.0]])
    ages = effective_ages([first, second])
    assert ages.tolist() == [3, 4]
    assert np.isclose(phase_variance(ages), 0.25)
