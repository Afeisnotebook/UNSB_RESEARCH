#!/usr/bin/env python3
"""CPU-only unit tests for the pure path-geometry primitives."""

from __future__ import annotations

import numpy as np
import torch

from measure_path_geometry import (
    bridge_times,
    joint_pca,
    paired_bootstrap,
    region_direction_dispersion,
    unit_direction_dispersion,
)


class FakeNetG(torch.nn.Module):
    """Tiny deterministic generator for rollout/proposal unit tests."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, bias=False)
        torch.nn.init.eye_(self.conv.weight[:, :, 0, 0])

    def forward(self, x, time_cond, z):
        return torch.tanh(self.conv(x) + 0.01 * time_cond.float().view(-1, 1, 1, 1))


def test_bridge_times_shape_and_ends():
    times = bridge_times(5)
    assert times.shape == (6,)
    assert abs(times[0]) < 1e-9
    assert abs(times[-1] - 1.0) < 1e-9
    # official grid
    expected = [0.0, 0.5, 0.74, 0.86, 0.94, 1.0]
    assert np.allclose(times, expected, atol=1e-2)


def test_unit_dispersion_zero_for_single_direction():
    base = torch.randn(3, 8, 8)
    flat = base.reshape(1, -1)
    flat = flat / flat.norm(dim=1, keepdim=True)
    d_norm = flat.reshape_as(base).unsqueeze(0).repeat(8, 1, 1, 1)
    u = unit_direction_dispersion(d_norm)
    assert u < 1e-6, f"expected ~0 dispersion, got {u}"


def test_unit_dispersion_increases_with_spread():
    a = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1).repeat(8, 1, 1, 1)
    b = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3, 1, 1).repeat(8, 1, 1, 1)
    a = a / a.norm(dim=1, keepdim=True)
    b = b / b.norm(dim=1, keepdim=True)
    spread = torch.cat([a, b], dim=0)  # 16 unit directions, two clusters
    u1 = unit_direction_dispersion(a)  # all same direction
    u2 = unit_direction_dispersion(spread)
    assert u2 > u1


def test_region_map_shape_and_positivity():
    d = torch.randn(64, 3, 128, 128)
    u_map = region_direction_dispersion(d, region_patch=32)
    assert u_map.shape == (4, 4)
    assert (u_map >= 0).all()


def test_joint_pca_returns_requested_components():
    mats = {m: [torch.randn(3, 4, 4)] for m in ["Single", "AIO"]}
    out = joint_pca(mats, n_components=2)
    assert out["proj"].shape[1] == 2
    assert out["proj"].shape[0] == 2


def test_paired_bootstrap_ci():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 50)
    b = rng.normal(0.0, 0.1, 50)
    res = paired_bootstrap(a, b)
    assert res["n"] == 50
    assert res["ci_low"] < res["mean"] < res["ci_high"]
    assert 0.5 < res["mean"] < 1.5


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
