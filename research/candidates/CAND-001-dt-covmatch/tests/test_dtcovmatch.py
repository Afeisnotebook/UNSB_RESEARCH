import math

import pytest
import torch
import torch.nn as nn

from dtcov.dtcovmatch import (
    DTCovMatch,
    DTCovMatchConfig,
    DomainTimeStats,
    compute_direction_statistics,
    domain_key_from_path,
    scheduled_lambda,
    time_norm_from_times,
)


class _FakeG(nn.Module):
    def __init__(self, latent_dim, c=3, h=8, w=8):
        super().__init__()
        self.proj = nn.Linear(latent_dim, c * h * w)
        self.h = h
        self.w = w

    def forward(self, X_t, time_idx, z):
        out = self.proj(z).reshape(z.size(0), 3, self.h, self.w)
        return X_t + 0.1 * out


def test_scheduled_lambda_ramp_hold_cosine_decay():
    base = 0.001
    kwargs = dict(
        ramp_start=1,
        ramp_end=5,
        decay_start=15,
        decay_end=25,
        min_value=0.0,
    )
    values = {
        0: 0.0,
        1: 0.0002,
        5: 0.001,
        15: 0.001,
        20: 0.0005,
        25: 0.0,
        30: 0.0,
    }
    for epoch, expected in values.items():
        got = scheduled_lambda(base, epoch, "ramp_hold_cosine_decay", **kwargs)
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_scheduled_lambda_fixed_and_cosine():
    assert scheduled_lambda(0.001, 7, "fixed") == pytest.approx(0.001)
    got = scheduled_lambda(0.001, 15, "cosine_decay", decay_start=10, decay_end=20)
    progress = 0.5
    expected = 0.001 * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert got == pytest.approx(expected)


def test_compute_direction_statistics_matches_reference():
    torch.manual_seed(0)
    X_t = torch.randn(2, 3, 8, 8)
    endpoint_samples = X_t.unsqueeze(0) + 0.2 * torch.randn(4, 2, 3, 8, 8)
    t_norm = 0.74
    stats = compute_direction_statistics(
        X_t=X_t,
        endpoint_samples=endpoint_samples,
        t_norm=t_norm,
        region_patch=4,
        detach_uncertainty=False,
    )

    denom = max(1.0 - t_norm, 1e-6)
    directions = (endpoint_samples - X_t.unsqueeze(0)) / denom
    v_bar = directions.mean(dim=0)
    centered = directions - v_bar.unsqueeze(0)
    var_dir = (centered * centered).sum(dim=0) / 3.0
    U_pix = var_dir.mean(dim=1, keepdim=True)
    signal = (v_bar * v_bar).mean(dim=1, keepdim=True)
    U_reg = torch.nn.functional.adaptive_avg_pool2d(U_pix, (2, 2))
    signal_reg = torch.nn.functional.adaptive_avg_pool2d(signal, (2, 2))
    expected = U_reg / (signal_reg + 1e-6)

    assert stats.U_reg_norm.shape == (2, 1, 2, 2)
    assert torch.allclose(stats.U_reg_norm, expected, atol=1e-6)


def test_compute_direction_statistics_signal_norm_off_uses_raw_u():
    torch.manual_seed(0)
    X_t = torch.randn(2, 3, 8, 8)
    endpoint_samples = X_t.unsqueeze(0) + 0.2 * torch.randn(4, 2, 3, 8, 8)
    stats = compute_direction_statistics(
        X_t=X_t,
        endpoint_samples=endpoint_samples,
        t_norm=0.74,
        region_patch=4,
        detach_uncertainty=False,
        signal_normalize=False,
    )
    assert torch.allclose(stats.U_reg_norm, stats.U_reg, atol=1e-6)


def test_domain_time_stats_ema_and_unknown():
    stats = DomainTimeStats(eps=1e-2, momentum=0.5)
    log_teacher = torch.tensor([[[[1.0]]], [[[2.0]]]])
    domain_keys = ["a", "b"]

    mu, sigma, known = stats.mu_sigma(log_teacher, domain_keys, 3)
    assert known == 0.0
    assert torch.allclose(mu, torch.zeros_like(mu))
    assert torch.allclose(sigma, torch.ones_like(sigma))

    batch = stats.batch_stats(log_teacher, domain_keys, 3)
    assert batch[("a", 3)][0] == pytest.approx(1.0)
    assert batch[("b", 3)][0] == pytest.approx(2.0)
    stats.update(batch)

    mu, sigma, known = stats.mu_sigma(log_teacher, domain_keys, 3)
    assert known == 1.0
    assert mu[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert mu[1, 0, 0, 0].item() == pytest.approx(2.0)


def test_domain_time_stats_global_mode_collapses_keys():
    stats = DomainTimeStats(eps=1e-2, momentum=0.5, norm_mode="global")
    log_teacher = torch.tensor([[[[1.0]]], [[[2.0]]]])
    domain_keys = ["a", "b"]

    batch = stats.batch_stats(log_teacher, domain_keys, 3)
    assert set(batch.keys()) == {("__global__", 0)}
    assert batch[("__global__", 0)][0] == pytest.approx(1.5)

    stats.update(batch)
    mu, sigma, known = stats.mu_sigma(log_teacher, domain_keys, 3)
    assert known == 1.0
    assert mu[0, 0, 0, 0].item() == pytest.approx(1.5)
    assert mu[1, 0, 0, 0].item() == pytest.approx(1.5)


def test_dtcovmatch_self_teacher_does_not_snapshot():
    torch.manual_seed(0)
    net = _FakeG(latent_dim=4)
    config = DTCovMatchConfig(
        lambda_value=0.001,
        warmup_iters=0,
        latent_dim=4,
        freeze_teacher=False,
    )
    alg = DTCovMatch(netG=net, config=config)
    X_t = torch.randn(2, 3, 8, 8)
    loss, diag = alg.forward(
        X_t=X_t,
        time_idx=torch.tensor([2]),
        time_id=2,
        t_norm=0.5,
        domain_keys=["a", "b"],
    )
    assert loss.isfinite()
    assert alg.teacher is None


def test_dtcovmatch_eval_off_returns_zero():
    net = _FakeG(latent_dim=4)
    config = DTCovMatchConfig(lambda_value=0.0, warmup_iters=0, latent_dim=4)
    alg = DTCovMatch(netG=net, config=config)
    X_t = torch.randn(2, 3, 8, 8)
    loss, diag = alg.forward(
        X_t=X_t,
        time_idx=torch.tensor([2]),
        time_id=2,
        t_norm=0.5,
        domain_keys=["a", "b"],
    )
    assert loss.dim() == 0
    assert loss.item() == 0.0
    assert alg.teacher is None


def test_dtcovmatch_enabled_forward_and_backward():
    torch.manual_seed(0)
    net = _FakeG(latent_dim=8)
    config = DTCovMatchConfig(
        m=3,
        region_patch=4,
        lambda_value=0.001,
        warmup_iters=0,
        latent_dim=8,
    )
    alg = DTCovMatch(netG=net, config=config)
    alg.ensure_teacher()
    with torch.no_grad():
        alg.teacher.proj.weight.copy_(alg.teacher.proj.weight + 0.1)
    X_t = torch.randn(2, 3, 8, 8)
    loss, diag = alg.forward(
        X_t=X_t,
        time_idx=torch.tensor([2]),
        time_id=2,
        t_norm=0.74,
        domain_keys=["a", "b"],
    )
    assert loss.isfinite()
    assert loss.item() > 0.0
    assert diag["group_count"] == 2
    loss.backward()
    assert net.proj.weight.grad is not None
    assert net.proj.weight.grad.abs().sum().item() > 0.0


def test_injected_teacher_is_exact_frozen_canonical_state():
    torch.manual_seed(7)
    live = _FakeG(latent_dim=4)
    canonical = _FakeG(latent_dim=4)
    canonical_state = {
        key: value.detach().clone() for key, value in canonical.state_dict().items()
    }
    alg = DTCovMatch(
        netG=live,
        config=DTCovMatchConfig(lambda_value=0.001, latent_dim=4),
    )

    teacher = alg.inject_teacher(canonical_state)
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, canonical_state[key])
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert len(alg._teacher_netG_sha256) == 64

    with torch.no_grad():
        live.proj.weight.add_(1.0)
    assert torch.equal(teacher.proj.weight, canonical_state["proj.weight"])


def test_domain_key_and_time_norm():
    assert domain_key_from_path("/data/RainDS-syn__0001.png") == "rainds-syn"
    assert domain_key_from_path("/data/FoggyCityscapes__0001.png") == "foggycityscapes"
    times = torch.tensor([0.0, 0.5, 0.75, 0.875, 0.9375, 1.0])
    assert time_norm_from_times(times, torch.tensor([2]), 5) == pytest.approx(0.75)
