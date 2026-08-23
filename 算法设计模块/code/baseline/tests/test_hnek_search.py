import torch

from models.hnek.hnek_search import (
    HnekSearchConfig,
    endpoint_from_residual_gamma,
    normalized_residual_gamma,
)


def test_gamma_roundtrip():
    x = torch.tensor([[1.0, 2.0], [0.5, -0.5]], requires_grad=True)
    r = torch.tensor([[0.2, -0.3], [0.4, 0.1]], requires_grad=True)
    h = torch.tensor([[1.0, 0.5], [0.25, 0.75]])
    for gamma in (0.25, 0.5, 0.75, 1.0):
        y = endpoint_from_residual_gamma(x, r, h, gamma=gamma)
        r2 = normalized_residual_gamma(x, y, h, gamma=gamma, eps=1e-8)
        assert torch.allclose(r, r2, atol=1e-6, rtol=1e-6)


def test_h1_and_h0_special_cases():
    x = torch.tensor([1.0, 2.0])
    r = torch.tensor([0.5, -0.5])
    y1 = endpoint_from_residual_gamma(x, r, torch.tensor([1.0, 1.0]), gamma=0.5)
    y0 = endpoint_from_residual_gamma(x, r, torch.tensor([0.0, 0.0]), gamma=0.5)
    assert torch.equal(y1, x + r)
    assert torch.equal(y0, x)


def test_config_defaults():
    cfg = HnekSearchConfig()
    assert cfg.gamma == 0.5
    assert cfg.coord == "residual"
    assert cfg.horizon_mode == "physical"
    assert cfg.partial == "all"
