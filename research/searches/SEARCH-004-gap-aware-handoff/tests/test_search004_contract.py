from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze import trapezoid  # noqa: E402
from src.interfaces import IdentityHandoff, StateObservation  # noqa: E402
from src.protocol import ARMS, Search004Protocol, assert_target_blind  # noqa: E402
from src.state import (  # noqa: E402
    ComponentMask,
    exact_equal,
    export_named_optimizers,
    load_named_optimizers,
)
from src.statistics import empirical_bernstein_cs, persistently_incompatible  # noqa: E402
from src.transports import least_change_native_moment_projection  # noqa: E402


class DummyModel:
    def __init__(self):
        self.netG = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        self.netF = torch.nn.Linear(2, 2)
        self.netD = torch.nn.Linear(2, 1)
        self.netE = torch.nn.Linear(2, 1)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1e-4)
        self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=1e-4)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=1e-4)
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=1e-4)


def populate(model: DummyModel) -> None:
    value = model.netG(torch.ones(1, 3)).sum()
    value = value + model.netF(torch.ones(1, 2)).sum()
    value = value + model.netD(torch.ones(1, 2)).sum()
    value = value + model.netE(torch.ones(1, 2)).sum()
    value.backward()
    for optimizer in (
        model.optimizer_G, model.optimizer_F, model.optimizer_D, model.optimizer_E,
    ):
        optimizer.step()


def test_protocol_is_frozen_and_has_all_core_arms():
    protocol = Search004Protocol()
    assert protocol.seed == 2026
    assert protocol.local_horizons == (1, 8, 32, 200)
    assert protocol.confirmation20_opened is False
    assert {
        "P_common_plain", "U_uninterrupted", "A_hard_disable",
        "B_gf_zero_moment", "C_local_native_moment", "D0_hold_only",
        "D_costate_equilibration", "E_combined", "F_g_only_transplant",
        "H_native_moment_projection",
        "K_gf_state_transplant",
    } == set(ARMS)


@pytest.mark.parametrize("field", ["psnr", "ssim", "lpips", "paired_target", "confirmation"])
def test_target_blind_schema_rejects_paired_fields(field):
    with pytest.raises(ValueError):
        assert_target_blind({field: 1})


def test_state_observation_rejects_target_aware_fields():
    with pytest.raises(ValueError):
        StateObservation({"paired_psnr": 1.0})


def test_named_optimizer_roundtrip_ignores_dictionary_order():
    model = DummyModel()
    populate(model)
    before = export_named_optimizers(model)
    reordered = {
        key: {
            "groups": value["groups"],
            "states": dict(reversed(list(value["states"].items()))),
        }
        for key, value in before.items()
    }
    for optimizer in (
        model.optimizer_G, model.optimizer_F, model.optimizer_D, model.optimizer_E,
    ):
        optimizer.state.clear()
    load_named_optimizers(model, reordered)
    after = export_named_optimizers(model)
    assert exact_equal(before, after)[0]


def test_named_optimizer_rejects_shape_mismatch():
    model = DummyModel()
    populate(model)
    value = export_named_optimizers(model)
    first = next(iter(value["G"]["states"].values()))
    first["exp_avg"] = torch.zeros(999)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_named_optimizers(model, value, only=("G",))


def test_named_optimizer_rejects_dtype_mismatch():
    model = DummyModel()
    populate(model)
    value = export_named_optimizers(model)
    first = next(iter(value["G"]["states"].values()))
    first["exp_avg"] = first["exp_avg"].double()
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        load_named_optimizers(model, value, only=("G",))


def test_component_mask_rejects_unknown_network():
    with pytest.raises(ValueError):
        ComponentMask(networks=("unknown",))


def test_auc_is_preregistered_trapezoid():
    assert trapezoid([(0, 0.0), (32, 0.2), (200, 0.4)], 200) == pytest.approx(0.268)
    assert trapezoid([(32, 0.2), (200, 0.4)], 200) != trapezoid([(0, 0.0), (32, 0.2), (200, 0.4)], 200)


def test_confidence_sequence_needs_eight_target_blind_observations():
    assert not empirical_bernstein_cs([-1.0] * 7).valid
    assert empirical_bernstein_cs([-1.0] * 8).valid
    # The conservative anytime boundary must not claim incompatibility merely
    # because the first eight bounded observations share a sign.
    assert not persistently_incompatible([-1.0] * 8)
    assert persistently_incompatible([-1.0] * 256)


def test_native_moment_projection_is_identity_when_native_compatible():
    model = DummyModel()
    populate(model)
    gradients = {}
    for player in ("G", "F"):
        optimizer = getattr(model, f"optimizer_{player}")
        network = getattr(model, f"net{player}")
        for name, parameter in network.named_parameters():
            state = optimizer.state[parameter]
            gradients[f"{player}.{name}"] = (
                state["exp_avg"] / (state["exp_avg_sq"].sqrt() + 1e-8)
            ).detach().clone()
    before = export_named_optimizers(model)
    record = least_change_native_moment_projection(model, gradients)
    after = export_named_optimizers(model)
    assert record["identity"] is True
    assert exact_equal(before, after)[0]


def test_native_moment_projection_removes_only_conflicting_component():
    model = DummyModel()
    populate(model)
    gradients = {}
    for player in ("G", "F"):
        optimizer = getattr(model, f"optimizer_{player}")
        network = getattr(model, f"net{player}")
        for name, parameter in network.named_parameters():
            state = optimizer.state[parameter]
            gradients[f"{player}.{name}"] = -(
                state["exp_avg"] / (state["exp_avg_sq"].sqrt() + 1e-8)
            ).detach().clone()
    record = least_change_native_moment_projection(model, gradients)
    assert record["identity"] is False
    assert record["target_blind_defect_before"] > 0.0
    assert record["target_blind_defect_after"] == pytest.approx(0.0, abs=1e-6)
    assert all(
        row["after_dot"] >= -1e-6
        for row in record["players"].values()
        if row.get("available")
    )
