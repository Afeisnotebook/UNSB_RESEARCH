from __future__ import annotations

import random
from types import SimpleNamespace

import torch

from src.evaluate import select_discovery_rows
from src.protocol import LaneSpec, frozen_lanes, synthesize
from src.runtime import install_import_paths, option_args


def test_frozen_lane_pool_is_exact():
    assert [lane.name for lane in frozen_lanes()] == [
        "plain", "dt_anchor", "hj_anchor", "hnek_anchor",
        "lbst", "ptq", "dcum", "aeb",
    ]


def test_ptq_has_exact_fifty_step_mass_and_seeded_order():
    install_import_paths()
    from models.sb_model import SBModel

    model = object.__new__(SBModel)
    model.opt = SimpleNamespace(search_ptq_seed=2026)
    first = [model._ptq_index(step, 5) for step in range(50)]
    second = [model._ptq_index(step, 5) for step in range(50)]
    assert first == second
    assert [first.count(index) for index in range(5)] == [25, 12, 6, 4, 3]


def test_lbst_teacher_can_be_constructed_and_is_frozen():
    install_import_paths()
    from models.sb_model import SBModel

    model = object.__new__(SBModel)
    model.opt = SimpleNamespace(search_lbst=True)
    model.device = torch.device("cpu")
    model.netG = torch.nn.Linear(3, 2)
    model._lbst_netG = None
    teacher = model._ensure_lbst_teacher()
    assert teacher is not model.netG
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.netG.parameters(), teacher.parameters())
    )


def test_dcum_never_crosses_domain_or_reuses_stem():
    install_import_paths()
    from data.unaligned_dataset import UnalignedDataset

    dataset = object.__new__(UnalignedDataset)
    dataset._B_by_domain = {
        "Fog": [r"x\Fog__001.png", r"x\Fog__002.png", r"x\Fog__003.png"]
    }
    random.seed(2026)
    for _ in range(20):
        selected = dataset._sample_dcum_B(r"x\Fog__001.png")
        domain, stem = dataset._domain_and_stem(selected)
        assert domain == "Fog"
        assert stem != "001"


def test_discovery_selector_cannot_select_confirmation():
    rows = [
        {"domain": "D", "split": "discovery", "order": str(index), "stem": str(index)}
        for index in range(3)
    ] + [
        {"domain": "D", "split": "confirmation", "order": "3", "stem": "sealed"}
    ]
    assert [row["stem"] for row in select_discovery_rows(
        rows, start_per_domain=1, count_per_domain=2
    )] == ["1", "2"]


def test_synthesis_preserves_single_legacy_owner():
    legacy = LaneSpec("legacy", model="hj", family="legacy")
    new = LaneSpec("new", mechanisms=("aeb",), family="new", estimated_g_flops_multiplier=2)
    combined = synthesize("combined", legacy, new)
    assert combined.model == "hj"
    assert combined.mechanisms == ("aeb",)
    assert combined.estimated_g_flops_multiplier == 2


def test_finite_hj_window_scales_with_data_exposure_not_total_budget(tmp_path):
    spec = LaneSpec("hj_finite", model="hj", family="dthj_derived")
    values = option_args(
        spec,
        dataroot=tmp_path,
        checkpoint_dir=tmp_path,
        steps_per_epoch=600,
        total_steps=120000,
        seed=2026,
        gpu=0,
    )
    start = values[values.index("--hj_search_start_step") + 1]
    duration = values[values.index("--hj_search_duration_steps") + 1]
    assert start == "960"
    assert duration == "3840"
