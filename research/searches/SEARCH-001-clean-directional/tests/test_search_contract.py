from __future__ import annotations

import random
from types import SimpleNamespace

from src.evaluate import select_discovery_rows
from src.protocol import LaneSpec, frozen_lanes, synthesize
from src.runtime import install_import_paths


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
