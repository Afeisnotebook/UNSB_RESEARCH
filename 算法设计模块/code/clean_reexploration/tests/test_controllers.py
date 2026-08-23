import math

import pytest

from clean_reexploration.controllers import (
    AuditRecord,
    DTController,
    HJController,
    HNEKController,
    cluster_bootstrap_draws,
    point_estimate,
)


def test_cluster_bootstrap_equal_weight_deterministic():
    clusters = {
        "a": [[1.0, 1.0, 1.0]],
        "b": [[3.0, 3.0, 3.0]],
    }
    d1 = cluster_bootstrap_draws(clusters, statistic="mean", n_draws=20, seed=7)
    d2 = cluster_bootstrap_draws(clusters, statistic="mean", n_draws=20, seed=7)
    assert d1 == d2
    # equal weight across the two domains -> all draws equal to 2.0
    assert all(abs(x - 2.0) < 1e-9 for x in d1)


def test_controller_bootstrap_seed_is_63_bit():
    from clean_reexploration.controllers import controller_bootstrap_seed

    s = controller_bootstrap_seed("r", "DT", 22, "E_DT")
    assert 0 <= s < 2**63


def test_dt_off_signal_exhausted():
    c = DTController("r")
    for epoch in (26, 27):
        a = AuditRecord(
            method="DT",
            epoch=epoch,
            statistics={"E_DT": {"upper": -0.1, "lower": -0.2}, "R_DT": {"lower": 1.0}},
        )
        c.record(a)
    assert c.state.status == "OFF"
    assert c.state.reason == "DT_SIGNAL_EXHAUSTED"


def test_hnek_handoff_after_two_consecutive():
    c = HNEKController("r")
    for epoch in (30, 31):
        a = AuditRecord(
            method="HNEK",
            epoch=epoch,
            statistics={"C_H": {"upper": -0.5}, "B_H": {"lower": 0.1}},
        )
        c.record(a)
    assert c.state.status == "HANDOFF"
    assert c.state.reason == "HNEK_SIGNAL_EXHAUSTED"
    assert c.state.frozen_epoch == 31
