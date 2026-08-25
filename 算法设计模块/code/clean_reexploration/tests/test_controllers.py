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


def test_dt_invalid_not_reachable_only_before_countable_epoch():
    c = DTController("r")
    a = AuditRecord(method="DT", epoch=25, valid=False, reason="engineering")
    c.record(a)
    assert c.state.status == "ACTIVE"
    assert len(c.history) == 1

    c.record(AuditRecord(method="DT", epoch=26, valid=False, reason="engineering"))
    assert c.state.status == "OFF"
    assert c.state.reason == "DT_ENGINEERING_LANE_STOP"
    assert len(c.history) == 2


def test_dt_no_target_blind_response_three_consecutive():
    c = DTController("r")
    for epoch in (26, 27, 28):
        c.record(
            AuditRecord(
                method="DT",
                epoch=epoch,
                statistics={"E_DT": {"upper": 1.0}, "R_DT": {"lower": -0.1}},
            )
        )
    assert c.state.status == "OFF"
    assert c.state.reason == "DT_NO_TARGET_BLIND_RESPONSE"


def test_hj_valid_invalid_valid_does_not_exit():
    c = HJController("r")
    for epoch, valid in ((20, True), (30, False), (40, True)):
        c.record(AuditRecord(method="HJ", epoch=epoch, valid=valid))
    assert c.state.status == "ACTIVE"


def test_hj_two_invalid_off_only_after_countable_epochs():
    c = HJController("r")
    for epoch, valid in ((10, False), (20, False), (30, False)):
        c.record(AuditRecord(method="HJ", epoch=epoch, valid=valid))
    assert c.state.status == "OFF"
    assert c.state.reason == "HJ_SIGNAL_NOT_ALIVE"


def test_hnek_safety_and_engineering_invalid():
    c = HNEKController("r")
    for epoch in (30, 31):
        c.record(
            AuditRecord(
                method="HNEK",
                epoch=epoch,
                statistics={"safety_lost": {"point": True}},
            )
        )
    assert c.state.status == "HANDOFF"
    assert c.state.reason == "HNEK_NATIVE_SAFETY_LOST"

    c2 = HNEKController("r")
    c2.record(AuditRecord(method="HNEK", epoch=30, valid=False))
    assert c2.state.status == "HANDOFF"
    assert c2.state.reason == "HNEK_ENGINEERING_LANE_STOP"


def test_state_roundtrip_and_next_decision_equal():
    c = DTController("r")
    for epoch in (26, 27):
        c.record(
            AuditRecord(
                method="DT",
                epoch=epoch,
                statistics={"E_DT": {"upper": -0.2}},
            )
        )
    saved = c.state_dict()
    restored = DTController("r")
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    nxt = AuditRecord(
        method="DT",
        epoch=28,
        statistics={"E_DT": {"upper": -0.2}},
    )
    c.record(nxt)
    restored.record(nxt)
    assert c.state_dict() == restored.state_dict()
