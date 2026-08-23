import json

import pytest

from clean_reexploration.access_guard import TargetAccessGuard


def _training():
    return [
        {"domain": "d", "side": "A", "stem": "0001", "absolute_path": "/root/d/input/0001.png"},
        {"domain": "d", "side": "B", "stem": "0002", "absolute_path": "/root/d/input/0002.png"},
    ]


def _paired():
    return [
        {"domain": "d", "role": "T3_A", "stem": "0001", "absolute_path": "/root/d/input/0001.png"},
        {"domain": "d", "role": "T3_A_TARGET", "stem": "0001", "absolute_path": "/root/d/target/0001.png"},
    ]


def test_guard_allows_unpaired_a():
    g = TargetAccessGuard(
        training_manifest=_training(),
        paired_manifest=_paired(),
        ledger_path="/tmp/guard_ledger.csv",
        data_root="/root",
        frozen_ok_path="/tmp/no_freeze.ok",
    )
    p = g.request("/root/d/input/0001.png", role="A", purpose="training")
    assert p.endswith("0001.png")


def test_guard_rejects_target():
    g = TargetAccessGuard(
        training_manifest=_training(),
        paired_manifest=_paired(),
        ledger_path="/tmp/guard_ledger2.csv",
        data_root="/root",
        frozen_ok_path="/tmp/no_freeze2.ok",
    )
    with pytest.raises(PermissionError):
        g.request("/root/d/target/0001.png", role="sealed", purpose="probe")
    assert g.target_read_count() == 1
