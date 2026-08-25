from __future__ import annotations

import numpy as np

from phase_statistics import (
    bootstrap_phase_distributions,
    common_age,
    in_sample_sync_regret,
    two_fold_sync_regret,
)


def _repeated(profile: list[float], n: int = 20) -> np.ndarray:
    return np.repeat(np.asarray(profile, dtype=np.float64)[None, :], n, axis=0)


def test_common_age_and_zero_regret_when_synchronized() -> None:
    profiles = np.asarray(
        [
            [3.0, 1.0, 0.0, 2.0, 4.0],
            [6.0, 2.0, 0.0, 3.0, 8.0],
            [2.0, 0.5, 0.0, 1.0, 2.0],
        ]
    )
    assert common_age(profiles) == 2
    result = in_sample_sync_regret(profiles)
    assert result["regret"] == 0.0


def test_crossfit_regret_detects_stable_domain_phase_offsets() -> None:
    arrays = {
        "d1": _repeated([0.0, 2.0, 4.0, 6.0, 8.0]),
        "d2": _repeated([2.0, 0.0, 2.0, 4.0, 6.0]),
        "d3": _repeated([8.0, 6.0, 4.0, 2.0, 0.0]),
    }
    folds = {domain: (np.arange(10), np.arange(10, 20)) for domain in arrays}
    regret, per_domain, _ = two_fold_sync_regret(arrays, folds)
    assert regret > 0.5
    assert sum(value > 0 for value in per_domain.values()) >= 2


def test_crossfit_regret_is_zero_for_identical_clocks() -> None:
    arrays = {
        "d1": _repeated([3.0, 1.0, 0.0, 2.0, 4.0]),
        "d2": _repeated([7.0, 2.0, 0.0, 3.0, 9.0]),
        "d3": _repeated([2.0, 0.4, 0.0, 0.9, 2.0]),
    }
    folds = {domain: (np.arange(10), np.arange(10, 20)) for domain in arrays}
    regret, _, _ = two_fold_sync_regret(arrays, folds)
    assert regret == 0.0


def test_wasserstein_phase_energy_respects_phase_identity() -> None:
    synchronized = {
        "d1": _repeated([3.0, 1.0, 0.0, 2.0, 4.0]),
        "d2": _repeated([6.0, 2.0, 0.0, 3.0, 8.0]),
    }
    _, sync_summary = bootstrap_phase_distributions(synchronized, draws=100, seed=7)
    assert sync_summary["wasserstein_barycenter_energy"] == 0.0

    desynchronized = {
        "d1": _repeated([0.0, 2.0, 4.0, 6.0, 8.0]),
        "d2": _repeated([8.0, 6.0, 4.0, 2.0, 0.0]),
    }
    _, desync_summary = bootstrap_phase_distributions(desynchronized, draws=100, seed=7)
    assert desync_summary["wasserstein_barycenter_energy"] > 0.0
    assert desync_summary["phase_reliability_ratio"] == 1.0
