"""Target-blind intervention controllers and clustered bootstrap primitives.

Controllers never read paired targets.  They consume pre-computed, unpaired
diagnostic statistics, apply the frozen decision rules from the task prompt,
and expose a serializable state so a resumed run replays bitwise-identical
decisions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Clustered bootstrap
# ---------------------------------------------------------------------------


def controller_bootstrap_seed(run_id: str, method: str, epoch: int, statistic: str) -> int:
    text = "\x1f".join([str(run_id), str(method), str(int(epoch)), str(statistic)]).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def cluster_bootstrap_draws(
    clusters_by_domain: dict[str, list[list[float]]],
    *,
    statistic: str,
    n_draws: int = 999,
    seed: int = 0,
) -> list[float]:
    """Six-domain equal-weight clustered bootstrap.

    ``clusters_by_domain`` maps domain -> list of source clusters, where each
    source cluster is a list of raw unit values.  A draw resamples source
    clusters within each domain, computes the per-domain statistic, then takes
    the equal-weight average of the six domain statistics.
    """
    stat_fn = _median if statistic == "median" else _mean
    domains = sorted(clusters_by_domain)
    if not domains:
        return []
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_draws)):
        per_domain = []
        for domain in domains:
            clusters = clusters_by_domain[domain]
            n = len(clusters)
            if n == 0:
                per_domain.append(float("nan"))
                continue
            idx = rng.integers(0, n, size=n)
            units = [v for i in idx for v in clusters[int(i)]]
            per_domain.append(stat_fn(units) if units else float("nan"))
        draws.append(float(np.nanmean(np.asarray(per_domain, dtype=np.float64))))
    return draws


def point_estimate(clusters_by_domain: dict[str, list[list[float]]], statistic: str) -> float:
    per_domain = []
    for domain in sorted(clusters_by_domain):
        units = [v for c in clusters_by_domain[domain] for v in c]
        if not units:
            continue
        per_domain.append(_median(units) if statistic == "median" else _mean(units))
    if not per_domain:
        return float("nan")
    return float(np.mean(np.asarray(per_domain, dtype=np.float64)))


def lower_bound(draws: Sequence[float], alpha: float = 0.05) -> float:
    arr = np.asarray([d for d in draws if math.isfinite(d)], dtype=np.float64)
    return float(np.quantile(arr, alpha)) if arr.size else float("nan")


def upper_bound(draws: Sequence[float], alpha: float = 0.05) -> float:
    arr = np.asarray([d for d in draws if math.isfinite(d)], dtype=np.float64)
    return float(np.quantile(arr, 1.0 - alpha)) if arr.size else float("nan")


def statistic_record(
    run_id: str,
    method: str,
    epoch: int,
    clusters_by_domain: dict[str, list[list[float]]],
    *,
    statistic: str,
    n_draws: int = 999,
) -> dict:
    """Return a serializable statistic bundle with point estimate and CI."""
    seed = controller_bootstrap_seed(run_id, method, epoch, statistic)
    draws = cluster_bootstrap_draws(
        clusters_by_domain, statistic=statistic, n_draws=n_draws, seed=seed
    )
    return {
        "statistic": statistic,
        "point": point_estimate(clusters_by_domain, statistic),
        "lower": lower_bound(draws),
        "upper": upper_bound(draws),
        "n_draws": n_draws,
        "bootstrap_seed": seed,
        "draws": draws,
        "raw_clusters": clusters_by_domain,
    }


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------


@dataclass
class AuditRecord:
    """One epoch's target-blind audit with all frozen statistics."""

    method: str
    epoch: int
    statistics: dict = field(default_factory=dict)
    valid: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "epoch": self.epoch,
            "statistics": self.statistics,
            "valid": self.valid,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AuditRecord":
        return cls(
            method=d["method"],
            epoch=int(d["epoch"]),
            statistics=d.get("statistics", {}),
            valid=bool(d.get("valid", True)),
            reason=d.get("reason", ""),
        )


def _stat(audit: AuditRecord, name: str, key: str, default: float = math.nan) -> float:
    s = audit.statistics.get(name)
    if not isinstance(s, dict):
        return default
    v = s.get(key)
    return default if v is None else float(v)


# ---------------------------------------------------------------------------
# Controller base and rule engines
# ---------------------------------------------------------------------------


@dataclass
class ControllerState:
    status: str = "ACTIVE"  # ACTIVE | HANDOFF | OFF
    reason: str = ""
    frozen_epoch: int | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "frozen_epoch": self.frozen_epoch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControllerState":
        return cls(
            status=str(d.get("status", "ACTIVE")),
            reason=str(d.get("reason", "")),
            frozen_epoch=None if d.get("frozen_epoch") is None else int(d["frozen_epoch"]),
        )


class Controller:
    method = ""
    # The minimum epoch whose audits may enter consecutive counting.
    countable_min_epoch = 1
    # Number of consecutive same-trigger audits required.
    consecutive_required = 2

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.state = ControllerState()
        self.history: list[AuditRecord] = []

    def state_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "state": self.state.to_dict(),
            "history": [a.to_dict() for a in self.history],
        }

    def load_state_dict(self, d: dict) -> None:
        self.run_id = str(d.get("run_id", self.run_id))
        self.state = ControllerState.from_dict(d.get("state", {}))
        self.history = [AuditRecord.from_dict(a) for a in d.get("history", [])]

    def record(self, audit: AuditRecord) -> None:
        self.history.append(audit)
        if self.state.status in ("OFF", "HANDOFF"):
            return
        status, reason, frozen_epoch = self.decide(self.history)
        if status != "ACTIVE":
            self.state.status = status
            self.state.reason = reason
            self.state.frozen_epoch = frozen_epoch
        elif reason:
            self.state.reason = reason

    def decide(self, history: Sequence[AuditRecord]) -> tuple[str, str, int | None]:
        raise NotImplementedError


def _consecutive(audits: Sequence[AuditRecord], predicate, n: int) -> bool:
    count = 0
    for a in reversed(list(audits)):
        if predicate(a):
            count += 1
            if count >= n:
                return True
        else:
            count = 0
    return False


class DTController(Controller):
    method = "DT"
    countable_min_epoch = 26  # physical e26 == active age 6

    def decide(self, history: Sequence[AuditRecord]) -> tuple[str, str, int | None]:
        valid = [a for a in history if a.valid]
        countable = [a for a in valid if a.epoch >= self.countable_min_epoch]
        if len(countable) < 2:
            return ("ACTIVE", "", None)

        if _consecutive(
            countable,
            lambda a: _stat(a, "E_DT", "upper", math.inf) <= 0.0,
            2,
        ):
            return ("OFF", "DT_SIGNAL_EXHAUSTED", None)
        if _consecutive(
            countable,
            lambda a: _stat(a, "R_DT", "lower", math.inf) <= 0.0,
            3,
        ):
            return ("OFF", "DT_NO_TARGET_BLIND_RESPONSE", None)
        if any(not a.valid for a in countable):
            return ("OFF", "DT_ENGINEERING_LANE_STOP", None)
        return ("ACTIVE", "", None)


class HJController(Controller):
    method = "HJ"
    countable_min_epoch = 20

    def decide(self, history: Sequence[AuditRecord]) -> tuple[str, str, int | None]:
        valid = [a for a in history if a.valid]
        countable = [a for a in valid if a.epoch >= self.countable_min_epoch]
        if len(countable) < 2:
            return ("ACTIVE", "", None)
        if _consecutive(countable, lambda a: not a.valid, 2):
            return ("OFF", "HJ_SIGNAL_NOT_ALIVE", None)
        return ("ACTIVE", "", None)


class HNEKController(Controller):
    method = "HNEK"
    countable_min_epoch = 30

    def decide(self, history: Sequence[AuditRecord]) -> tuple[str, str, int | None]:
        valid = [a for a in history if a.valid]
        countable = [a for a in valid if a.epoch >= self.countable_min_epoch]
        if len(countable) < 2:
            return ("ACTIVE", "", None)

        if _consecutive(
            countable,
            lambda a: _stat(a, "C_H", "upper", math.inf) <= 0.0,
            2,
        ):
            return ("HANDOFF", "HNEK_SIGNAL_EXHAUSTED", countable[-1].epoch)
        if _consecutive(
            countable,
            lambda a: _stat(a, "B_H", "lower", -math.inf) <= 0.0,
            2,
        ):
            return ("HANDOFF", "HNEK_NO_LONGER_BEATS_PLAIN_UPDATE", countable[-1].epoch)
        if _consecutive(
            countable,
            lambda a: bool(a.statistics.get("safety_lost", {}).get("point", False)),
            2,
        ):
            return ("HANDOFF", "HNEK_NATIVE_SAFETY_LOST", countable[-1].epoch)
        if any(not a.valid for a in countable):
            return ("HANDOFF", "HNEK_ENGINEERING_LANE_STOP", countable[-1].epoch)
        return ("ACTIVE", "", None)


def make_controller(method: str, run_id: str) -> Controller:
    if method == "DT":
        return DTController(run_id)
    if method == "HJ":
        return HJController(run_id)
    if method == "HNEK":
        return HNEKController(run_id)
    raise ValueError(f"unknown method: {method}")
