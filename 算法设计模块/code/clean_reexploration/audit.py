"""Target-blind, source-only controller audit statistics.

The functions in this module are called inside the training loop only at the
frozen audit cadence.  They do not consume paired targets, do not advance the
main training RNG/optimizer/scheduler/sampler, and return serializable raw
statistics that the controllers turn into ACTIVE/OFF/HANDOFF decisions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from clean_reexploration import controllers
from clean_reexploration.controller_audits import (
    compute_dt_logu,
    compute_hj_structure_loss,
    compute_hnek_c_h,
)


def _inner_netG(model):
    return model.netG.module if hasattr(model.netG, "module") else model.netG


def _bootstrap_stat(
    raw_clusters: dict,
    *,
    run_id: str,
    method: str,
    epoch: int,
    statistic: str,
    floor: float = 0.0,
) -> dict:
    seed = controllers.controller_bootstrap_seed(run_id, method, epoch, statistic)
    draws = controllers.cluster_bootstrap_draws(
        raw_clusters, statistic="mean", n_draws=999, seed=seed
    )
    point = controllers.point_estimate(raw_clusters, "mean")
    return {
        "point": point - floor,
        "point_unfloored": point,
        "lower": controllers.lower_bound(draws) - floor,
        "upper": controllers.upper_bound(draws) - floor,
        "draws": draws,
        "raw_clusters": raw_clusters,
        "floor": floor,
        "bootstrap_seed": seed,
    }


def _finite_stat(stat: dict) -> bool:
    return all(
        math.isfinite(float(stat.get(k, float("nan"))))
        for k in ("point", "lower", "upper")
    )


def compute_hnek_audit(
    model,
    panel_rows: list[dict],
    *,
    run_id: str,
    epoch: int,
    num_timesteps: int,
    tau: float,
    ngf: int,
) -> tuple[dict, bool, str]:
    rollout_seed = controllers.controller_bootstrap_seed(run_id, "HNEK", epoch, "C_H_rollout")
    sig = compute_hnek_c_h(
        _inner_netG(model),
        panel_rows,
        gamma=0.25,
        num_timesteps=num_timesteps,
        tau=tau,
        ngf=ngf,
        seed=rollout_seed,
    )
    floor = float(sig.get("repeat_floor", 0.0))
    c_h = _bootstrap_stat(
        sig["raw_clusters"],
        run_id=run_id,
        method="HNEK",
        epoch=epoch,
        statistic="C_H",
        floor=floor,
    )
    valid = _finite_stat(c_h)
    reason = "" if valid else "HNEK_C_H_NONFINITE"
    return {
        "C_H": c_h,
        "repeat_floor": floor,
        "repeat_estimates": sig.get("repeat_estimates", []),
    }, valid, reason


def _flatten_clusters(raw: dict) -> dict:
    """Convert a per-domain list-of-values signal into one-value clusters."""
    return {domain: [[float(v)]] for domain, vals in raw.items() for v in vals}


def _per_domain_mean_distance(lane: dict, teacher: dict) -> dict:
    out = {}
    for domain in sorted(set(lane) | set(teacher)):
        a = np.asarray(lane.get(domain, []), dtype=np.float64)
        b = np.asarray(teacher.get(domain, []), dtype=np.float64)
        if a.size == 0 or b.size == 0:
            out[domain] = [float("nan")]
            continue
        # Source-only statistic: median absolute difference between the lane and
        # frozen teacher logU realizations over the same domain.
        n = min(a.size, b.size)
        out[domain] = [float(np.median(np.abs(a[:n] - b[:n])))]
    return out


def compute_dt_audit(
    model,
    teacher_netG,
    panel_rows: list[dict],
    *,
    run_id: str,
    epoch: int,
    m: int,
    ngf: int,
    num_timesteps: int,
    tau: float,
) -> tuple[dict, bool, str]:
    rollout_seed = controllers.controller_bootstrap_seed(run_id, "DT", epoch, "E_DT_rollout")
    lane = compute_dt_logu(
        _inner_netG(model),
        panel_rows,
        m=m,
        ngf=ngf,
        num_timesteps=num_timesteps,
        tau=tau,
        seed=rollout_seed,
    )
    teacher = compute_dt_logu(
        teacher_netG,
        panel_rows,
        m=m,
        ngf=ngf,
        num_timesteps=num_timesteps,
        tau=tau,
        seed=rollout_seed,
    )
    raw = _per_domain_mean_distance(lane, teacher)
    stat = _bootstrap_stat(
        _flatten_clusters(raw),
        run_id=run_id,
        method="DT",
        epoch=epoch,
        statistic="E_DT",
        floor=0.0,
    )
    valid = _finite_stat(stat)
    reason = "" if valid else "DT_E_DT_NONFINITE"
    return {"E_DT": stat}, valid, reason


def compute_hj_audit(
    model,
    panel_rows: list[dict],
    *,
    run_id: str,
    epoch: int,
    ngf: int,
    num_timesteps: int,
    tau: float,
) -> tuple[dict, bool, str]:
    rollout_seed = controllers.controller_bootstrap_seed(run_id, "HJ", epoch, "V_HJ_rollout")
    raw = compute_hj_structure_loss(
        _inner_netG(model),
        panel_rows,
        ngf=ngf,
        num_timesteps=num_timesteps,
        tau=tau,
        seed=rollout_seed,
    )
    clusters = _flatten_clusters(raw)
    stat = _bootstrap_stat(
        clusters,
        run_id=run_id,
        method="HJ",
        epoch=epoch,
        statistic="V_HJ",
        floor=0.0,
    )
    valid = _finite_stat(stat)
    reason = "" if valid else "HJ_STRUCTURE_NONFINITE"
    return {"V_HJ": stat}, valid, reason
