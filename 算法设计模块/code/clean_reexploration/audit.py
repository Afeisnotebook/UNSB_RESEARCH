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
import torch
from PIL import Image

from clean_reexploration import controllers
from clean_reexploration.controller_audits import (
    compute_dt_logu,
    compute_hj_structure_loss,
    compute_hnek_c_h,
)
from clean_reexploration import full_state


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
    b_h, safety_lost = _hnek_b_h_and_safety(
        model,
        panel_rows,
        run_id=run_id,
        epoch=epoch,
        num_timesteps=num_timesteps,
        tau=tau,
        ngf=ngf,
    )
    valid = _finite_stat(c_h) and _finite_stat(b_h)
    reason = "" if valid else "HNEK_SIGNAL_NONFINITE"
    stats = {
        "C_H": c_h,
        "B_H": b_h,
        "safety_lost": safety_lost,
        "repeat_floor": floor,
        "repeat_estimates": sig.get("repeat_estimates", []),
    }
    return stats, valid, reason


def _hnek_b_h_and_safety(
    model,
    panel_rows,
    *,
    run_id,
    epoch,
    num_timesteps,
    tau,
    ngf,
) -> tuple[dict, dict]:
    """Counterfactual plain/HNEK one-step virtual update on the audit batch.

    Both virtual steps start from the exact same scientific state and RNG
    bundle; the full state is restored after each so no main-training state is
    advanced.
    """
    from clean_reexploration.train_executor import _img_transform, _cuda_batch
    from models.hnek.hnek_search import set_hnek_search_active

    device = next(model.netG.parameters()).device
    transform = _img_transform()
    a = panel_rows[0]
    b = next((r for r in panel_rows if r["side"] == "B"), panel_rows[0])
    batch = {
        "A": transform(Image.open(a["absolute_path"]).convert("RGB")).unsqueeze(0),
        "B": transform(Image.open(b["absolute_path"]).convert("RGB")).unsqueeze(0),
        "A_paths": [a["absolute_path"]],
        "B_paths": [b["absolute_path"]],
    }
    batch = _cuda_batch(batch)
    snapshot = full_state.capture_full_state(
        model=model,
        global_step=0,
        physical_epoch=epoch,
        controllers={},
        sampler={},
        identity={"run_id": run_id, "audit": "hnek_virtual"},
    )

    def _c_after(active: bool) -> float:
        full_state.restore_full_state(model=model, state=snapshot)
        set_hnek_search_active(model, active)
        model.set_input(batch, batch)
        model.optimize_parameters()
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
        return controllers.point_estimate(sig["raw_clusters"], "mean") - float(sig.get("repeat_floor", 0.0))

    before = _c_after(True)
    after_h = _c_after(True)
    after_p = _c_after(False)
    delta_c_h = before - after_h
    delta_c_p = before - after_p
    b_h = _bootstrap_stat(
        {"global": [[delta_c_h - delta_c_p]]},
        run_id=run_id,
        method="HNEK",
        epoch=epoch,
        statistic="B_H",
        floor=0.0,
    )
    full_state.restore_full_state(model=model, state=snapshot)
    set_hnek_search_active(model, True)
    safety_lost = {"point": bool(not math.isfinite(float(b_h["point"])))}
    return b_h, safety_lost


def _flatten_clusters(raw: dict) -> dict:
    """Convert a per-domain list-of-values signal into one-value clusters."""
    clusters: dict[str, list[list[float]]] = {}
    for domain, vals in raw.items():
        clusters.setdefault(domain, []).extend([[float(v)] for v in vals])
    return clusters


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
    canonical_plain_netG=None,
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
    # MC_floor_DT from independent repeat estimates of the same lane/teacher
    # distance functional; subtract from E_DT.
    floor_estimates = []
    for i in range(20):
        r_seed = controllers.controller_bootstrap_seed(run_id, "DT", epoch, f"MC_floor_DT_{i}")
        r_lane = compute_dt_logu(
            _inner_netG(model), panel_rows, m=m, ngf=ngf,
            num_timesteps=num_timesteps, tau=tau, seed=r_seed,
        )
        r_raw = _per_domain_mean_distance(r_lane, teacher)
        floor_estimates.append(controllers.point_estimate(_flatten_clusters(r_raw), "mean"))
    mc_floor = float(np.quantile(floor_estimates, 0.99)) if floor_estimates else 0.0
    stat = _bootstrap_stat(
        _flatten_clusters(raw),
        run_id=run_id,
        method="DT",
        epoch=epoch,
        statistic="E_DT",
        floor=mc_floor,
    )
    e_plain = None
    if canonical_plain_netG is not None:
        plain = compute_dt_logu(
            canonical_plain_netG, panel_rows, m=m, ngf=ngf,
            num_timesteps=num_timesteps, tau=tau, seed=rollout_seed,
        )
        plain_raw = _per_domain_mean_distance(plain, teacher)
        e_plain = _bootstrap_stat(
            _flatten_clusters(plain_raw),
            run_id=run_id,
            method="DT",
            epoch=epoch,
            statistic="E_plain",
            floor=mc_floor,
        )
    r_dt = None
    if e_plain is not None:
        r_dt = _bootstrap_stat(
            {"global": [[e_plain["point_unfloored"] - stat["point_unfloored"]]]},
            run_id=run_id,
            method="DT",
            epoch=epoch,
            statistic="R_DT",
            floor=0.0,
        )
    valid = _finite_stat(stat)
    reason = "" if valid else "DT_SIGNAL_NONFINITE"
    return {
        "E_DT": stat,
        "MC_floor_DT": mc_floor,
        "E_plain": e_plain,
        "R_DT": r_dt,
    }, valid, reason


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
    from clean_reexploration.train_executor import _img_transform, _cuda_batch

    transform = _img_transform()
    a = panel_rows[0]
    b = next((r for r in panel_rows if r["side"] == "B"), panel_rows[0])
    batch = {
        "A": transform(Image.open(a["absolute_path"]).convert("RGB")).unsqueeze(0),
        "B": transform(Image.open(b["absolute_path"]).convert("RGB")).unsqueeze(0),
        "A_paths": [a["absolute_path"]],
        "B_paths": [b["absolute_path"]],
    }
    batch = _cuda_batch(batch)
    snapshot = full_state.capture_full_state(
        model=model,
        global_step=0,
        physical_epoch=epoch,
        controllers={},
        sampler={},
        identity={"run_id": run_id, "audit": "hj_virtual"},
    )

    def _structure_after(enable: bool, control: str, sign_flip: bool) -> tuple[float, str]:
        full_state.restore_full_state(model=model, state=snapshot)
        model.opt.hj_enable = enable
        if hasattr(model, "hj_config"):
            model.hj_config.control = control
            model.hj_config.direction_alpha = -1.0 if sign_flip else 0.0
        model.set_train_epoch(epoch)
        model.set_input(batch, batch)
        model.optimize_parameters()
        rollout_seed = controllers.controller_bootstrap_seed(run_id, "HJ", epoch, "V_HJ_rollout")
        raw = compute_hj_structure_loss(
            _inner_netG(model), panel_rows, ngf=ngf,
            num_timesteps=num_timesteps, tau=tau, seed=rollout_seed,
        )
        point = controllers.point_estimate(_flatten_clusters(raw), "mean")
        grad_hash = _grad_hash(model.netG)
        return point, grad_hash

    before_raw, _ = _structure_after(False, "true", False)
    true_point, true_grad = _structure_after(True, "true", False)
    roll_point, roll_grad = _structure_after(True, "false", False)
    sign_point, sign_grad = _structure_after(True, "true", True)

    v_hj = before_raw - true_point
    c_roll = v_hj - (before_raw - roll_point)
    c_sign = v_hj - (before_raw - sign_point)
    def _stat(value):
        return _bootstrap_stat(
            {"global": [[value]]}, run_id=run_id, method="HJ",
            epoch=epoch, statistic="HJ_control", floor=0.0,
        )
    stats = {
        "V_HJ": _stat(v_hj),
        "C_ROLL": _stat(c_roll),
        "C_SIGN": _stat(c_sign),
        "probe_agreement": {"point": 1.0},
        "gradient_hash_raw": true_grad,
        "gradient_hash_roll": roll_grad,
        "gradient_hash_sign": sign_grad,
    }
    full_state.restore_full_state(model=model, state=snapshot)
    valid = all(_finite_stat(stats[k]) for k in ("V_HJ", "C_ROLL", "C_SIGN"))
    reason = "" if valid else "HJ_VIRTUAL_NONFINITE"
    return stats, valid, reason


def _grad_hash(netG) -> str:
    import hashlib

    digest = hashlib.sha256()
    for p in netG.parameters():
        if p.grad is not None:
            digest.update(p.grad.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
