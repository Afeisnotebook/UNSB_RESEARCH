"""Generation-0 actual-update counterfactual auditing.

Each cell starts from one complete historical state, restores its networks,
optimizers, data streams and RNG bundle, then commits exactly one real UNSB
optimizer update under either plain or the historical proposal.  The artifact
contains only target-blind losses/diagnostics and update geometry.  Paired
metrics are joined later, after a branch is complete.
"""

from __future__ import annotations

import copy
import contextlib
import gc
import io
import json
from pathlib import Path

import torch

from .catalog import MatchedCheckpoint
from .observations import (
    StateObservation,
    state_dict_delta_cosine,
    state_dict_update_geometry,
)
from .search001_compat import modules


PROBE_OPTIONS = {
    "plain": {"model": "sb", "mechanisms": ()},
    "dt": {"model": "dtcov", "mechanisms": ()},
    "hj": {"model": "hj", "mechanisms": ()},
    "hj_handoff": {"model": "hj", "mechanisms": ()},
    "hnek": {"model": "hnek_search", "mechanisms": ()},
    "lbst": {"model": "sb", "mechanisms": ("lbst",)},
    "ptq": {"model": "sb", "mechanisms": ("ptq",)},
    "dcum": {"model": "sb", "mechanisms": ("dcum",)},
    "aeb": {"model": "sb", "mechanisms": ("aeb",)},
    "lttr_tangent": {"model": "lttr", "mechanisms": ()},
    "lttr_direction": {"model": "lttr", "mechanisms": ()},
}


def _lane(probe: str, suffix: str):
    protocol, _, _ = modules()
    config = PROBE_OPTIONS[probe]
    return protocol.LaneSpec(
        name=f"audit_{probe}_{suffix}",
        model=config["model"],
        mechanisms=tuple(config["mechanisms"]),
        family="probe",
        estimated_g_flops_multiplier=2.0 if probe == "aeb" else 1.0,
    )


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clean-unsb-directional-v1":
        raise RuntimeError(f"unsupported checkpoint schema: {path}")
    return payload


def _cpu_state_dict(net) -> dict[str, torch.Tensor]:
    runtime = modules()[1]
    return {
        key: value.detach().cpu().clone()
        for key, value in runtime.inner(net).state_dict().items()
    }


def _force_operator_active(model, probe: str, step: int) -> None:
    model.set_search_step(step, max(step + 10_000_000, 10_000_000))
    if probe == "dt":
        model.opt.dtcov_lambda_schedule = "fixed"
        model.opt.dtcov_lambda = 0.001
        model.opt.dtcov_search_start_step = 0
        model.opt.dtcov_search_duration_steps = max(step + 10_000_000, 10_000_000)
    elif probe in {"hj", "hj_handoff"}:
        model.opt.hj_enable = True
        model.opt.hj_search_start_step = 0
        model.opt.hj_search_duration_steps = max(step + 10_000_000, 10_000_000)
        # The clean HJ implementation accumulates diagnostics only when a
        # logger exists.  A truthy sentinel enables accumulation; no epoch
        # boundary is crossed in this one-step audit, so .log is never used.
        model._hj_diag = _NullDiagnostics()
    elif probe.startswith("lttr"):
        model.opt.lttr_enable = True
        model.opt.lttr_start_step = 0
        model.opt.lttr_duration_steps = max(step + 10_000_000, 10_000_000)


def _method_diagnostics(model, probe: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if hasattr(model, "time_idx"):
        result["bridge_time_index"] = float(model.time_idx.reshape(-1)[0].item())
    if probe == "dt":
        result["dt_loss_u_match"] = float(getattr(model, "loss_U_match", 0.0))
        result["dt_lambda"] = float(model.dtcov.config.lambda_value)
        result["dt_chart_cells"] = float(len(model.dtcov.stats.store))
    elif probe in {"hj", "hj_handoff"}:
        result["hj_gate_sum"] = float(getattr(model, "_hj_gate_sum", 0.0))
        result["hj_risk_sum"] = float(getattr(model, "_hj_risk_sum", 0.0))
        result["hj_probe_sum"] = float(getattr(model, "_hj_probe_sum", 0.0))
        result["hj_risk_positive_sum"] = float(
            getattr(model, "_hj_risk_positive_sum", 0.0)
        )
    elif probe == "hnek":
        cfg = model._hnek_search_cfg
        result.update({
            "hnek_gamma": float(cfg.gamma),
            "hnek_physical_horizon": float(cfg.horizon_mode == "physical"),
            "hnek_residual_coordinate": float(cfg.coord == "residual"),
        })
    elif probe == "lbst" and getattr(model, "_lbst_netG", None) is not None:
        current = modules()[1].inner(model.netG)
        total = 0.0
        base = 0.0
        with torch.no_grad():
            for teacher, live in zip(model._lbst_netG.parameters(), current.parameters()):
                total += float((teacher.detach() - live.detach()).square().sum().item())
                base += float(live.detach().square().sum().item())
        result["lbst_parameter_gap_ratio"] = (total / max(base, 1e-20)) ** 0.5
    return result


def _native_state_diagnostics(model) -> dict[str, float]:
    """Cheap target-blind state diagnostics available after a real update."""
    result: dict[str, float] = {}

    def rms(value: torch.Tensor) -> float:
        return float(value.detach().float().square().mean().sqrt().item())

    with torch.no_grad():
        if all(hasattr(model, name) for name in ("fake_B", "real_A_noisy")):
            residual = model.fake_B - model.real_A_noisy
            result["endpoint_residual_l2"] = rms(residual)
            horizon = 1.0
            if hasattr(model, "times") and hasattr(model, "time_idx"):
                time_value = float(
                    model.times[model.time_idx.reshape(-1)[0]].detach().item()
                )
                horizon = max(1.0 - time_value, 1e-8)
                result["physical_horizon"] = horizon
            result["rollout_velocity_l2"] = result["endpoint_residual_l2"] / horizon
        if all(hasattr(model, name) for name in ("real_A_noisy", "real_A")):
            result["rollout_input_displacement_l2"] = rms(
                model.real_A_noisy - model.real_A
            )
        if all(hasattr(model, name) for name in ("fake_B", "fake_B2")):
            # The two tensors use independent inputs and latents.  The name is
            # deliberately not "latent dispersion" because that would
            # overclaim what this observable isolates.
            result["independent_endpoint_separation_l2"] = rms(
                model.fake_B - model.fake_B2
            )

    gradient_sq = 0.0
    moment_sq = 0.0
    gradient_moment = 0.0
    optimizer = getattr(model, "optimizer_G", None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                moment = optimizer.state.get(parameter, {}).get("exp_avg")
                if gradient is None or moment is None:
                    continue
                g = gradient.detach().double()
                m = moment.detach().double()
                gradient_sq += float(torch.sum(g * g).item())
                moment_sq += float(torch.sum(m * m).item())
                gradient_moment += float(torch.sum(g * m).item())
    result["generator_grad_norm"] = gradient_sq ** 0.5
    result["adam_first_moment_norm"] = moment_sq ** 0.5
    denominator = (gradient_sq * moment_sq) ** 0.5
    result["adam_moment_gradient_cosine"] = (
        gradient_moment / denominator if denominator > 0.0 else 0.0
    )

    current_losses = model.get_current_losses()
    generator_loss = abs(float(current_losses.get("G", 0.0)))
    scale = max(generator_loss, 1e-12)
    discriminator_loss = abs(float(current_losses.get("D_real", 0.0))) + abs(
        float(current_losses.get("D_fake", 0.0))
    )
    result["d_to_g_loss_ratio"] = discriminator_loss / scale
    if hasattr(model, "loss_E"):
        result["e_to_g_loss_ratio"] = abs(float(model.loss_E.detach().item())) / scale
        result["bridge_kdd_critic_loss"] = float(model.loss_E.detach().item())
    return result


class _NullDiagnostics:
    def log(self, **fields) -> None:
        del fields


def _build_and_branch(
    *,
    source: dict,
    method_state: dict,
    target_probe: str,
    source_label: str,
    per_domain: int,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    horizon: int,
    data_root: Path | None,
    evaluate_after: bool,
) -> tuple[
    StateObservation,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict | None,
]:
    _, runtime, _ = modules()
    spec = _lane(target_probe, source_label)
    steps_per_epoch = 6 * int(per_domain)
    step = int(source["step"])
    runtime.seed_everything(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = runtime.build_options(
            spec,
            dataroot=train_view,
            checkpoint_dir=work_dir / "option_records",
            steps_per_epoch=steps_per_epoch,
            total_steps=max(int(source.get("target_steps", step + horizon)), step + horizon),
            seed=seed,
            gpu=gpu,
        )
    datasets = runtime.build_datasets(opt, rows, per_domain)
    stream_a = runtime.SerializableDataStream(datasets[0], seed=seed + 101)
    stream_b = runtime.SerializableDataStream(datasets[1], seed=seed + 202)
    with contextlib.redirect_stdout(io.StringIO()):
        model = runtime.build_model(opt, stream_a.next(), stream_b.next())
    runtime.load_model_state(model, source["model"], load_extra=False)
    matched_costate = target_probe != "plain" and source_label == target_probe
    if matched_costate:
        model.load_extra_training_state(copy.deepcopy(method_state["model"]["extra"]))
    else:
        # u_i(S_plain) must initialize i's co-state from S_plain.  Transplanting
        # a frozen DT teacher/statistics bundle from S_i changes the operator
        # being audited and can create a wholly artificial teacher gap.
        model.load_extra_training_state({
            "search_global_step": step,
            "search_total_steps": int(source.get("target_steps", step + 1)),
        })
    stream_a.load_state_dict(copy.deepcopy(source["stream_a"]))
    stream_b.load_state_dict(copy.deepcopy(source["stream_b"]))
    runtime.restore_rng(copy.deepcopy(source["rng"]))
    _force_operator_active(model, target_probe, step)
    before = _cpu_state_dict(model.netG)
    loss_sums: dict[str, float] = {}
    diagnostic_sums: dict[str, float] = {}
    domains_seen: dict[str, int] = {}
    times_seen: dict[int, int] = {}
    grouped_values: dict[tuple[str, str, str], list[float]] = {}
    previous_diagnostics = _method_diagnostics(model, target_probe)
    for offset in range(int(horizon)):
        current_step = step + offset
        _force_operator_active(model, target_probe, current_step)
        physical_epoch = 1 + current_step // steps_per_epoch
        model.set_train_epoch(physical_epoch)
        batch_a = stream_a.next()
        batch_b = stream_b.next()
        paths = list(batch_a.get("A_paths", []))
        domain = "unknown"
        if paths:
            stem = Path(paths[0]).stem
            domain = stem.split("__", 1)[0] if "__" in stem else "unknown"
            domains_seen[domain] = domains_seen.get(domain, 0) + 1
        model.set_input(batch_a, batch_b)
        model.optimize_parameters()
        for key, value in model.get_current_losses().items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
        diagnostics = {
            **_method_diagnostics(model, target_probe),
            **_native_state_diagnostics(model),
        }
        if "bridge_time_index" in diagnostics:
            time_index = int(diagnostics["bridge_time_index"])
            times_seen[time_index] = times_seen.get(time_index, 0) + 1
        for key, value in diagnostics.items():
            current_value = float(value)
            if key.startswith("hj_") and key.endswith("_sum"):
                step_value = current_value - float(previous_diagnostics.get(key, 0.0))
            else:
                step_value = current_value
            diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + step_value
        for diagnostic_name in (
            "generator_grad_norm",
            "endpoint_residual_l2",
            "rollout_velocity_l2",
            "adam_moment_gradient_cosine",
        ):
            if diagnostic_name not in diagnostics:
                continue
            grouped_values.setdefault(
                ("domain", domain, diagnostic_name), []
            ).append(float(diagnostics[diagnostic_name]))
            if "bridge_time_index" in diagnostics:
                grouped_values.setdefault(
                    ("time", str(int(diagnostics["bridge_time_index"])), diagnostic_name), []
                ).append(float(diagnostics[diagnostic_name]))
        completed = current_step + 1
        if completed % steps_per_epoch == 0:
            model.update_learning_rate()
        previous_diagnostics = _method_diagnostics(model, target_probe)
    after = _cpu_state_dict(model.netG)
    losses = {key: value / float(horizon) for key, value in loss_sums.items()}
    diagnostics = {
        key: value / float(horizon) for key, value in diagnostic_sums.items()
    }
    for domain, count in sorted(domains_seen.items()):
        diagnostics[f"domain_count::{domain}"] = float(count)
    for time_index, count in sorted(times_seen.items()):
        diagnostics[f"time_count::{time_index}"] = float(count)
    for (group_kind, group_name, diagnostic_name), values in sorted(grouped_values.items()):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        prefix = f"{group_kind}_moment::{group_name}::{diagnostic_name}"
        diagnostics[f"{prefix}::mean"] = mean
        diagnostics[f"{prefix}::variance"] = variance
    observation = StateObservation(
        source_probe=target_probe,
        source_state=source_label,
        step=step,
        domain=(next(iter(domains_seen)) if len(domains_seen) == 1 else None),
        bridge_time=(next(iter(times_seen)) if len(times_seen) == 1 else None),
        losses=losses,
        diagnostics=diagnostics,
        paired_metrics_accessed=False,
    )
    metrics = None
    if evaluate_after:
        if data_root is None:
            raise ValueError("data_root is required for post-branch evaluation")
        evaluate = modules()[2].evaluate
        metrics = evaluate(
            model,
            rows=rows,
            data_root=data_root,
            start_per_domain=0,
            count_per_domain=10,
            eval_seed=seed,
            include_lpips=False,
        )
    del model, stream_a, stream_b, datasets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return observation, before, after, metrics


def audit_cell(
    cell: MatchedCheckpoint,
    *,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    horizon: int = 1,
    data_root: Path | None = None,
    evaluate_after: bool = False,
    source_states: tuple[str, ...] | None = None,
) -> list[dict]:
    plain_state = _load(cell.plain)
    method_state = _load(cell.method)
    results = []
    selected_states = set(source_states or ("plain", cell.probe))
    for source_label, source in (("plain", plain_state), (cell.probe, method_state)):
        if source_label not in selected_states:
            continue
        reference, before_ref, after_ref, reference_metrics = _build_and_branch(
            source=source,
            method_state=method_state,
            target_probe="plain",
            source_label=source_label,
            per_domain=cell.per_domain,
            rows=rows,
            train_view=train_view,
            work_dir=work_dir,
            seed=seed,
            gpu=gpu,
            horizon=horizon,
            data_root=data_root,
            evaluate_after=evaluate_after,
        )
        proposal, before_prop, after_prop, proposal_metrics = _build_and_branch(
            source=source,
            method_state=method_state,
            target_probe=cell.probe,
            source_label=source_label,
            per_domain=cell.per_domain,
            rows=rows,
            train_view=train_view,
            work_dir=work_dir,
            seed=seed,
            gpu=gpu,
            horizon=horizon,
            data_root=data_root,
            evaluate_after=evaluate_after,
        )
        next_native_consensus = None
        if int(horizon) == 1:
            _, before_ref2, after_ref2, _ = _build_and_branch(
                source=source,
                method_state=method_state,
                target_probe="plain",
                source_label=source_label,
                per_domain=cell.per_domain,
                rows=rows,
                train_view=train_view,
                work_dir=work_dir,
                seed=seed,
                gpu=gpu,
                horizon=2,
                data_root=None,
                evaluate_after=False,
            )
            for key in before_ref:
                if not torch.equal(before_ref[key], before_ref2[key]):
                    raise AssertionError(f"two-step native start mismatch: {key}")
            next_native_consensus = state_dict_delta_cosine(
                after_ref,
                after_prop,
                after_ref,
                after_ref2,
            )
            del before_ref2, after_ref2
        if tuple(before_ref) != tuple(before_prop):
            raise AssertionError("reference/proposal generator identity differs")
        for key in before_ref:
            if not torch.equal(before_ref[key], before_prop[key]):
                raise AssertionError(f"reference/proposal start mismatch: {key}")
        global_geometry, blocks = state_dict_update_geometry(
            before_ref, after_ref, after_prop
        )
        branch_label = None
        if reference_metrics is not None and proposal_metrics is not None:
            compare = modules()[2].compare
            branch_label = compare(
                proposal_metrics, reference_metrics, step=cell.step + int(horizon)
            )
            branch_label["available_to_controller"] = False
            branch_label["computed_after_branch"] = True
        results.append({
            "schema": "clean-unsb-search003-atlas-row-v1",
            "probe": cell.probe,
            "stage": cell.stage,
            "step": cell.step,
            "horizon": int(horizon),
            "per_domain": cell.per_domain,
            "source_state": source_label,
            "operator_costate": (
                "reinitialized_from_source_state"
                if source_label == "plain" else "matched_historical_costate"
            ),
            "reference": reference.to_dict(),
            "proposal": proposal.to_dict(),
            "update_geometry": global_geometry,
            "block_geometry": blocks,
            "next_independent_native_consensus": next_native_consensus,
            "post_branch_development_label": branch_label,
            "paired_metrics_accessed_by_controller": False,
            "paired_development_evaluated_after_branch": bool(branch_label is not None),
            "confirmation20_opened": False,
        })
        del before_ref, before_prop, after_ref, after_prop
        gc.collect()
    del plain_state, method_state
    gc.collect()
    return results


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_atlas(path: Path) -> dict:
    """Atomically migrate early SEARCH-003 rows to the explicit access schema."""
    rows = []
    legacy_rows = 0
    invalid_hj_accumulators = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            was_legacy = "paired_metrics_accessed" in row
            if was_legacy:
                legacy_rows += 1
                row.pop("paired_metrics_accessed", None)
            row["paired_metrics_accessed_by_controller"] = False
            row["paired_development_evaluated_after_branch"] = bool(
                row.get("post_branch_development_label") is not None
            )
            validity = dict(row.get("diagnostic_validity", {}))
            if "operator_costate" not in row:
                if row.get("source_state") == row.get("probe"):
                    row["operator_costate"] = "matched_historical_costate"
                elif row.get("probe") == "dt":
                    row["operator_costate"] = "legacy_transplanted_method_costate"
                    validity["causal_operator_state"] = "invalid_teacher_transplant"
                elif row.get("probe") in {"hj", "hj_handoff"}:
                    row["operator_costate"] = "legacy_transplanted_method_costate"
                    # HJ's audited configuration uses a constant schedule; its
                    # restored controller counters do not enter the update.
                    validity["causal_operator_state"] = "valid_constant_schedule_equivalent"
                else:
                    row["operator_costate"] = "stateless_equivalent"
                    validity["causal_operator_state"] = "valid_stateless_equivalent"
            else:
                validity.setdefault("causal_operator_state", "valid")
            if (
                was_legacy
                and row.get("probe") in {"hj", "hj_handoff"}
                and int(row.get("horizon", 1)) > 1
            ):
                # These early rows averaged already-cumulative HJ counters.
                # Update geometry and paired-after-branch labels remain valid,
                # but those four counters must never train a signal rule.
                validity["hj_accumulators"] = "invalid_cumulative_legacy"
                invalid_hj_accumulators += 1
            else:
                validity.setdefault("hj_accumulators", "valid")
            row["diagnostic_validity"] = validity
            rows.append(row)
    temporary = Path(path).with_suffix(Path(path).suffix + ".normalized.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return {
        "rows": len(rows),
        "legacy_rows_migrated": legacy_rows,
        "invalid_hj_accumulator_rows_marked": invalid_hj_accumulators,
    }
