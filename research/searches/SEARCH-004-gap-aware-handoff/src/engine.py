"""Deterministic component-level handoff continuation engine."""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import math
import os
import time
import types
from pathlib import Path

import torch

from .catalog import HandoffCheckpoint
from .protocol import Search004Protocol
from .search001_compat import modules
from .search003_compat import load_receding
from .search005_compat import load_model_operators
from .statistics import empirical_bernstein_cs
from .transports import least_change_native_moment_projection
from .state import (
    checkpoint_method_costate,
    cpu_clone,
    exact_equal,
    export_named_optimizers,
    load_named_optimizers,
    optimizer_bindings,
    torch_digest,
    zero_named_optimizer,
)


class _NullDiagnostics:
    def log(self, **fields) -> None:
        del fields


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _lane(checkpoint: HandoffCheckpoint):
    protocol, _, _ = modules()
    return protocol.LaneSpec(
        name=f"search004_{checkpoint.checkpoint_id}",
        model=checkpoint.model,
        family="search004_handoff_audit",
    )


def _force_active(model, family: str, step: int) -> None:
    model.set_search_step(step, max(step + 10_000_000, 10_000_000))
    if family == "dt":
        model.opt.dtcov_lambda_schedule = "fixed"
        model.opt.dtcov_lambda = 0.001
        model.opt.dtcov_search_start_step = 0
        model.opt.dtcov_search_duration_steps = max(step + 10_000_000, 10_000_000)
    elif family == "hj":
        model.opt.hj_enable = True
        model.opt.hj_search_start_step = 0
        model.opt.hj_search_duration_steps = max(step + 10_000_000, 10_000_000)
        model._hj_diag = _NullDiagnostics()
    elif family == "hnek":
        from models.hnek.hnek_search import set_hnek_search_active
        set_hnek_search_active(model, True)
    elif family == "pcoa":
        load_model_operators().set_pcoa_active(model, True)


def _disable_hnek(model) -> None:
    if hasattr(model, "_hnek_search_cfg"):
        from models.hnek.hnek_search import set_hnek_search_active
        set_hnek_search_active(model, False)


class HandoffEngine:
    """One live source-class model reused transactionally across audit arms."""

    def __init__(
        self,
        *,
        checkpoint: HandoffCheckpoint,
        rows: list[dict],
        train_view: Path,
        work_dir: Path,
        seed: int,
        gpu: int,
        max_horizon: int,
    ) -> None:
        self.checkpoint = checkpoint
        self.rows = rows
        self.seed = int(seed)
        self.gpu = int(gpu)
        self.train_view = Path(train_view)
        self.work_dir = Path(work_dir)
        self.max_horizon = int(max_horizon)
        self.steps_per_epoch = 6 * int(checkpoint.per_domain)
        _, self.runtime, self.evaluator = modules()
        self.runtime.seed_everything(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            opt = self.runtime.build_options(
                _lane(checkpoint),
                dataroot=Path(train_view),
                checkpoint_dir=Path(work_dir) / "option_records" / checkpoint.checkpoint_id,
                steps_per_epoch=self.steps_per_epoch,
                total_steps=checkpoint.step + max_horizon,
                seed=seed,
                gpu=gpu,
            )
            dataset_a, dataset_b = self.runtime.build_datasets(
                opt, rows, checkpoint.per_domain
            )
            self.stream_a = self.runtime.SerializableDataStream(dataset_a, seed=seed + 101)
            self.stream_b = self.runtime.SerializableDataStream(dataset_b, seed=seed + 202)
            self.model = self.runtime.build_model(
                opt, self.stream_a.next(), self.stream_b.next()
            )
            if checkpoint.family == "pcoa":
                load_model_operators().install_pcoa(self.model)
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self._fresh_extra = cpu_clone(self.model.get_extra_training_state())
        self._base_class = next(
            cls for cls in type(self.model).__mro__ if cls.__name__ == "SBModel"
        )
        self.plain_payload = load_payload(checkpoint.plain)
        self.method_payload = load_payload(checkpoint.method)
        self.parent_digest = torch_digest(self.method_payload)
        self._transport_record: dict | None = None

    def close(self) -> None:
        del self.model, self.stream_a, self.stream_b, self.dataset_a, self.dataset_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_extra(self, value: dict | None) -> None:
        self.model.load_extra_training_state(cpu_clone(self._fresh_extra))
        if self.checkpoint.family == "dt":
            self.model.dtcov.teacher = None
            self.model.dtcov.stats.store = {}
            self.model.dtcov.iter = 0
        if value is not None:
            self.model.load_extra_training_state(cpu_clone(value))

    @contextlib.contextmanager
    def operator(self, *, active: bool):
        if active:
            _force_active(self.model, self.checkpoint.family, self.current_step)
            try:
                yield
            finally:
                pass
            return
        if self.checkpoint.family == "hnek":
            _disable_hnek(self.model)
            try:
                yield
            finally:
                pass
            return
        if self.checkpoint.family == "pcoa":
            load_model_operators().set_pcoa_active(self.model, False)
            try:
                yield
            finally:
                pass
            return
        patched = ["compute_G_loss"]
        self.model.compute_G_loss = types.MethodType(
            self._base_class.compute_G_loss, self.model
        )
        if self.checkpoint.family == "hj":
            patched.append("calculate_NCE_loss")
            self.model.calculate_NCE_loss = types.MethodType(
                self._base_class.calculate_NCE_loss, self.model
            )
        try:
            yield
        finally:
            for name in patched:
                delattr(self.model, name)

    def _load_common_clock(self) -> None:
        source = self.method_payload
        self.stream_a.load_state_dict(cpu_clone(source["stream_a"]))
        self.stream_b.load_state_dict(cpu_clone(source["stream_b"]))
        self.runtime.restore_rng(cpu_clone(source["rng"]))

    def _load_base(self, payload: dict, *, method_costate: bool) -> None:
        self.runtime.load_model_state(self.model, payload["model"], load_extra=False)
        self._reset_extra(
            checkpoint_method_costate(self.method_payload) if method_costate else None
        )
        self._load_common_clock()
        self.current_step = int(self.checkpoint.step)

    def prepare_arm(self, arm: str, protocol: Search004Protocol) -> dict:
        self._transport_record = None
        source = self.method_payload
        plain = self.plain_payload
        if arm == "P_common_plain":
            self._load_base(plain, method_costate=False)
        elif arm in {
            "F_g_only_transplant", "G_gf_transplant", "K_gf_state_transplant",
        }:
            self._load_base(plain, method_costate=False)
            inner = self.runtime.inner(self.model.netG)
            inner.load_state_dict(source["model"]["networks"]["G"], strict=True)
            if arm in {"G_gf_transplant", "K_gf_state_transplant"}:
                inner_f = self.runtime.inner(self.model.netF)
                inner_f.load_state_dict(source["model"]["networks"]["F"], strict=True)
            if arm == "K_gf_state_transplant":
                # Positional loading is safe here because the canonical
                # checkpoint schema and engineering gate have already proved
                # the G/F optimizer bindings.  All other optimizer/co-state
                # components remain from the common plain control.
                state = self.runtime.model_state(self.model)
                state["optimizers"][0] = cpu_clone(source["model"]["optimizers"][0])
                state["optimizers"][3] = cpu_clone(source["model"]["optimizers"][3])
                self.runtime.load_model_state(self.model, state, load_extra=False)
                self._reset_extra(None)
        else:
            self._load_base(source, method_costate=True)
        if arm == "B_gf_zero_moment":
            zero_named_optimizer(self.model, ("G", "F"))
        if arm in {"C_local_native_moment", "E_combined"}:
            self._reconstruct_local_native_moments(protocol.shadow_steps)
        if arm == "H_native_moment_projection":
            gradients = self._collect_native_mean_gradients(
                protocol.moment_projection_batches
            )
            self._transport_record = least_change_native_moment_projection(
                self.model, gradients, players=("G", "F")
            )
        return self.capture_state(arm=arm, completed=0)

    def capture_state(self, *, arm: str, completed: int) -> dict:
        return {
            "schema": "clean-unsb-search004-arm-state-v1",
            "checkpoint_id": self.checkpoint.checkpoint_id,
            "arm": arm,
            "step": int(self.current_step),
            "completed": int(completed),
            "model": cpu_clone(self.runtime.model_state(self.model)),
            "rng": self.runtime.capture_rng(),
            "stream_a": self.stream_a.state_dict(),
            "stream_b": self.stream_b.state_dict(),
            "transport_record": cpu_clone(self._transport_record),
            "confirmation20_opened": False,
        }

    def load_state(self, state: dict) -> None:
        self.runtime.load_model_state(self.model, state["model"], load_extra=False)
        self._reset_extra(state["model"].get("extra"))
        self.stream_a.load_state_dict(cpu_clone(state["stream_a"]))
        self.stream_b.load_state_dict(cpu_clone(state["stream_b"]))
        self.runtime.restore_rng(cpu_clone(state["rng"]))
        self.current_step = int(state["step"])
        self._transport_record = cpu_clone(state.get("transport_record"))

    def _native_generator_backward(self) -> torch.Tensor:
        self.model.set_requires_grad(self.model.netD, False)
        self.model.set_requires_grad(self.model.netE, False)
        self.model.optimizer_G.zero_grad()
        if self.model.opt.netF == "mlp_sample":
            self.model.optimizer_F.zero_grad()
        loss = self.model.compute_G_loss()
        loss.backward()
        return loss

    def _reconstruct_local_native_moments(self, count: int) -> None:
        parent = cpu_clone(self.runtime.model_state(self.model))
        parent_rng = self.runtime.capture_rng()
        parent_a = self.stream_a.state_dict()
        parent_b = self.stream_b.state_dict()
        zero_named_optimizer(self.model, ("G", "F"))
        with self.operator(active=False):
            for offset in range(int(count)):
                step = self.checkpoint.step + offset
                self.current_step = step
                self.model.set_train_epoch(1 + step // self.steps_per_epoch)
                self.model.set_search_step(step, self.checkpoint.step + self.max_horizon)
                self.model.set_input(self.stream_a.next(), self.stream_b.next())
                self.model.forward()
                self._native_generator_backward()
                for name in ("G", "F"):
                    optimizer, _ = optimizer_bindings(self.model)[name]
                    beta1, beta2 = optimizer.param_groups[0]["betas"]
                    for group in optimizer.param_groups:
                        for parameter in group["params"]:
                            if parameter.grad is None:
                                continue
                            state = optimizer.state[parameter]
                            if not state:
                                state["step"] = torch.tensor(0.0, device=parameter.device)
                                state["exp_avg"] = torch.zeros_like(parameter)
                                state["exp_avg_sq"] = torch.zeros_like(parameter)
                            state["step"] += 1
                            state["exp_avg"].mul_(beta1).add_(parameter.grad, alpha=1.0 - beta1)
                            state["exp_avg_sq"].mul_(beta2).addcmul_(
                                parameter.grad, parameter.grad, value=1.0 - beta2
                            )
        shadow = export_named_optimizers(self.model)
        self.runtime.load_model_state(self.model, parent, load_extra=False)
        self._reset_extra(parent.get("extra"))
        load_named_optimizers(self.model, shadow, only=("G", "F"))
        self.stream_a.load_state_dict(parent_a)
        self.stream_b.load_state_dict(parent_b)
        self.runtime.restore_rng(parent_rng)
        self.current_step = int(self.checkpoint.step)

    def _collect_native_mean_gradients(self, count: int) -> dict[str, torch.Tensor]:
        """Estimate a native tangent without consuming the continuation clock."""
        if int(count) <= 0:
            raise ValueError("native moment projection requires positive batch count")
        parent = cpu_clone(self.runtime.model_state(self.model))
        parent_rng = self.runtime.capture_rng()
        parent_a = self.stream_a.state_dict()
        parent_b = self.stream_b.state_dict()
        parent_step = int(self.current_step)
        totals: dict[str, torch.Tensor] = {}
        try:
            for offset in range(int(count)):
                self.current_step = parent_step
                sample = self._generator_gradient_sample(active=False)
                for key, gradient in sample["gradients"].items():
                    if key not in totals:
                        totals[key] = gradient.clone()
                    else:
                        totals[key].add_(gradient)
                # Restore all model/co-state buffers while retaining the
                # advanced shadow RNG and streams for an independent batch.
                shadow_rng = self.runtime.capture_rng()
                shadow_a = self.stream_a.state_dict()
                shadow_b = self.stream_b.state_dict()
                self.runtime.load_model_state(self.model, parent, load_extra=False)
                self._reset_extra(parent.get("extra"))
                self.stream_a.load_state_dict(shadow_a)
                self.stream_b.load_state_dict(shadow_b)
                self.runtime.restore_rng(shadow_rng)
            for gradient in totals.values():
                gradient.div_(float(count))
            return totals
        finally:
            self.runtime.load_model_state(self.model, parent, load_extra=False)
            self._reset_extra(parent.get("extra"))
            self.stream_a.load_state_dict(parent_a)
            self.stream_b.load_state_dict(parent_b)
            self.runtime.restore_rng(parent_rng)
            self.current_step = parent_step
            for network_name in self.model.model_names:
                for parameter in getattr(self.model, "net" + network_name).parameters():
                    parameter.grad = None

    def _hold_step(self, *, equilibrate: bool) -> None:
        self.model.forward()
        if not equilibrate:
            with torch.no_grad():
                self.model.compute_D_loss()
                self.model.compute_E_loss()
                self.model.compute_G_loss()
            return
        self.model.set_requires_grad(self.model.netD, True)
        self.model.optimizer_D.zero_grad()
        self.model.compute_D_loss().backward()
        self.model.optimizer_D.step()
        self.model.set_requires_grad(self.model.netE, True)
        self.model.optimizer_E.zero_grad()
        self.model.compute_E_loss().backward()
        self.model.optimizer_E.step()
        self.model.set_requires_grad(self.model.netD, False)
        self.model.set_requires_grad(self.model.netE, False)
        self.model.optimizer_G.zero_grad()
        self.model.optimizer_F.zero_grad()
        self.model.compute_G_loss().backward()
        self.model.optimizer_F.step()

    def _step_diagnostics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, (optimizer, parameters) in optimizer_bindings(self.model).items():
            grad_sq = moment_sq = effective_sq = grad_moment = 0.0
            parameter_sq = 0.0
            for _, parameter in parameters:
                parameter_sq += float(parameter.detach().double().square().sum().item())
                state = optimizer.state.get(parameter, {})
                moment = state.get("exp_avg")
                second = state.get("exp_avg_sq")
                gradient = parameter.grad
                if moment is not None:
                    m = moment.detach().double()
                    moment_sq += float((m * m).sum().item())
                    if second is not None:
                        effective = m / (second.detach().double().sqrt() + 1e-8)
                        effective_sq += float((effective * effective).sum().item())
                if gradient is not None:
                    g = gradient.detach().double()
                    grad_sq += float((g * g).sum().item())
                    if moment is not None:
                        grad_moment += float((g * moment.detach().double()).sum().item())
            denom = math.sqrt(max(grad_sq * moment_sq, 0.0))
            result[f"{name}_gradient_norm"] = math.sqrt(grad_sq)
            result[f"{name}_moment_norm"] = math.sqrt(moment_sq)
            result[f"{name}_effective_step_norm"] = math.sqrt(effective_sq)
            result[f"{name}_relative_step_norm"] = math.sqrt(effective_sq) / max(math.sqrt(parameter_sq), 1e-20)
            result[f"{name}_gradient_moment_cosine"] = grad_moment / denom if denom else 0.0
        losses = self.model.get_current_losses()
        for key, value in losses.items():
            result[f"loss::{key}"] = float(value)
        if hasattr(self.model, "loss_E"):
            result["loss::E"] = float(self.model.loss_E.detach().item())
        if hasattr(self.model, "time_idx"):
            result["bridge_time_index"] = float(self.model.time_idx.reshape(-1)[0].item())
        g_norm = result.get("G_gradient_norm", 0.0)
        result["D_over_G_gradient_balance"] = (
            result.get("D_gradient_norm", 0.0) / max(g_norm, 1e-20)
        )
        result["E_over_G_gradient_balance"] = (
            result.get("E_gradient_norm", 0.0) / max(g_norm, 1e-20)
        )
        return result

    def _generator_gradient_sample(self, *, active: bool) -> dict:
        """Measure one unpaired G/F gradient without committing any state."""
        self.model.set_train_epoch(1 + self.current_step // self.steps_per_epoch)
        if active:
            _force_active(self.model, self.checkpoint.family, self.current_step)
        else:
            self.model.set_search_step(
                self.current_step, self.checkpoint.step + self.max_horizon
            )
        self.model.set_input(self.stream_a.next(), self.stream_b.next())
        with self.operator(active=active):
            self.model.forward()
            loss = self._native_generator_backward()
        gradients = {}
        blocks: dict[str, dict[str, float]] = {}
        for network_name in ("G", "F"):
            network = self.runtime.inner(getattr(self.model, "net" + network_name))
            for name, parameter in network.named_parameters():
                if parameter.grad is None:
                    continue
                key = f"{network_name}.{name}"
                gradient = parameter.grad.detach().cpu().double().clone()
                gradients[key] = gradient
                block = f"{network_name}.{name.split('.', 1)[0]}"
                entry = blocks.setdefault(block, {"sq_norm": 0.0})
                entry["sq_norm"] += float(gradient.square().sum().item())
        paths = getattr(self.model, "image_paths", [])
        if isinstance(paths, str):
            paths = [paths]
        return {
            "loss": float(loss.detach().item()),
            "gradients": gradients,
            "blocks": blocks,
            "bridge_time_index": (
                None
                if not hasattr(self.model, "time_idx")
                else float(self.model.time_idx.reshape(-1)[0].item())
            ),
            "unpaired_batch_paths": [str(value) for value in paths],
        }

    def _transactional_gradient_geometry(self, *, skip_batches: int = 0) -> dict:
        """Compare a correction with native descent on the next independent batch."""
        parent = cpu_clone(self.runtime.model_state(self.model))
        parent_rng = self.runtime.capture_rng()
        parent_a = self.stream_a.state_dict()
        parent_b = self.stream_b.state_dict()
        parent_step = int(self.current_step)
        requires_grad = {
            id(parameter): bool(parameter.requires_grad)
            for network_name in self.model.model_names
            for parameter in getattr(self.model, "net" + network_name).parameters()
        }

        def restore_model() -> None:
            self.runtime.load_model_state(self.model, parent, load_extra=False)
            self._reset_extra(parent.get("extra"))
            self.current_step = parent_step
            for network_name in self.model.model_names:
                for parameter in getattr(self.model, "net" + network_name).parameters():
                    parameter.requires_grad_(requires_grad[id(parameter)])
                    parameter.grad = None

        def restore_point(rng: dict, stream_a: dict, stream_b: dict) -> None:
            restore_model()
            self.stream_a.load_state_dict(cpu_clone(stream_a))
            self.stream_b.load_state_dict(cpu_clone(stream_b))
            self.runtime.restore_rng(cpu_clone(rng))

        try:
            # Select a deterministic, non-overlapping audit pair without
            # committing the skipped samples to the real continuation stream.
            for _ in range(int(skip_batches)):
                self.stream_a.next()
                self.stream_b.next()
            audit_rng = self.runtime.capture_rng()
            audit_a = self.stream_a.state_dict()
            audit_b = self.stream_b.state_dict()
            intervention = self._generator_gradient_sample(active=True)
            restore_point(audit_rng, audit_a, audit_b)
            native = self._generator_gradient_sample(active=False)
            next_rng = self.runtime.capture_rng()
            next_a = self.stream_a.state_dict()
            next_b = self.stream_b.state_dict()
            restore_point(next_rng, next_a, next_b)
            future_native = self._generator_gradient_sample(active=False)
            keys = sorted(
                set(intervention["gradients"])
                | set(native["gradients"])
                | set(future_native["gradients"])
            )
            native_sq = future_native_sq = intervention_sq = correction_sq = 0.0
            native_intervention_dot = native_correction_dot = future_native_correction_dot = 0.0
            block_rows = {}
            for key in keys:
                gi = intervention["gradients"].get(key)
                gn = native["gradients"].get(key)
                gf = future_native["gradients"].get(key)
                if gi is None:
                    gi = torch.zeros_like(gn if gn is not None else gf)
                if gn is None:
                    gn = torch.zeros_like(gi)
                if gf is None:
                    gf = torch.zeros_like(gi)
                correction = gi - gn
                n_sq = float(gn.square().sum().item())
                f_sq = float(gf.square().sum().item())
                i_sq = float(gi.square().sum().item())
                c_sq = float(correction.square().sum().item())
                ni = float((gn * gi).sum().item())
                nc = float((gn * correction).sum().item())
                fc = float((gf * correction).sum().item())
                native_sq += n_sq
                future_native_sq += f_sq
                intervention_sq += i_sq
                correction_sq += c_sq
                native_intervention_dot += ni
                native_correction_dot += nc
                future_native_correction_dot += fc
                block = key.split(".", 2)
                block_name = ".".join(block[:2])
                row = block_rows.setdefault(block_name, {
                    "native_sq": 0.0,
                    "future_native_sq": 0.0,
                    "intervention_sq": 0.0,
                    "correction_sq": 0.0,
                    "native_intervention_dot": 0.0,
                    "native_correction_dot": 0.0,
                    "future_native_correction_dot": 0.0,
                })
                row["native_sq"] += n_sq
                row["future_native_sq"] += f_sq
                row["intervention_sq"] += i_sq
                row["correction_sq"] += c_sq
                row["native_intervention_dot"] += ni
                row["native_correction_dot"] += nc
                row["future_native_correction_dot"] += fc

            def cosine(dot: float, left_sq: float, right_sq: float) -> float:
                denominator = math.sqrt(max(left_sq * right_sq, 0.0))
                return dot / denominator if denominator else 0.0

            for row in block_rows.values():
                row["native_intervention_cosine"] = cosine(
                    row["native_intervention_dot"],
                    row["native_sq"], row["intervention_sq"],
                )
                row["native_correction_cosine"] = cosine(
                    row["native_correction_dot"],
                    row["native_sq"], row["correction_sq"],
                )
                row["future_native_correction_cosine"] = cosine(
                    row["future_native_correction_dot"],
                    row["future_native_sq"], row["correction_sq"],
                )
            return {
                "native_loss": native["loss"],
                "future_native_loss": future_native["loss"],
                "intervention_loss": intervention["loss"],
                "native_gradient_norm": math.sqrt(native_sq),
                "future_native_gradient_norm": math.sqrt(future_native_sq),
                "intervention_gradient_norm": math.sqrt(intervention_sq),
                "correction_gradient_norm": math.sqrt(correction_sq),
                "native_intervention_cosine": cosine(
                    native_intervention_dot, native_sq, intervention_sq
                ),
                "native_correction_cosine": cosine(
                    native_correction_dot, native_sq, correction_sq
                ),
                "future_native_correction_cosine": cosine(
                    future_native_correction_dot, future_native_sq, correction_sq
                ),
                "block_geometry": block_rows,
                "bridge_time_index": native["bridge_time_index"],
                "unpaired_batch_paths": native["unpaired_batch_paths"],
                "future_unpaired_batch_paths": future_native["unpaired_batch_paths"],
                "audit_skip_batches": int(skip_batches),
                "paired_target_access": False,
            }
        finally:
            restore_point(parent_rng, parent_a, parent_b)

    def audit_parent_compatibility(self, count: int = 8) -> dict:
        rows = [
            self._transactional_gradient_geometry(skip_batches=2 * index)
            for index in range(int(count))
        ]
        informative = [
            row for row in rows if row["correction_gradient_norm"] > 1e-12
        ]
        samples = [
            row["future_native_correction_cosine"] for row in informative
        ]
        sequence = empirical_bernstein_cs(samples)
        return {
            "schema": "clean-unsb-search004-parent-compatibility-v2",
            "checkpoint_id": self.checkpoint.checkpoint_id,
            "requested_observations": int(count),
            "informative_observations": len(informative),
            "samples": samples,
            "confidence_sequence": sequence.__dict__,
            "persistently_incompatible": sequence.valid and sequence.upper < 0.0,
            "observations": rows,
            "paired_target_access": False,
            "confirmation20_opened": False,
        }

    def _run_g2_full_selector(
        self,
        *,
        horizon: int,
        data_root: Path,
        eval_count: int,
        eval_start: int,
        include_lpips: bool,
        save_state: Path | None,
        evaluation_horizons: tuple[int, ...] | None,
    ) -> dict:
        """Continue a G2 full checkpoint with its actual eight-step selector.

        Treating this source as continuously-on HJ would change the causal arm:
        the original intervention is the SEARCH-003 transactional selector, not
        the proposal branch it sometimes commits.  Block boundaries therefore
        define the first committed state (h=8); h=1 is explicitly unavailable.
        """
        receding = load_receding()
        spec = receding.CANDIDATES["G2-HJ-FBDFC8"]
        selector = receding.RecedingEngine(
            spec=spec,
            rows=self.rows,
            train_view=self.train_view,
            work_dir=self.work_dir / "g2_exact" / self.checkpoint.checkpoint_id,
            per_domain=self.checkpoint.per_domain,
            seed=self.seed,
            gpu=self.gpu,
            target_steps=self.checkpoint.step + int(horizon),
        )
        parent_before = torch_digest(self.method_payload)
        state = cpu_clone(self.method_payload)
        selected_evaluations = set(
            evaluation_horizons
            if evaluation_horizons is not None
            else (32, 200, horizon)
        )
        diagnostics: list[dict] = [{
            "horizon": 1,
            "defined": False,
            "reason": "selector commits only at its preregistered 8-update boundary",
        }]
        evaluations: list[dict] = []
        decision_start = len(state.get("decisions", []))
        started = time.time()
        try:
            while int(state["step"]) < self.checkpoint.step + int(horizon):
                state = selector.advance(state, mode="full")
                completed = int(state["step"]) - self.checkpoint.step
                if completed in {8, 32, 200, horizon}:
                    decision = state["decisions"][-1]
                    diagnostics.append({
                        "horizon": completed,
                        "defined": True,
                        "committed": decision["committed"],
                        "signed_margin": decision["signed_margin"],
                        "plain_value": decision["plain_value"],
                        "proposal_value": decision["proposal_value"],
                        "future_native_descent_cosine": (
                            decision.get("future_native_consensus") or {}
                        ).get("descent_cosine"),
                    })
                if completed in selected_evaluations and completed <= horizon:
                    saved_rng = selector.runtime.capture_rng()
                    selector.load_for_evaluation(state)
                    metrics = self.evaluator.evaluate(
                        selector.model,
                        rows=self.rows,
                        data_root=Path(data_root),
                        start_per_domain=eval_start,
                        count_per_domain=eval_count,
                        eval_seed=self.seed,
                        include_lpips=include_lpips and completed == horizon,
                    )
                    selector.runtime.restore_rng(saved_rng)
                    evaluations.append({"horizon": completed, **metrics})
            if save_state is not None:
                atomic_torch(save_state, state)
            new_decisions = state.get("decisions", [])[decision_start:]
            parent_after = torch_digest(self.method_payload)
            result = {
                "schema": "clean-unsb-search004-continuation-v1",
                "diagnostic_schema": "exact-search003-selector-v1",
                "checkpoint": self.checkpoint.to_dict(),
                "arm": "U_uninterrupted",
                "horizon": int(horizon),
                "diagnostics": diagnostics,
                "final_component_diagnostics": {
                    "selector_blocks": len(new_decisions),
                    "proposal_commits": sum(
                        row.get("committed") == "proposal" for row in new_decisions
                    ),
                    "plain_commits": sum(
                        row.get("committed") == "plain" for row in new_decisions
                    ),
                },
                "evaluations": evaluations,
                "initial_state_digest": torch_digest(self.method_payload),
                "final_state_digest": torch_digest(state),
                "parent_digest_before": parent_before,
                "parent_digest_after": parent_after,
                "parent_immutable": parent_before == parent_after == self.parent_digest,
                "continuation_semantics": "exact_search003_g2_full_selector",
                "paired_target_access_by_operator": False,
                "wall_seconds": time.time() - started,
                "confirmation20_opened": False,
            }
            if not result["parent_immutable"]:
                raise RuntimeError("G2 continuation mutated its parent checkpoint")
            return result
        finally:
            selector.close()

    def evaluate_exact_historical_endpoint(
        self,
        *,
        arm: str,
        horizon: int,
        endpoint: Path,
        data_root: Path,
        eval_count: int = 70,
        eval_start: int = 10,
        include_lpips: bool = True,
    ) -> dict:
        """Evaluate an explicit same-lineage future state without retraining."""
        endpoint = Path(endpoint)
        payload = load_payload(endpoint)
        expected_step = self.checkpoint.step + int(horizon)
        if int(payload["step"]) != expected_step:
            raise RuntimeError(
                f"historical endpoint step mismatch: {payload['step']} != {expected_step}"
            )
        parent_before = torch_digest(self.method_payload)
        self.runtime.load_model_state(self.model, payload["model"], load_extra=False)
        self._reset_extra(checkpoint_method_costate(payload))
        saved_rng = self.runtime.capture_rng()
        started = time.time()
        metrics = self.evaluator.evaluate(
            self.model,
            rows=self.rows,
            data_root=Path(data_root),
            start_per_domain=eval_start,
            count_per_domain=eval_count,
            eval_seed=self.seed,
            include_lpips=include_lpips,
        )
        self.runtime.restore_rng(saved_rng)
        parent_after = torch_digest(self.method_payload)
        if parent_before != parent_after or parent_after != self.parent_digest:
            raise RuntimeError("historical endpoint evaluation mutated parent")
        return {
            "schema": "clean-unsb-search004-continuation-v1",
            "diagnostic_schema": "imported-exact-lineage-endpoint-v1",
            "checkpoint": self.checkpoint.to_dict(),
            "arm": arm,
            "horizon": int(horizon),
            "diagnostics": [],
            "final_component_diagnostics": {},
            "evaluations": [{"horizon": int(horizon), **metrics}],
            "initial_state_digest": torch_digest(
                self.plain_payload if arm == "P_common_plain" else self.method_payload
            ),
            "final_state_digest": torch_digest(payload),
            "parent_digest_before": parent_before,
            "parent_digest_after": parent_after,
            "parent_immutable": True,
            "continuation_semantics": "exact_historical_same_lineage_endpoint",
            "imported_from": str(endpoint),
            "paired_target_access_by_operator": False,
            "wall_seconds": time.time() - started,
            "confirmation20_opened": False,
        }

    def run_arm(
        self,
        *,
        arm: str,
        horizon: int,
        protocol: Search004Protocol,
        data_root: Path,
        eval_count: int,
        eval_start: int,
        include_lpips: bool,
        save_state: Path | None = None,
        evaluation_horizons: tuple[int, ...] | None = None,
    ) -> dict:
        if arm == "U_uninterrupted" and self.checkpoint.source_mode == "g2_full":
            return self._run_g2_full_selector(
                horizon=horizon,
                data_root=data_root,
                eval_count=eval_count,
                eval_start=eval_start,
                include_lpips=include_lpips,
                save_state=save_state,
                evaluation_horizons=evaluation_horizons,
            )
        parent_before = torch_digest(self.method_payload)
        initial = self.prepare_arm(arm, protocol)
        self.load_state(initial)
        diagnostics = []
        evaluations = []
        selected_evaluations = set(
            evaluation_horizons
            if evaluation_horizons is not None
            else (32, 200, horizon)
        )
        started = time.time()
        active = arm == "U_uninterrupted"
        for offset in range(int(horizon)):
            global_step = self.checkpoint.step + offset
            self.current_step = global_step
            self.model.set_train_epoch(1 + global_step // self.steps_per_epoch)
            if active:
                _force_active(self.model, self.checkpoint.family, global_step)
            else:
                self.model.set_search_step(global_step, self.checkpoint.step + horizon)
            self.model.set_input(self.stream_a.next(), self.stream_b.next())
            in_hold = offset < protocol.equilibration_steps
            with self.operator(active=active):
                if arm == "D0_hold_only" and in_hold:
                    self._hold_step(equilibrate=False)
                elif arm in {"D_costate_equilibration", "E_combined"} and in_hold:
                    self._hold_step(equilibrate=True)
                else:
                    self.model.optimize_parameters()
            completed = offset + 1
            self.current_step = global_step + 1
            if self.current_step % self.steps_per_epoch == 0:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.model.update_learning_rate()
            if completed in {1, 8, 32, 200, horizon}:
                step_row = {
                    "horizon": completed,
                    **self._step_diagnostics(),
                }
                if completed in {1, 8, 32}:
                    step_row["future_unpaired_gradient_geometry"] = (
                        self._transactional_gradient_geometry()
                    )
                diagnostics.append(step_row)
            if completed in selected_evaluations and completed <= horizon:
                saved_rng = self.runtime.capture_rng()
                metrics = self.evaluator.evaluate(
                    self.model,
                    rows=self.rows,
                    data_root=Path(data_root),
                    start_per_domain=eval_start,
                    count_per_domain=eval_count,
                    eval_seed=self.seed,
                    include_lpips=include_lpips and completed == horizon,
                )
                self.runtime.restore_rng(saved_rng)
                evaluations.append({"horizon": completed, **metrics})
        final = self.capture_state(arm=arm, completed=horizon)
        final_summary = self._step_diagnostics()
        if save_state is not None:
            atomic_torch(save_state, final)
        parent_after = torch_digest(self.method_payload)
        result = {
            "schema": "clean-unsb-search004-continuation-v1",
            "diagnostic_schema": "future-unpaired-gradient-geometry-v1",
            "checkpoint": self.checkpoint.to_dict(),
            "arm": arm,
            "horizon": int(horizon),
            "diagnostics": diagnostics,
            "final_component_diagnostics": final_summary,
            "evaluations": evaluations,
            "initial_state_digest": torch_digest(initial),
            "final_state_digest": torch_digest(final),
            "parent_digest_before": parent_before,
            "parent_digest_after": parent_after,
            "parent_immutable": parent_before == parent_after == self.parent_digest,
            "paired_target_access_by_operator": False,
            "transport_record": cpu_clone(self._transport_record),
            "wall_seconds": time.time() - started,
            "confirmation20_opened": False,
        }
        if not result["parent_immutable"]:
            raise RuntimeError("continuation arm mutated its parent checkpoint")
        return result


def compare_component_defect(method: dict, plain: dict) -> float:
    values = []
    for name in ("D", "E", "F"):
        key = f"{name}_relative_step_norm"
        left = max(abs(float(method.get(key, 0.0))), 1e-30)
        right = max(abs(float(plain.get(key, 0.0))), 1e-30)
        values.append(abs(math.log(left / right)))
    return float(sum(values) / len(values))
