"""Search-scoped model installers for Generation-1 update operators."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import types

import torch

from .operators import (
    acmp_project,
    block_confidence_precondition,
    brownian_antithetic_variance_projection,
    cndrp_precondition,
    pathwise_horizon_residual_projection,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
HJ_ROOT = REPO_ROOT / "research" / "candidates" / "CAND-002-hj-patchnce"
if str(HJ_ROOT) not in sys.path:
    sys.path.insert(0, str(HJ_ROOT))


def _capture_torch_rng() -> dict:
    return {
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_torch_rng(state: dict) -> None:
    torch.random.set_rng_state(state["cpu"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _dispersion_log_statistic(model, x_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    positive = model.netG(x_t, model.time_idx, z)
    negative = model.netG(x_t, model.time_idx, -z)
    midpoint = 0.5 * (positive + negative)
    half_difference = 0.5 * (positive - negative)
    dispersion = half_difference.square().mean()
    signal = (midpoint - x_t).square().mean()
    return torch.log(1e-6 + dispersion / (signal + 1e-6))


def _sensitivity(model, parameters, x_t, z):
    statistic = _dispersion_log_statistic(model, x_t, z)
    gradients = torch.autograd.grad(
        statistic,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return statistic.detach(), tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients)
    )


def cndrp_optimize_parameters(self):
    if not bool(getattr(self, "search005_cndrp_active", True)):
        return self._search005_cndrp_original_optimize_parameters()

    self.forward()
    self.netG.train()
    self.netE.train()
    self.netD.train()
    self.netF.train()

    self.set_requires_grad(self.netD, True)
    self.optimizer_D.zero_grad()
    self.loss_D = self.compute_D_loss()
    self.loss_D.backward()
    self.optimizer_D.step()

    self.set_requires_grad(self.netE, True)
    self.optimizer_E.zero_grad()
    self.loss_E = self.compute_E_loss()
    self.loss_E.backward()
    self.optimizer_E.step()

    self.set_requires_grad(self.netD, False)
    self.set_requires_grad(self.netE, False)
    self.optimizer_G.zero_grad()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.zero_grad()
    self.loss_G = self.compute_G_loss()
    self.loss_G.backward()

    parameters = tuple(self.netG.parameters())
    native_gradients = tuple(
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )

    # Probe randomness is isolated from the native UNSB stream and reconstructed
    # solely from the saved operator step.
    main_rng = _capture_torch_rng()
    try:
        probe_seed = int(self._search005_cndrp_seed) + 1_000_003 * int(
            self._search005_cndrp_step
        )
        torch.manual_seed(probe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(probe_seed)
        latent_shape = (self.real_A.size(0), 4 * int(self.opt.ngf))
        first_z = torch.randn(latent_shape, device=self.real_A.device)
        second_z = torch.randn(latent_shape, device=self.real_A.device)
        x_t = self.real_A_noisy.detach()
        first_stat, first_sensitivity = _sensitivity(
            self, parameters, x_t, first_z
        )
        second_stat, second_sensitivity = _sensitivity(
            self, parameters, x_t, second_z
        )
    finally:
        _restore_torch_rng(main_rng)

    weighted_scale = 0.0
    total_values = 0
    minimum_scale = 1.0
    native_sq = 0.0
    changed_sq = 0.0
    for parameter, native, first, second in zip(
        parameters, native_gradients, first_sensitivity, second_sensitivity
    ):
        preconditioner = (
            block_confidence_precondition
            if bool(getattr(self, "_search005_cndrp_blockwise", False))
            else cndrp_precondition
        )
        transformed, diagnostics = preconditioner(native, first, second)
        parameter.grad = transformed
        count = int(parameter.numel())
        weighted_scale += diagnostics["mean_scale"] * count
        total_values += count
        minimum_scale = min(minimum_scale, diagnostics["minimum_scale"])
        native_sq += float(native.square().sum().item())
        changed_sq += float((transformed - native).square().sum().item())

    self.optimizer_G.step()
    self._update_lbst_teacher()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.step()

    self._search005_cndrp_last = {
        "step": int(self._search005_cndrp_step),
        "log_dispersion_first": float(first_stat.item()),
        "log_dispersion_second": float(second_stat.item()),
        "mean_scale": weighted_scale / max(total_values, 1),
        "minimum_scale": minimum_scale,
        "relative_gradient_change": (changed_sq / max(native_sq, 1e-30)) ** 0.5,
    }
    self._search005_cndrp_step += 1


def install_cndrp(model, *, seed: int, blockwise: bool = False) -> None:
    if hasattr(model, "_search005_cndrp_original_optimize_parameters"):
        raise RuntimeError("CNDRP is already installed")
    model._search005_cndrp_original_optimize_parameters = model.optimize_parameters
    model._search005_cndrp_original_get_extra_training_state = (
        model.get_extra_training_state
    )
    model._search005_cndrp_original_load_extra_training_state = (
        model.load_extra_training_state
    )
    model._search005_cndrp_seed = int(seed)
    model._search005_cndrp_step = 0
    model._search005_cndrp_last = {}
    model._search005_cndrp_blockwise = bool(blockwise)
    model.search005_cndrp_active = True

    def get_extra_training_state(self):
        state = self._search005_cndrp_original_get_extra_training_state()
        state["search005_cndrp"] = {
            "seed": int(self._search005_cndrp_seed),
            "step": int(self._search005_cndrp_step),
            "last": copy.deepcopy(self._search005_cndrp_last),
            "active": bool(self.search005_cndrp_active),
            "blockwise": bool(self._search005_cndrp_blockwise),
        }
        return state

    def load_extra_training_state(self, state):
        self._search005_cndrp_original_load_extra_training_state(state)
        saved = (state or {}).get("search005_cndrp")
        if saved is None:
            raise RuntimeError("CNDRP checkpoint is missing operator state")
        if int(saved["seed"]) != int(self._search005_cndrp_seed):
            raise RuntimeError("CNDRP seed mismatch")
        if bool(saved.get("blockwise", False)) != bool(self._search005_cndrp_blockwise):
            raise RuntimeError("CNDRP blockwise mode mismatch")
        self._search005_cndrp_step = int(saved["step"])
        self._search005_cndrp_last = copy.deepcopy(saved.get("last", {}))
        self.search005_cndrp_active = bool(saved.get("active", True))

    model.optimize_parameters = types.MethodType(cndrp_optimize_parameters, model)
    model.get_extra_training_state = types.MethodType(get_extra_training_state, model)
    model.load_extra_training_state = types.MethodType(load_extra_training_state, model)


def set_cndrp_active(model, active: bool) -> None:
    if not hasattr(model, "_search005_cndrp_original_optimize_parameters"):
        raise RuntimeError("CNDRP is not installed")
    model.search005_cndrp_active = bool(active)


def _hj_layer_correction(model, src, tgt, z):
    """HJ-minus-plain PatchNCE loss for the audited layers at one latent."""
    from hj.core import structure_project_nce_step

    feat_q = model.netG(
        tgt, model.time_idx * 0, z, model.nce_layers, encode_only=True
    )
    if model.opt.flip_equivariance and model.flipped_for_equivariance:
        feat_q = [torch.flip(value, [3]) for value in feat_q]
    feat_k = model.netG(
        src, model.time_idx * 0, z, model.nce_layers, encode_only=True
    )
    feat_k_pool, sample_ids = model.netF(
        feat_k, model.opt.num_patches, None
    )
    feat_q_pool, _ = model.netF(
        feat_q, model.opt.num_patches, sample_ids
    )
    correction = feat_q_pool[0].new_zeros(())
    diagnostics = []
    for f_q, f_k, criterion, layer in zip(
        feat_q_pool, feat_k_pool, model.criterionNCE, model.nce_layers
    ):
        if int(layer) not in model.hj_layers:
            continue
        def probe_fn(tgt_img):
            values = model.netG(
                tgt_img, model.time_idx * 0, z,
                model.nce_layers, encode_only=True,
            )
            if model.opt.flip_equivariance and model.flipped_for_equivariance:
                values = [torch.flip(value, [3]) for value in values]
            pooled, _ = model.netF(values, model.opt.num_patches, sample_ids)
            return pooled[model.nce_layers.index(int(layer))]

        projected, diag = structure_project_nce_step(
            feat_q=f_q,
            feat_k=f_k,
            criterion=criterion,
            source=src,
            tgt_nce=tgt,
            probe_fn=probe_fn,
            batch_size=model.real_A.size(0),
            cfg=model.hj_config,
            lambda_nce=model.opt.lambda_NCE,
            schedule_weight=1.0,
        )
        plain = (criterion(f_q, f_k) * model.opt.lambda_NCE).mean()
        correction = correction + projected - plain
        diagnostics.append(diag)
    return correction / len(model.nce_layers), diagnostics


def _hj_correction_loss(model, main_z, identity_z):
    main, main_diag = _hj_layer_correction(
        model, model.real_A, model.fake_B, main_z
    )
    if model.opt.nce_idt and model.opt.lambda_NCE > 0.0:
        identity, identity_diag = _hj_layer_correction(
            model, model.real_B, model.idt_B, identity_z
        )
        return 0.5 * (main + identity), main_diag + identity_diag
    return main, main_diag


def _gradients(loss, parameters, *, retain_graph):
    values = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if value is None else value.detach()
        for parameter, value in zip(parameters, values)
    )


def _adam_metric(optimizer, parameter):
    state = optimizer.state.get(parameter, {})
    variance = state.get("exp_avg_sq")
    if variance is None:
        return torch.ones_like(parameter)
    eps = float(optimizer.param_groups[0].get("eps", 1e-8))
    return variance.detach().sqrt().add(eps).reciprocal()


def acmp_optimize_parameters(self):
    if not bool(getattr(self, "search005_acmp_active", True)):
        return self._search005_acmp_original_optimize_parameters()

    self.forward()
    self.netG.train()
    self.netE.train()
    self.netD.train()
    self.netF.train()

    self.set_requires_grad(self.netD, True)
    self.optimizer_D.zero_grad()
    self.loss_D = self.compute_D_loss()
    self.loss_D.backward()
    self.optimizer_D.step()

    self.set_requires_grad(self.netE, True)
    self.optimizer_E.zero_grad()
    self.loss_E = self.compute_E_loss()
    self.loss_E.backward()
    self.optimizer_E.step()

    self.set_requires_grad(self.netD, False)
    self.set_requires_grad(self.netE, False)
    self.optimizer_G.zero_grad()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.zero_grad()
    self.loss_G = self.compute_G_loss()
    parameters = tuple(self.netG.parameters())
    bridge_loss = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB
    bridge_gradients = _gradients(bridge_loss, parameters, retain_graph=True)

    main_rng = _capture_torch_rng()
    try:
        probe_seed = int(self._search005_acmp_seed) + 1_000_003 * int(
            self._search005_acmp_step
        )
        torch.manual_seed(probe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(probe_seed)
        latent_shape = (self.real_A.size(0), 4 * int(self.opt.ngf))
        main_z = torch.randn(latent_shape, device=self.real_A.device)
        identity_z = torch.randn(latent_shape, device=self.real_A.device)
        patch_rng = _capture_torch_rng()
        positive_loss, positive_diag = _hj_correction_loss(
            self, main_z, identity_z
        )
        positive_gradients = _gradients(
            positive_loss, parameters, retain_graph=True
        )
        del positive_loss
        _restore_torch_rng(patch_rng)
        negative_loss, negative_diag = _hj_correction_loss(
            self, -main_z, -identity_z
        )
        negative_gradients = _gradients(
            negative_loss, parameters, retain_graph=True
        )
        del negative_loss
    finally:
        _restore_torch_rng(main_rng)

    raw_corrections = tuple(
        0.5 * (positive + negative)
        for positive, negative in zip(positive_gradients, negative_gradients)
    )
    positive_sq = sum(float(value.square().sum().item()) for value in positive_gradients)
    negative_sq = sum(float(value.square().sum().item()) for value in negative_gradients)
    raw_sq = sum(float(value.square().sum().item()) for value in raw_corrections)
    antithetic_ratio = raw_sq / max(0.5 * (positive_sq + negative_sq), 1e-30)
    del positive_gradients, negative_gradients

    self.loss_G.backward()
    proposal_corrections = raw_corrections
    future_cosine = None
    future_weight = 1.0
    if bool(getattr(self, "_search005_acmp_future_consensus", False)):
        previous = self._search005_acmp_previous
        if previous is None:
            proposal_corrections = tuple(torch.zeros_like(value) for value in raw_corrections)
            future_weight = 0.0
        else:
            previous_device = tuple(
                value.to(device=parameter.device, dtype=parameter.dtype)
                for value, parameter in zip(previous, parameters)
            )
            dot = raw_corrections[0].new_zeros(())
            previous_norm_sq = raw_corrections[0].new_zeros(())
            current_norm_sq = raw_corrections[0].new_zeros(())
            for parameter, old, current in zip(
                parameters, previous_device, raw_corrections
            ):
                metric = _adam_metric(self.optimizer_G, parameter)
                dot = dot + torch.sum(old * metric * current)
                previous_norm_sq = previous_norm_sq + torch.sum(old * metric * old)
                current_norm_sq = current_norm_sq + torch.sum(current * metric * current)
            denominator = (previous_norm_sq * current_norm_sq).clamp_min(0).sqrt()
            cosine = dot / denominator.clamp_min(1e-12)
            weight = cosine.clamp(min=0.0, max=1.0)
            future_cosine = float(cosine.detach().item())
            future_weight = float(weight.detach().item())
            proposal_corrections = tuple(
                0.5 * weight * (old + current)
                for old, current in zip(previous_device, raw_corrections)
            )
            del previous_device

    active_counts = {"native": 0, "bridge_adversarial": 0}
    native_alignment = 0.0
    bridge_alignment = 0.0
    raw_metric_sq = 0.0
    projected_metric_sq = 0.0
    trust_scale_weighted = 0.0
    total_parameters = 0
    for parameter, bridge, raw in zip(
        parameters, bridge_gradients, proposal_corrections
    ):
        native = (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        metric = _adam_metric(self.optimizer_G, parameter)
        projected, diag = acmp_project(raw, native, bridge, metric)
        parameter.grad = native + projected
        for label in diag.active_constraints:
            active_counts[label] += 1
        native_alignment += diag.native_alignment
        bridge_alignment += diag.bridge_alignment
        raw_metric_sq += diag.raw_norm ** 2
        projected_metric_sq += diag.projected_norm ** 2
        count = int(parameter.numel())
        trust_scale_weighted += diag.trust_scale * count
        total_parameters += count

    self.optimizer_G.step()
    self._update_lbst_teacher()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.step()

    all_diags = positive_diag + negative_diag
    self._search005_acmp_last = {
        "step": int(self._search005_acmp_step),
        "antithetic_energy_ratio": antithetic_ratio,
        "future_batch_cosine": future_cosine,
        "future_consensus_weight": future_weight,
        "raw_metric_norm": raw_metric_sq ** 0.5,
        "projected_metric_norm": projected_metric_sq ** 0.5,
        "native_metric_alignment": native_alignment,
        "bridge_metric_alignment": bridge_alignment,
        "mean_trust_scale": trust_scale_weighted / max(total_parameters, 1),
        "active_constraint_tensors": active_counts,
        "mean_gate_active": (
            sum(float(row["gate_active"]) for row in all_diags)
            / max(len(all_diags), 1)
        ),
        "mean_probe_agreement": (
            sum(float(row["probe_agreement"]) for row in all_diags)
            / max(len(all_diags), 1)
        ),
    }
    if bool(getattr(self, "_search005_acmp_future_consensus", False)):
        self._search005_acmp_previous = tuple(
            value.detach().cpu().clone() for value in raw_corrections
        )
    self._search005_acmp_step += 1


def install_acmp(model, *, seed: int, future_consensus: bool = False) -> None:
    if hasattr(model, "_search005_acmp_original_optimize_parameters"):
        raise RuntimeError("ACMP is already installed")
    model._search005_acmp_original_optimize_parameters = model.optimize_parameters
    model._search005_acmp_original_get_extra_training_state = model.get_extra_training_state
    model._search005_acmp_original_load_extra_training_state = model.load_extra_training_state
    model._search005_acmp_seed = int(seed)
    model._search005_acmp_step = 0
    model._search005_acmp_last = {}
    model._search005_acmp_future_consensus = bool(future_consensus)
    model._search005_acmp_previous = None
    model.search005_acmp_active = True
    from hj.core import StructureProjectConfig

    model.hj_layers = [0]
    model.hj_config = StructureProjectConfig(
        direction="joint",
        scales="1,2,4",
        step=0.01,
        quantile=0.75,
        gate_quantile=0.75,
        strength=0.5,
        boundary_scale=0.001,
        min_risk=0.05,
        min_delta=0.0,
        probe_mode="central_consensus",
        control="true",
        amplitude="constant",
        update_mode="remove",
        start_epoch=5,
        direction_alpha=0.0,
        random_seed=int(seed),
    )

    def get_extra_training_state(self):
        state = self._search005_acmp_original_get_extra_training_state()
        state["search005_acmp"] = {
            "seed": int(self._search005_acmp_seed),
            "step": int(self._search005_acmp_step),
            "last": copy.deepcopy(self._search005_acmp_last),
            "active": bool(self.search005_acmp_active),
            "future_consensus": bool(self._search005_acmp_future_consensus),
            "previous_correction": self._search005_acmp_previous,
        }
        return state

    def load_extra_training_state(self, state):
        self._search005_acmp_original_load_extra_training_state(state)
        saved = (state or {}).get("search005_acmp")
        if saved is None:
            raise RuntimeError("ACMP checkpoint is missing operator state")
        if int(saved["seed"]) != int(self._search005_acmp_seed):
            raise RuntimeError("ACMP seed mismatch")
        if bool(saved.get("future_consensus", False)) != bool(
            self._search005_acmp_future_consensus
        ):
            raise RuntimeError("ACMP future-consensus mode mismatch")
        self._search005_acmp_step = int(saved["step"])
        self._search005_acmp_last = copy.deepcopy(saved.get("last", {}))
        self.search005_acmp_active = bool(saved.get("active", True))
        previous = saved.get("previous_correction")
        self._search005_acmp_previous = (
            None
            if previous is None
            else tuple(value.detach().cpu().clone() for value in previous)
        )

    model.optimize_parameters = types.MethodType(acmp_optimize_parameters, model)
    model.get_extra_training_state = types.MethodType(get_extra_training_state, model)
    model.load_extra_training_state = types.MethodType(load_extra_training_state, model)


def set_acmp_active(model, active: bool) -> None:
    if not hasattr(model, "_search005_acmp_original_optimize_parameters"):
        raise RuntimeError("ACMP is not installed")
    model.search005_acmp_active = bool(active)


def _make_bcavp_forward():
    def bcavp_forward(self, x, time_cond, z, layers=None, encode_only=False):
        requested_layers = [] if layers is None else layers
        if requested_layers or not bool(self._search005_bcavp_active):
            return self._search005_bcavp_original_forward(
                x, time_cond, z, requested_layers, encode_only
            )

        # Replaying the same RNG bundle for -z gives both endpoints the same
        # dropout mask and restores the main stream to exactly the +z state.
        rng_before = _capture_torch_rng()
        positive = self._search005_bcavp_original_forward(
            x, time_cond, z, requested_layers, encode_only
        )
        rng_after_positive = _capture_torch_rng()
        try:
            _restore_torch_rng(rng_before)
            negative = self._search005_bcavp_original_forward(
                x, time_cond, -z, requested_layers, encode_only
            )
        finally:
            _restore_torch_rng(rng_after_positive)

        from models.hnek.hnek_kernel import horizon_from_condition

        horizon = horizon_from_condition(
            time_cond,
            num_timesteps=int(self._search005_bcavp_num_timesteps),
        )
        endpoint, diagnostics = brownian_antithetic_variance_projection(
            positive,
            negative,
            horizon,
            tau=float(self._search005_bcavp_tau),
        )
        self._search005_bcavp_generator_last = {
            "active_fraction": diagnostics.active_fraction,
            "mean_scale": diagnostics.mean_scale,
            "minimum_scale": diagnostics.minimum_scale,
            "maximum_scale": diagnostics.maximum_scale,
            "mean_variance": diagnostics.mean_variance,
            "mean_cap": diagnostics.mean_cap,
            "maximum_projected_ratio": diagnostics.maximum_projected_ratio,
            "physical_horizon": float(horizon.detach().float().mean().item()),
        }
        return endpoint

    return bcavp_forward


def install_bcavp(model) -> None:
    """Install the search-scoped Brownian antithetic variance projection."""
    generator = model.netG.module if hasattr(model.netG, "module") else model.netG
    if hasattr(generator, "_search005_bcavp_original_forward"):
        raise RuntimeError("BCAVP is already installed")
    before_keys = tuple(generator.state_dict().keys())
    before_count = sum(parameter.numel() for parameter in generator.parameters())
    generator._search005_bcavp_original_forward = generator.forward
    generator._search005_bcavp_num_timesteps = int(model.opt.num_timesteps)
    generator._search005_bcavp_tau = float(model.opt.tau)
    generator._search005_bcavp_active = True
    generator._search005_bcavp_generator_last = {}
    generator.forward = types.MethodType(_make_bcavp_forward(), generator)
    if tuple(generator.state_dict().keys()) != before_keys:
        raise RuntimeError("BCAVP changed generator state keys")
    if sum(parameter.numel() for parameter in generator.parameters()) != before_count:
        raise RuntimeError("BCAVP changed generator parameter count")

    model._search005_bcavp_original_optimize_parameters = model.optimize_parameters
    model._search005_bcavp_original_get_extra_training_state = model.get_extra_training_state
    model._search005_bcavp_original_load_extra_training_state = model.load_extra_training_state
    model._search005_bcavp_last = {}

    def optimize_parameters(self):
        self._search005_bcavp_original_optimize_parameters()
        inner = self.netG.module if hasattr(self.netG, "module") else self.netG
        self._search005_bcavp_last = copy.deepcopy(
            inner._search005_bcavp_generator_last
        )

    def get_extra_training_state(self):
        state = self._search005_bcavp_original_get_extra_training_state()
        inner = self.netG.module if hasattr(self.netG, "module") else self.netG
        state["search005_bcavp"] = {
            "active": bool(inner._search005_bcavp_active),
            "last": copy.deepcopy(self._search005_bcavp_last),
            "tau": float(inner._search005_bcavp_tau),
            "num_timesteps": int(inner._search005_bcavp_num_timesteps),
        }
        return state

    def load_extra_training_state(self, state):
        self._search005_bcavp_original_load_extra_training_state(state)
        saved = (state or {}).get("search005_bcavp")
        if saved is None:
            raise RuntimeError("BCAVP checkpoint is missing operator state")
        inner = self.netG.module if hasattr(self.netG, "module") else self.netG
        if float(saved["tau"]) != float(inner._search005_bcavp_tau):
            raise RuntimeError("BCAVP tau mismatch")
        if int(saved["num_timesteps"]) != int(inner._search005_bcavp_num_timesteps):
            raise RuntimeError("BCAVP timestep mismatch")
        inner._search005_bcavp_active = bool(saved.get("active", True))
        self._search005_bcavp_last = copy.deepcopy(saved.get("last", {}))
        inner._search005_bcavp_generator_last = copy.deepcopy(
            self._search005_bcavp_last
        )

    model.optimize_parameters = types.MethodType(optimize_parameters, model)
    model.get_extra_training_state = types.MethodType(get_extra_training_state, model)
    model.load_extra_training_state = types.MethodType(load_extra_training_state, model)


def set_bcavp_active(model, active: bool) -> None:
    generator = model.netG.module if hasattr(model.netG, "module") else model.netG
    if not hasattr(generator, "_search005_bcavp_original_forward"):
        raise RuntimeError("BCAVP is not installed")
    generator._search005_bcavp_active = bool(active)


def bcavp_installation_status(model) -> dict:
    generator = model.netG.module if hasattr(model.netG, "module") else model.netG
    return {
        "installed": hasattr(generator, "_search005_bcavp_original_forward"),
        "active": bool(getattr(generator, "_search005_bcavp_active", False)),
        "tau": float(getattr(generator, "_search005_bcavp_tau", float("nan"))),
        "num_timesteps": int(getattr(generator, "_search005_bcavp_num_timesteps", -1)),
        "parameter_count": sum(parameter.numel() for parameter in generator.parameters()),
    }


def _phcrp_energy(state: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
    return (endpoint - state).square().mean(dim=tuple(range(1, state.ndim)))


def _phcrp_project(self, state, native, horizon, history):
    previous_h = None if history is None else history["horizon"]
    previous_q = None if history is None else history["accepted_energy"]
    endpoint, accepted, diagnostics = pathwise_horizon_residual_projection(
        state, native, horizon, previous_h, previous_q
    )
    raw = _phcrp_energy(state, native)
    tiny = torch.finfo(state.dtype).tiny
    scale = torch.where(
        accepted < raw,
        torch.sqrt(accepted / raw.clamp_min(tiny)),
        torch.ones_like(raw),
    ).detach()
    next_history = {
        "horizon": horizon.detach().reshape(-1),
        "accepted_energy": accepted.detach().reshape(-1),
        "scale": scale,
    }
    self._search005_phcrp_last = {
        "active_fraction": diagnostics.active_fraction,
        "mean_scale": diagnostics.mean_scale,
        "minimum_scale": diagnostics.minimum_scale,
        "maximum_scale": diagnostics.maximum_scale,
        "mean_raw_energy": diagnostics.mean_raw_energy,
        "mean_accepted_energy": diagnostics.mean_accepted_energy,
        "mean_physical_cap": diagnostics.mean_physical_cap,
        "maximum_cap_ratio": diagnostics.maximum_cap_ratio,
        "physical_horizon": float(horizon.detach().float().mean().item()),
    }
    return endpoint, next_history


def _phcrp_apply_recorded_scale(self, state, native, histories):
    scales = torch.cat([history["scale"] for history in histories], dim=0).to(
        device=state.device, dtype=state.dtype
    )
    view = scales.reshape((state.shape[0],) + (1,) * (state.ndim - 1))
    projected = state + view * (native - state)
    active = scales < 1
    active_view = active.reshape((state.shape[0],) + (1,) * (state.ndim - 1))
    result = torch.where(active_view, projected, native)
    raw = _phcrp_energy(state, native)
    accepted = raw * scales.square()
    self._search005_phcrp_last = {
        "active_fraction": float(active.float().mean().detach().item()),
        "mean_scale": float(scales.mean().detach().item()),
        "minimum_scale": float(scales.min().detach().item()),
        "maximum_scale": float(scales.max().detach().item()),
        "mean_raw_energy": float(raw.mean().detach().item()),
        "mean_accepted_energy": float(accepted.mean().detach().item()),
        "mean_physical_cap": float(accepted.mean().detach().item()),
        "maximum_cap_ratio": 1.0,
        "physical_horizon": float(
            torch.cat([history["horizon"] for history in histories]).float().mean().item()
        ),
    }
    return result


def phcrp_predict_endpoint(self, net, state, time_idx, z):
    if not bool(getattr(self, "search005_phcrp_active", True)):
        return self._search005_phcrp_original_predict_endpoint(net, state, time_idx, z)
    native = self._search005_phcrp_original_predict_endpoint(net, state, time_idx, z)
    from models.hnek.hnek_kernel import horizon_from_condition

    horizon = horizon_from_condition(
        time_idx, num_timesteps=int(self.opt.num_timesteps)
    ).to(device=state.device, dtype=state.dtype).reshape(-1)
    context = getattr(self, "_search005_phcrp_context", None)

    if context is not None and bool(context.get("inside_training_forward", False)):
        base_batch = int(self.real_A.shape[0])
        if not context["final_started"] and int(state.shape[0]) == 2 * base_batch:
            context["final_started"] = True
            return _phcrp_apply_recorded_scale(
                self, state, native, [context["histories"][0], context["histories"][2]]
            )
        if context["final_started"]:
            return _phcrp_apply_recorded_scale(
                self, state, native, [context["histories"][1]]
            )
        branch = int(context["rollout_calls"] % 3)
        endpoint, history = _phcrp_project(
            self, state, native, horizon, context["histories"][branch]
        )
        context["histories"][branch] = history
        context["rollout_calls"] += 1
        return endpoint

    # The frozen evaluator calls _predict_endpoint directly, one bridge point
    # at a time.  Index zero is the exact, intrinsic trajectory reset.
    index = int(time_idx.reshape(-1)[0].item())
    if index == 0:
        self._search005_phcrp_eval_history = None
    endpoint, history = _phcrp_project(
        self,
        state,
        native,
        horizon,
        getattr(self, "_search005_phcrp_eval_history", None),
    )
    self._search005_phcrp_eval_history = history
    return endpoint


def install_phcrp(model) -> None:
    """Install a stateless-per-trajectory physical residual projection."""
    if hasattr(model, "_search005_phcrp_original_predict_endpoint"):
        raise RuntimeError("PHCRP is already installed")
    model._search005_phcrp_original_predict_endpoint = model._predict_endpoint
    model._search005_phcrp_original_forward = model.forward
    model._search005_phcrp_original_get_extra_training_state = model.get_extra_training_state
    model._search005_phcrp_original_load_extra_training_state = model.load_extra_training_state
    model.search005_phcrp_active = True
    model._search005_phcrp_last = {}
    model._search005_phcrp_eval_history = None

    def forward(self):
        self._search005_phcrp_context = {
            "inside_training_forward": True,
            "rollout_calls": 0,
            "final_started": False,
            "histories": [None, None, None],
        }
        try:
            return self._search005_phcrp_original_forward()
        finally:
            self._search005_phcrp_context["inside_training_forward"] = False

    def get_extra_training_state(self):
        state = self._search005_phcrp_original_get_extra_training_state()
        state["search005_phcrp"] = {
            "active": bool(self.search005_phcrp_active),
            "last": copy.deepcopy(self._search005_phcrp_last),
        }
        return state

    def load_extra_training_state(self, state):
        self._search005_phcrp_original_load_extra_training_state(state)
        saved = (state or {}).get("search005_phcrp")
        if saved is None:
            raise RuntimeError("PHCRP checkpoint is missing operator state")
        self.search005_phcrp_active = bool(saved.get("active", True))
        self._search005_phcrp_last = copy.deepcopy(saved.get("last", {}))
        self._search005_phcrp_eval_history = None

    model._predict_endpoint = types.MethodType(phcrp_predict_endpoint, model)
    model.forward = types.MethodType(forward, model)
    model.get_extra_training_state = types.MethodType(get_extra_training_state, model)
    model.load_extra_training_state = types.MethodType(load_extra_training_state, model)


def set_phcrp_active(model, active: bool) -> None:
    if not hasattr(model, "_search005_phcrp_original_predict_endpoint"):
        raise RuntimeError("PHCRP is not installed")
    model.search005_phcrp_active = bool(active)


def phcrp_installation_status(model) -> dict:
    return {
        "installed": hasattr(model, "_search005_phcrp_original_predict_endpoint"),
        "active": bool(getattr(model, "search005_phcrp_active", False)),
        "trajectory_state_persistent_across_updates": False,
        "paired_target_access": False,
    }


def phrsup_predict_endpoint(self, net, state, time_idx, z):
    native = self._search005_phrsup_original_predict_endpoint(net, state, time_idx, z)
    if not bool(getattr(self, "search005_phrsup_active", True)):
        return native
    context = getattr(self, "_search005_phrsup_context", None)
    if context is None or not bool(context.get("inside_training_forward", False)):
        return native

    base_batch = int(self.real_A.shape[0])
    if not context["final_started"] and int(state.shape[0]) == 2 * base_batch:
        context["final_started"] = True
        return native
    if context["final_started"]:
        return native

    from models.hnek.hnek_kernel import horizon_from_condition

    horizon = horizon_from_condition(
        time_idx, num_timesteps=int(self.opt.num_timesteps)
    ).to(device=state.device, dtype=state.dtype).reshape(-1)
    energy = _phcrp_energy(state, native).detach()
    branch = int(context["rollout_calls"] % 3)
    previous = context["histories"][branch]
    cap = None
    if previous is not None:
        cap = previous["energy"] * (horizon / previous["horizon"])
    context["histories"][branch] = {
        "horizon": horizon.detach(),
        "energy": energy,
        "cap": None if cap is None else cap.detach(),
    }
    context["rollout_calls"] += 1
    return native


def _phrsup_defect_loss(model):
    histories = model._search005_phrsup_context["histories"]
    if any(history is None for history in histories):
        raise RuntimeError("PHRSUP rollout histories are incomplete")
    if all(history["cap"] is None for history in histories):
        zero = model.fake.sum() * 0.0
        return zero, {"active_fraction": 0.0, "mean_log_violation": 0.0}

    endpoints = (model.fake[0:1], model.fake_B2, model.fake[1:2])
    states = (model.realt[0:1], model.real_A_noisy2, model.realt[1:2])
    terms = []
    values = []
    for endpoint, state, history in zip(endpoints, states, histories):
        cap = history["cap"]
        if cap is None:
            value = _phcrp_energy(state, endpoint) * 0.0
        else:
            energy = _phcrp_energy(state, endpoint)
            tiny = torch.finfo(endpoint.dtype).tiny
            value = torch.relu(
                torch.log(energy.clamp_min(tiny))
                - torch.log(cap.to(energy).clamp_min(tiny))
            )
        terms.append(value.mean())
        values.append(value.detach().reshape(-1))
    stacked = torch.cat(values)
    return torch.stack(terms).mean(), {
        "active_fraction": float((stacked > 0).float().mean().item()),
        "mean_log_violation": float(stacked.mean().item()),
    }


def phrsup_optimize_parameters(self):
    if not bool(getattr(self, "search005_phrsup_active", True)):
        return self._search005_phrsup_original_optimize_parameters()

    self.forward()
    self.netG.train()
    self.netE.train()
    self.netD.train()
    self.netF.train()

    self.set_requires_grad(self.netD, True)
    self.optimizer_D.zero_grad()
    self.loss_D = self.compute_D_loss()
    self.loss_D.backward()
    self.optimizer_D.step()

    self.set_requires_grad(self.netE, True)
    self.optimizer_E.zero_grad()
    self.loss_E = self.compute_E_loss()
    self.loss_E.backward()
    self.optimizer_E.step()

    self.set_requires_grad(self.netD, False)
    self.set_requires_grad(self.netE, False)
    self.optimizer_G.zero_grad()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.zero_grad()
    self.loss_G = self.compute_G_loss()
    defect_loss, defect_observation = _phrsup_defect_loss(self)
    parameters = tuple(self.netG.parameters())
    defect_gradients = _gradients(defect_loss, parameters, retain_graph=True)
    self.loss_G.backward()
    native_gradients = tuple(
        torch.zeros_like(parameter)
        if parameter.grad is None
        else parameter.grad.detach().clone()
        for parameter in parameters
    )
    metrics = tuple(_adam_metric(self.optimizer_G, parameter) for parameter in parameters)
    alignment = sum(
        (defect * metric * native).sum()
        for defect, metric, native in zip(defect_gradients, metrics, native_gradients)
    )
    defect_norm = sum(
        (defect * metric * defect).sum()
        for defect, metric in zip(defect_gradients, metrics)
    )
    active = bool(alignment < 0 and defect_norm > 0)
    coefficient = alignment / defect_norm if active else alignment.new_zeros(())
    projected_gradients = []
    for parameter, native, defect in zip(parameters, native_gradients, defect_gradients):
        projected = native - coefficient * defect if active else native
        parameter.grad = projected
        projected_gradients.append(projected)
    projected_alignment = sum(
        (defect * metric * projected).sum()
        for defect, metric, projected in zip(defect_gradients, metrics, projected_gradients)
    )
    native_descent = sum(
        (native * metric * projected).sum()
        for native, metric, projected in zip(native_gradients, metrics, projected_gradients)
    )

    self.optimizer_G.step()
    self._update_lbst_teacher()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.step()
    self._search005_phrsup_last = {
        **defect_observation,
        "defect_loss": float(defect_loss.detach().item()),
        "projection_active": active,
        "native_defect_alignment": float(alignment.detach().item()),
        "projected_defect_alignment": float(projected_alignment.detach().item()),
        "native_descent_alignment": float(native_descent.detach().item()),
        "projection_coefficient": float(coefficient.detach().item()),
    }


def install_phrsup(model) -> None:
    """Install the PHCRP revision as a rate-safe native update projection."""
    if hasattr(model, "_search005_phrsup_original_predict_endpoint"):
        raise RuntimeError("PHRSUP is already installed")
    model._search005_phrsup_original_predict_endpoint = model._predict_endpoint
    model._search005_phrsup_original_forward = model.forward
    model._search005_phrsup_original_optimize_parameters = model.optimize_parameters
    model._search005_phrsup_original_get_extra_training_state = model.get_extra_training_state
    model._search005_phrsup_original_load_extra_training_state = model.load_extra_training_state
    model.search005_phrsup_active = True
    model._search005_phrsup_last = {}

    def forward(self):
        self._search005_phrsup_context = {
            "inside_training_forward": True,
            "rollout_calls": 0,
            "final_started": False,
            "histories": [None, None, None],
        }
        try:
            return self._search005_phrsup_original_forward()
        finally:
            self._search005_phrsup_context["inside_training_forward"] = False

    def get_extra_training_state(self):
        state = self._search005_phrsup_original_get_extra_training_state()
        state["search005_phrsup"] = {
            "active": bool(self.search005_phrsup_active),
            "last": copy.deepcopy(self._search005_phrsup_last),
        }
        return state

    def load_extra_training_state(self, state):
        self._search005_phrsup_original_load_extra_training_state(state)
        saved = (state or {}).get("search005_phrsup")
        if saved is None:
            raise RuntimeError("PHRSUP checkpoint is missing operator state")
        self.search005_phrsup_active = bool(saved.get("active", True))
        self._search005_phrsup_last = copy.deepcopy(saved.get("last", {}))

    model._predict_endpoint = types.MethodType(phrsup_predict_endpoint, model)
    model.forward = types.MethodType(forward, model)
    model.optimize_parameters = types.MethodType(phrsup_optimize_parameters, model)
    model.get_extra_training_state = types.MethodType(get_extra_training_state, model)
    model.load_extra_training_state = types.MethodType(load_extra_training_state, model)


def set_phrsup_active(model, active: bool) -> None:
    if not hasattr(model, "_search005_phrsup_original_optimize_parameters"):
        raise RuntimeError("PHRSUP is not installed")
    model.search005_phrsup_active = bool(active)


def phrsup_installation_status(model) -> dict:
    return {
        "installed": hasattr(model, "_search005_phrsup_original_optimize_parameters"),
        "active": bool(getattr(model, "search005_phrsup_active", False)),
        "changes_endpoint_law": False,
        "paired_target_access": False,
    }


def _pcoa_parameters(optimizer):
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _pcoa_step(model, player: str, optimizer) -> dict:
    parameters = _pcoa_parameters(optimizer)
    before = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer.step()
    current = tuple(
        parameter.detach() - old for parameter, old in zip(parameters, before)
    )
    previous = model._search005_pcoa_previous.get(player)
    if previous is None:
        rho = current[0].new_zeros(()) if current else torch.tensor(0.0)
        innovation_sq = rho.clone()
        correction_sq = rho.clone()
        current_sq = sum(value.square().sum() for value in current)
        previous_sq = rho.clone()
    else:
        if len(previous) != len(current):
            raise RuntimeError(f"PCOA {player} previous-update length mismatch")
        previous = tuple(value.to(parameter.device) for value, parameter in zip(previous, parameters))
        dot = sum((now * old).sum() for now, old in zip(current, previous))
        previous_sq = sum(old.square().sum() for old in previous)
        rho = (
            dot.new_zeros(())
            if previous_sq <= 0
            else torch.clamp(dot / previous_sq, 0.0, 1.0)
        )
        innovation = tuple(now - old for now, old in zip(current, previous))
        with torch.no_grad():
            for parameter, delta in zip(parameters, innovation):
                parameter.add_(delta, alpha=float(rho.item()))
        innovation_sq = sum(value.square().sum() for value in innovation)
        correction_sq = rho.square() * innovation_sq
        current_sq = sum(value.square().sum() for value in current)
    model._search005_pcoa_previous[player] = tuple(
        value.detach().clone() for value in current
    )
    return {
        "predictability": float(rho.detach().item()),
        "current_native_update_norm": float(current_sq.detach().sqrt().item()),
        "previous_native_update_norm": float(previous_sq.detach().sqrt().item()),
        "innovation_norm": float(innovation_sq.detach().sqrt().item()),
        "optimistic_correction_norm": float(correction_sq.detach().sqrt().item()),
    }


def pcoa_optimize_parameters(self):
    if not bool(getattr(self, "search005_pcoa_active", True)):
        return self._search005_pcoa_original_optimize_parameters()
    self.forward()
    self.netG.train()
    self.netE.train()
    self.netD.train()
    self.netF.train()

    self.set_requires_grad(self.netD, True)
    self.optimizer_D.zero_grad()
    self.loss_D = self.compute_D_loss()
    self.loss_D.backward()
    d_diag = _pcoa_step(self, "D", self.optimizer_D)

    self.set_requires_grad(self.netE, True)
    self.optimizer_E.zero_grad()
    self.loss_E = self.compute_E_loss()
    self.loss_E.backward()
    e_diag = _pcoa_step(self, "E", self.optimizer_E)

    self.set_requires_grad(self.netD, False)
    self.set_requires_grad(self.netE, False)
    self.optimizer_G.zero_grad()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.zero_grad()
    self.loss_G = self.compute_G_loss()
    self.loss_G.backward()
    g_diag = _pcoa_step(self, "G", self.optimizer_G)
    self._update_lbst_teacher()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.step()
    self._search005_pcoa_last = {"G": g_diag, "D": d_diag, "E": e_diag}


def install_pcoa(model) -> None:
    """Install predictability-calibrated Optimistic Adam on G/D/E."""
    if hasattr(model, "_search005_pcoa_original_optimize_parameters"):
        raise RuntimeError("PCOA is already installed")
    model._search005_pcoa_original_optimize_parameters = model.optimize_parameters
    model._search005_pcoa_original_get_extra_training_state = model.get_extra_training_state
    model._search005_pcoa_original_load_extra_training_state = model.load_extra_training_state
    model._search005_pcoa_previous = {"G": None, "D": None, "E": None}
    model._search005_pcoa_last = {}
    model.search005_pcoa_active = True

    def get_extra_training_state(self):
        state = self._search005_pcoa_original_get_extra_training_state()
        state["search005_pcoa"] = {
            "active": bool(self.search005_pcoa_active),
            "previous_native_updates": self._search005_pcoa_previous,
            "last": copy.deepcopy(self._search005_pcoa_last),
        }
        return state

    def load_extra_training_state(self, state):
        self._search005_pcoa_original_load_extra_training_state(state)
        saved = (state or {}).get("search005_pcoa")
        if saved is None:
            raise RuntimeError("PCOA checkpoint is missing operator state")
        self.search005_pcoa_active = bool(saved.get("active", True))
        self._search005_pcoa_last = copy.deepcopy(saved.get("last", {}))
        recorded = saved.get("previous_native_updates", {})
        restored = {}
        optimizers = {"G": self.optimizer_G, "D": self.optimizer_D, "E": self.optimizer_E}
        for player, optimizer in optimizers.items():
            values = recorded.get(player)
            parameters = tuple(
                parameter for group in optimizer.param_groups for parameter in group["params"]
            )
            if values is not None and len(values) != len(parameters):
                raise RuntimeError(f"PCOA {player} checkpoint length mismatch")
            restored[player] = (
                None
                if values is None
                else tuple(value.detach().to(parameter.device).clone() for value, parameter in zip(values, parameters))
            )
        self._search005_pcoa_previous = restored

    model.optimize_parameters = types.MethodType(pcoa_optimize_parameters, model)
    model.get_extra_training_state = types.MethodType(get_extra_training_state, model)
    model.load_extra_training_state = types.MethodType(load_extra_training_state, model)


def set_pcoa_active(model, active: bool) -> None:
    if not hasattr(model, "_search005_pcoa_original_optimize_parameters"):
        raise RuntimeError("PCOA is not installed")
    model.search005_pcoa_active = bool(active)


def pcoa_installation_status(model) -> dict:
    return {
        "installed": hasattr(model, "_search005_pcoa_original_optimize_parameters"),
        "active": bool(getattr(model, "search005_pcoa_active", False)),
        "players": ["G", "D", "E"],
        "feature_sampler_native": True,
        "changes_endpoint_law": False,
        "paired_target_access": False,
    }


def _npooa_step(model, player: str, optimizer) -> dict:
    """Apply a global, norm-preserving phase correction to one player update."""
    parameters = _pcoa_parameters(optimizer)
    before = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer.step()
    current = tuple(
        parameter.detach() - old for parameter, old in zip(parameters, before)
    )
    previous = model._search005_npooa_previous.get(player)
    current_sq = sum(value.square().sum() for value in current)
    zero = current_sq.new_zeros(()) if current else torch.tensor(0.0)
    if previous is None:
        rho = zero
        previous_sq = zero
        innovation_sq = zero
        orthogonal_sq = zero
        correction_sq = zero
        applied_sq = current_sq
    else:
        if len(previous) != len(current):
            raise RuntimeError(f"NPOOA {player} previous-update length mismatch")
        previous = tuple(
            value.to(parameter.device) for value, parameter in zip(previous, parameters)
        )
        previous_sq = sum(value.square().sum() for value in previous)
        dot = sum((now * old).sum() for now, old in zip(current, previous))
        rho = (
            zero
            if previous_sq <= 0 or current_sq <= 0
            else torch.clamp(dot / previous_sq, 0.0, 1.0)
        )
        innovation = tuple(now - old for now, old in zip(current, previous))
        innovation_sq = sum(value.square().sum() for value in innovation)
        if current_sq <= 0 or rho <= 0:
            orthogonal = tuple(torch.zeros_like(value) for value in current)
            applied = current
        else:
            innovation_current = sum(
                (delta * now).sum() for delta, now in zip(innovation, current)
            )
            parallel = innovation_current / current_sq
            orthogonal = tuple(
                delta - parallel * now for delta, now in zip(innovation, current)
            )
            proposal = tuple(
                now + rho * tangent for now, tangent in zip(current, orthogonal)
            )
            proposal_sq = sum(value.square().sum() for value in proposal)
            scale = (
                torch.ones_like(proposal_sq)
                if proposal_sq <= 0
                else torch.sqrt(current_sq / proposal_sq)
            )
            applied = tuple(scale * value for value in proposal)
        correction = tuple(
            accepted - native for accepted, native in zip(applied, current)
        )
        with torch.no_grad():
            for parameter, delta in zip(parameters, correction):
                parameter.add_(delta)
        orthogonal_sq = sum(value.square().sum() for value in orthogonal)
        correction_sq = sum(value.square().sum() for value in correction)
        applied_sq = sum(value.square().sum() for value in applied)
    model._search005_npooa_previous[player] = tuple(
        value.detach().clone() for value in current
    )
    current_norm = current_sq.detach().clamp_min(0).sqrt()
    applied_norm = applied_sq.detach().clamp_min(0).sqrt()
    norm_ratio = (
        torch.ones_like(current_norm)
        if current_norm <= 0
        else applied_norm / current_norm
    )
    return {
        "predictability": float(rho.detach().item()),
        "current_native_update_norm": float(current_norm.item()),
        "previous_native_update_norm": float(previous_sq.detach().clamp_min(0).sqrt().item()),
        "innovation_norm": float(innovation_sq.detach().clamp_min(0).sqrt().item()),
        "orthogonal_innovation_norm": float(orthogonal_sq.detach().clamp_min(0).sqrt().item()),
        "phase_correction_norm": float(correction_sq.detach().clamp_min(0).sqrt().item()),
        "applied_update_norm": float(applied_norm.item()),
        "norm_ratio": float(norm_ratio.item()),
    }


def npooa_optimize_parameters(self):
    if not bool(getattr(self, "search005_npooa_active", True)):
        return self._search005_npooa_original_optimize_parameters()
    self.forward()
    self.netG.train()
    self.netE.train()
    self.netD.train()
    self.netF.train()

    self.set_requires_grad(self.netD, True)
    self.optimizer_D.zero_grad()
    self.loss_D = self.compute_D_loss()
    self.loss_D.backward()
    d_diag = _npooa_step(self, "D", self.optimizer_D)

    self.set_requires_grad(self.netE, True)
    self.optimizer_E.zero_grad()
    self.loss_E = self.compute_E_loss()
    self.loss_E.backward()
    e_diag = _npooa_step(self, "E", self.optimizer_E)

    self.set_requires_grad(self.netD, False)
    self.set_requires_grad(self.netE, False)
    self.optimizer_G.zero_grad()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.zero_grad()
    self.loss_G = self.compute_G_loss()
    self.loss_G.backward()
    g_diag = _npooa_step(self, "G", self.optimizer_G)
    self._update_lbst_teacher()
    if self.opt.netF == "mlp_sample":
        self.optimizer_F.step()
    self._search005_npooa_last = {"G": g_diag, "D": d_diag, "E": e_diag}


def install_npooa(model) -> None:
    """Install norm-preserving orthogonal Optimistic Adam on G/D/E."""
    if hasattr(model, "_search005_npooa_original_optimize_parameters"):
        raise RuntimeError("NPOOA is already installed")
    model._search005_npooa_original_optimize_parameters = model.optimize_parameters
    model._search005_npooa_original_get_extra_training_state = model.get_extra_training_state
    model._search005_npooa_original_load_extra_training_state = model.load_extra_training_state
    model._search005_npooa_previous = {"G": None, "D": None, "E": None}
    model._search005_npooa_last = {}
    model.search005_npooa_active = True

    def get_extra_training_state(self):
        state = self._search005_npooa_original_get_extra_training_state()
        state["search005_npooa"] = {
            "active": bool(self.search005_npooa_active),
            "previous_native_updates": self._search005_npooa_previous,
            "last": copy.deepcopy(self._search005_npooa_last),
        }
        return state

    def load_extra_training_state(self, state):
        self._search005_npooa_original_load_extra_training_state(state)
        saved = (state or {}).get("search005_npooa")
        if saved is None:
            raise RuntimeError("NPOOA checkpoint is missing operator state")
        self.search005_npooa_active = bool(saved.get("active", True))
        self._search005_npooa_last = copy.deepcopy(saved.get("last", {}))
        recorded = saved.get("previous_native_updates", {})
        restored = {}
        optimizers = {"G": self.optimizer_G, "D": self.optimizer_D, "E": self.optimizer_E}
        for player, optimizer in optimizers.items():
            values = recorded.get(player)
            parameters = _pcoa_parameters(optimizer)
            if values is not None and len(values) != len(parameters):
                raise RuntimeError(f"NPOOA {player} checkpoint length mismatch")
            restored[player] = (
                None
                if values is None
                else tuple(
                    value.detach().to(parameter.device).clone()
                    for value, parameter in zip(values, parameters)
                )
            )
        self._search005_npooa_previous = restored

    model.optimize_parameters = types.MethodType(npooa_optimize_parameters, model)
    model.get_extra_training_state = types.MethodType(get_extra_training_state, model)
    model.load_extra_training_state = types.MethodType(load_extra_training_state, model)


def set_npooa_active(model, active: bool) -> None:
    if not hasattr(model, "_search005_npooa_original_optimize_parameters"):
        raise RuntimeError("NPOOA is not installed")
    model.search005_npooa_active = bool(active)


def npooa_installation_status(model) -> dict:
    return {
        "installed": hasattr(model, "_search005_npooa_original_optimize_parameters"),
        "active": bool(getattr(model, "search005_npooa_active", False)),
        "players": ["G", "D", "E"],
        "feature_sampler_native": True,
        "native_update_norm_preserved": True,
        "changes_endpoint_law": False,
        "paired_target_access": False,
    }
