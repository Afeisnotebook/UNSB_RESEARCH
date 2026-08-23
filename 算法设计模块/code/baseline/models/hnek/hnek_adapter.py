"""Non-invasive HNEK adapter for the official UNSB ``SBModel``.

Installation leaves every parameter and state-dict key unchanged.  It only
rebinds three Python methods:

* ``netG.forward``: official proposal -> horizon-normalized endpoint;
* ``SBModel.compute_E_loss``: estimate entropy in (x_t, r) coordinates;
* ``SBModel.compute_G_loss``: use the actual physical horizon and r-coordinate.

The baseline source tree is intentionally not edited.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch

from .hnek_kernel import horizon_from_condition, normalized_residual


@dataclass(frozen=True)
class InstallRecord:
    num_timesteps: int
    generator_parameter_count: int
    generator_state_keys: tuple[str, ...]
    exponent: float = 0.5
    coordinate: str = "r=(y-x_t)/sqrt(1-t)"


def _inner(net):
    return net.module if hasattr(net, "module") else net


def install_hnek_generator(net_g, *, num_timesteps: int = 5) -> InstallRecord:
    """Install the analytic endpoint layer without changing module state."""
    generator = _inner(net_g)
    if hasattr(generator, "_hnek_original_forward"):
        raise RuntimeError("HNEK generator adapter is already installed")

    before_keys = tuple(generator.state_dict().keys())
    before_count = sum(parameter.numel() for parameter in generator.parameters())
    original_forward = generator.forward
    generator._hnek_original_forward = original_forward
    generator._hnek_num_timesteps = int(num_timesteps)

    def hnek_forward(self, x, time_cond, z, layers=None, encode_only=False):
        requested_layers = [] if layers is None else layers
        result = self._hnek_original_forward(
            x, time_cond, z, requested_layers, encode_only
        )
        # Feature extraction is a representation API, not an endpoint law.
        # Preserve it exactly so PatchNCE remains unchanged.
        if len(requested_layers) > 0:
            return result
        proposal = result
        horizon = horizon_from_condition(
            time_cond, num_timesteps=self._hnek_num_timesteps, like=x
        )
        endpoint = x + torch.sqrt(horizon) * (proposal - x)
        endpoint = torch.where(horizon == 1, proposal, endpoint)
        endpoint = torch.where(horizon == 0, x, endpoint)
        return endpoint

    generator.forward = types.MethodType(hnek_forward, generator)
    after_keys = tuple(generator.state_dict().keys())
    after_count = sum(parameter.numel() for parameter in generator.parameters())
    if before_keys != after_keys or before_count != after_count:
        raise RuntimeError("adapter changed generator state identity")
    return InstallRecord(num_timesteps, before_count, before_keys)


def uninstall_hnek_generator(net_g) -> None:
    generator = _inner(net_g)
    if not hasattr(generator, "_hnek_original_forward"):
        raise RuntimeError("HNEK generator adapter is not installed")
    generator.forward = generator._hnek_original_forward
    delattr(generator, "_hnek_original_forward")
    delattr(generator, "_hnek_num_timesteps")


def _horizon(model, like: torch.Tensor) -> torch.Tensor:
    return horizon_from_condition(
        model.time_idx,
        num_timesteps=int(model.opt.num_timesteps),
        like=like,
    )


def _residual_pair(model, *, detach: bool) -> tuple[torch.Tensor, torch.Tensor]:
    h1 = _horizon(model, model.fake_B)
    h2 = _horizon(model, model.fake_B2)
    y1 = model.fake_B.detach() if detach else model.fake_B
    y2 = model.fake_B2.detach() if detach else model.fake_B2
    r1 = normalized_residual(model.real_A_noisy, y1, h1)
    r2 = normalized_residual(model.real_A_noisy2, y2, h2)
    return r1, r2


def hnek_compute_E_loss(self):
    """Official NWJ-like critic loss in the invertible (x_t, r) chart."""
    r1, r2 = _residual_pair(self, detach=True)
    x_r1 = torch.cat([self.real_A_noisy, r1], dim=1)
    x_r2 = torch.cat([self.real_A_noisy2, r2], dim=1)
    temp = torch.logsumexp(
        self.netE(x_r1, self.time_idx, x_r2).reshape(-1), dim=0
    ).mean()
    self.loss_E = -self.netE(x_r1, self.time_idx, x_r1).mean() + temp + temp ** 2
    return self.loss_E


def hnek_compute_G_loss(self):
    """Original generator objective with only the HNEK SB coordinate changed."""
    fake = self.fake_B
    tau = float(self.opt.tau)

    if self.opt.lambda_GAN > 0.0:
        pred_fake = self.netD(fake, self.time_idx)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
    else:
        self.loss_G_GAN = 0.0

    self.loss_SB = 0.0
    if self.opt.lambda_SB > 0.0:
        r1, r2 = _residual_pair(self, detach=False)
        x_r1 = torch.cat([self.real_A_noisy, r1], dim=1)
        x_r2 = torch.cat([self.real_A_noisy2, r2], dim=1)
        et_r = self.netE(x_r1, self.time_idx, x_r1).mean() - torch.logsumexp(
            self.netE(x_r1, self.time_idx, x_r2).reshape(-1), dim=0
        )
        horizon_scalar = _horizon(self, self.fake_B).reshape(-1)[0]
        # ||y-x||^2 already equals h ||r||^2.  The entropy term must use the
        # same physical h; H(y|x)=H(r|x)+(d/2)log(h), and the last term is
        # independent of generator parameters.
        self.loss_SB = -horizon_scalar * tau * et_r
        self.loss_SB = self.loss_SB + tau * torch.mean(
            (self.real_A_noisy - self.fake_B) ** 2
        )

    if self.opt.lambda_NCE > 0.0:
        self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
    else:
        self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

    if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
        self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
        loss_nce_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
    else:
        loss_nce_both = self.loss_NCE

    # Preserve the official implementation, including its historical outer
    # lambda_NCE multiplication.  HNEK does not silently fix unrelated code.
    self.loss_G = (
        self.loss_G_GAN
        + self.opt.lambda_SB * self.loss_SB
        + self.opt.lambda_NCE * loss_nce_both
    )
    return self.loss_G


def install_hnek_model(model) -> InstallRecord:
    """Install HNEK on a constructed official time-dead ``SBModel``."""
    record = install_hnek_generator(
        model.netG, num_timesteps=int(model.opt.num_timesteps)
    )
    if hasattr(model, "_hnek_original_compute_E_loss"):
        raise RuntimeError("HNEK model adapter is already installed")
    model._hnek_original_compute_E_loss = model.compute_E_loss
    model._hnek_original_compute_G_loss = model.compute_G_loss
    model.compute_E_loss = types.MethodType(hnek_compute_E_loss, model)
    model.compute_G_loss = types.MethodType(hnek_compute_G_loss, model)
    model._hnek_install_record = record
    return record


def hnek_installation_status(model) -> dict:
    generator = _inner(model.netG)
    record = getattr(model, "_hnek_install_record", None)
    return {
        "installed": bool(
            record is not None
            and hasattr(generator, "_hnek_original_forward")
            and hasattr(model, "_hnek_original_compute_E_loss")
        ),
        "num_timesteps": None if record is None else record.num_timesteps,
        "parameter_count": None if record is None else record.generator_parameter_count,
        "exponent": None if record is None else record.exponent,
        "coordinate": None if record is None else record.coordinate,
    }
