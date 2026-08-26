"""UNSB model integration for Latent-Tangent Trust Region."""

from __future__ import annotations

import copy
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_CANONICAL_ROOT = os.path.join(_REPO_ROOT, "foundation", "canonical", "src")
if _CANONICAL_ROOT not in sys.path:
    sys.path.insert(0, _CANONICAL_ROOT)

from models.sb_model import SBModel  # noqa: E402
from util.util import str2bool  # noqa: E402

from .core import LTTRConfig, lttr_loss, response_statistics  # noqa: E402


class SBModelLTTR(SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--lttr_enable", type=str2bool, nargs="?", const=True, default=False)
        parser.add_argument("--lttr_mode", choices=["tangent", "safe"], default="safe")
        parser.add_argument("--lttr_lambda", type=float, default=0.001)
        parser.add_argument("--lttr_region_patch", type=int, default=32)
        parser.add_argument("--lttr_direction_margin", type=float, default=0.5)
        parser.add_argument("--lttr_direction_weight", type=float, default=0.25)
        parser.add_argument("--lttr_start_step", type=int, default=-1)
        parser.add_argument("--lttr_duration_steps", type=int, default=0)
        parser.add_argument("--lttr_latent_seed", type=int, default=2026)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names += ["LTTR", "LTTR_tangent", "LTTR_direction"]
        self._lttr_teacher = None
        self._lttr_teacher_step = None
        self._lttr_last_diag = {}
        self.lttr_config = LTTRConfig(
            region_patch=int(getattr(opt, "lttr_region_patch", 32)),
            direction_margin=float(getattr(opt, "lttr_direction_margin", 0.5)),
            direction_weight=float(getattr(opt, "lttr_direction_weight", 0.25)),
        )

    @staticmethod
    def _inner(net):
        return net.module if isinstance(net, torch.nn.DataParallel) else net

    def _ensure_lttr_teacher(self):
        if self._lttr_teacher is None:
            source = self._inner(self.netG)
            self._lttr_teacher = copy.deepcopy(source).to(self.device)
            self._lttr_teacher.eval()
            for parameter in self._lttr_teacher.parameters():
                parameter.requires_grad_(False)
            self._lttr_teacher_step = int(getattr(self, "_search_global_step", 0))
        return self._lttr_teacher

    def reset_lttr_teacher(self):
        """Drop any data-dependent-initialization snapshot before an e0 fork."""
        self._lttr_teacher = None
        self._lttr_teacher_step = None

    def _lttr_schedule(self) -> float:
        if not bool(getattr(self.opt, "lttr_enable", False)):
            return 0.0
        start = int(getattr(self.opt, "lttr_start_step", -1))
        duration = int(getattr(self.opt, "lttr_duration_steps", 0))
        step = int(getattr(self, "_search_global_step", 0))
        if start < 0 or duration <= 0 or step < start or step >= start + duration:
            return 0.0
        age = step - start
        ramp = max(1, round(0.2 * duration))
        decay_start = round(0.6 * duration)
        if age < ramp:
            return float(age + 1) / float(ramp)
        if age <= decay_start:
            return 1.0
        progress = float(age - decay_start) / float(max(1, duration - decay_start))
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

    def _lttr_latent(self, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device=device)
        step = int(getattr(self, "_search_global_step", 0))
        seed = int(getattr(self.opt, "lttr_latent_seed", 2026)) + 104729 * step
        generator.manual_seed(seed)
        return torch.randn(
            (batch, 4 * int(self.opt.ngf)), generator=generator, device=device, dtype=dtype
        )

    def _compute_lttr(self):
        x_t = self.real_A_noisy.detach()
        source = self.real_A.detach()
        z = self._lttr_latent(x_t.size(0), x_t.dtype, x_t.device)
        current_plus = self.netG(x_t, self.time_idx, z)
        current_minus = self.netG(x_t, self.time_idx, -z)
        teacher = self._ensure_lttr_teacher()
        with torch.no_grad():
            teacher_plus = teacher(x_t, self.time_idx, z)
            teacher_minus = teacher(x_t, self.time_idx, -z)
        kwargs = {
            "x_t": x_t,
            "region_patch": int(self.lttr_config.region_patch),
            "eps": float(self.lttr_config.eps),
        }
        current = response_statistics(
            endpoint_plus=current_plus, endpoint_minus=current_minus, **kwargs
        )
        reference = response_statistics(
            endpoint_plus=teacher_plus, endpoint_minus=teacher_minus, **kwargs
        )
        return lttr_loss(
            current=current,
            teacher=reference,
            source=source,
            mode=str(getattr(self.opt, "lttr_mode", "safe")),
            config=self.lttr_config,
        )

    def compute_G_loss(self):
        base = super().compute_G_loss()
        schedule = self._lttr_schedule()
        if schedule <= 0.0:
            zero = base.detach().new_zeros(())
            self.loss_LTTR = zero
            self.loss_LTTR_tangent = zero
            self.loss_LTTR_direction = zero
            return base
        auxiliary, diag = self._compute_lttr()
        self._lttr_last_diag = {key: float(value.item()) for key, value in diag.items()}
        self.loss_LTTR = auxiliary.detach()
        self.loss_LTTR_tangent = diag["tangent"]
        self.loss_LTTR_direction = diag["direction"]
        self.loss_G = base + float(getattr(self.opt, "lttr_lambda", 0.001)) * schedule * auxiliary
        return self.loss_G

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        state["lttr"] = {
            "mode": str(getattr(self.opt, "lttr_mode", "safe")),
            "teacher_step": self._lttr_teacher_step,
            "teacher": None if self._lttr_teacher is None else {
                key: value.detach().cpu() for key, value in self._lttr_teacher.state_dict().items()
            },
            "last_diag": copy.deepcopy(self._lttr_last_diag),
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        saved = (state or {}).get("lttr")
        if not saved:
            return
        if saved.get("mode") != str(getattr(self.opt, "lttr_mode", "safe")):
            raise RuntimeError("LTTR mode mismatch while restoring checkpoint")
        teacher_state = saved.get("teacher")
        if teacher_state is not None:
            teacher = self._ensure_lttr_teacher()
            teacher.load_state_dict(teacher_state, strict=True)
            teacher.eval()
        self._lttr_teacher_step = saved.get("teacher_step")
        self._lttr_last_diag = copy.deepcopy(saved.get("last_diag", {}))
