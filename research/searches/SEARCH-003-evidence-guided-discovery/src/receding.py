"""Generation-1 target-blind receding-horizon candidates.

The controller never edits an update.  At every audited block it executes an
actual plain UNSB branch and an actual proposal branch from the same immutable
full state, then commits one complete branch.  Consequently rejection is an
exact identity transition: rejected parameters, optimizer moments, scheduler
state, data streams, RNG and method co-state are all discarded.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import os
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .atlas import _force_operator_active
from .search001_compat import modules


@dataclass(frozen=True)
class RecedingSpec:
    candidate_id: str
    probe: str
    observable: str
    direction: int
    horizon: int = 8
    future_native_consensus: bool = False

    def __post_init__(self) -> None:
        if self.probe not in {"dt", "hj"}:
            raise ValueError(f"unsupported Generation-1 probe: {self.probe}")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be -1 or +1")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


CANDIDATES = {
    "G1-DT-RHGC8": RecedingSpec(
        candidate_id="G1-DT-RHGC8",
        probe="dt",
        observable="G_GAN",
        direction=-1,
    ),
    "G1-HJ-RHDFC8": RecedingSpec(
        candidate_id="G1-HJ-RHDFC8",
        probe="hj",
        observable="D_fake",
        direction=1,
    ),
    "G2-DT-FBGC8": RecedingSpec(
        candidate_id="G2-DT-FBGC8",
        probe="dt",
        observable="G_GAN",
        direction=-1,
        future_native_consensus=True,
    ),
    "G2-HJ-FBDFC8": RecedingSpec(
        candidate_id="G2-HJ-FBDFC8",
        probe="hj",
        observable="D_fake",
        direction=1,
        future_native_consensus=True,
    ),
}


def proposal_selected(direction: int, proposal_value: float, plain_value: float) -> bool:
    """The sole controller decision; values are unpaired training losses."""
    return int(direction) * (float(proposal_value) - float(plain_value)) > 0.0


def _lane(spec: RecedingSpec, name: str | None = None):
    protocol, _, _ = modules()
    model = "dtcov" if spec.probe == "dt" else "hj"
    return protocol.LaneSpec(
        name=name or spec.candidate_id,
        model=model,
        family="search003_generation1",
    )


def _plain_lane(name: str = "plain"):
    protocol, _, _ = modules()
    return protocol.LaneSpec(name=name)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _strip_operator_extra(model_state: dict, *, step: int, target_steps: int) -> dict:
    state = _cpu_clone(model_state)
    state["extra"] = {
        "search_global_step": int(step),
        "search_total_steps": int(target_steps),
    }
    return state


def create_e0(
    *,
    path: Path,
    rows: list[dict],
    train_view: Path,
    per_domain: int,
    seed: int,
    gpu: int,
    target_steps: int,
) -> dict:
    """Create the common plain initialization used by every matched lane."""
    if Path(path).is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "per_domain": int(per_domain),
            "seed": int(seed),
        }
        actual = {key: payload[key] for key in expected}
        if actual != expected:
            raise RuntimeError(f"e0 identity mismatch: {actual} != {expected}")
        return payload
    _, runtime, _ = modules()
    steps_per_epoch = 6 * int(per_domain)
    runtime.seed_everything(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = runtime.build_options(
            _plain_lane("search003_e0"),
            dataroot=Path(train_view),
            checkpoint_dir=Path(path).parent / "option_records",
            steps_per_epoch=steps_per_epoch,
            total_steps=int(target_steps),
            seed=seed,
            gpu=gpu,
        )
        dataset_a, dataset_b = runtime.build_datasets(opt, rows, per_domain)
        stream_a = runtime.SerializableDataStream(dataset_a, seed=seed + 101)
        stream_b = runtime.SerializableDataStream(dataset_b, seed=seed + 202)
        model = runtime.build_model(opt, stream_a.next(), stream_b.next())
    payload = {
        "schema": "clean-unsb-search003-e0-v1",
        "step": 0,
        "target_steps": int(target_steps),
        "per_domain": int(per_domain),
        "steps_per_epoch": steps_per_epoch,
        "seed": int(seed),
        "model": _strip_operator_extra(
            runtime.model_state(model), step=0, target_steps=target_steps
        ),
        "rng": runtime.capture_rng(),
        "stream_a": stream_a.state_dict(),
        "stream_b": stream_b.state_dict(),
        "confirmation20_opened": False,
    }
    _atomic_torch_save(path, payload)
    del model, stream_a, stream_b, dataset_a, dataset_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


class RecedingEngine:
    """One live proposal-class model, reset exactly between serial branches."""

    def __init__(
        self,
        *,
        spec: RecedingSpec,
        rows: list[dict],
        train_view: Path,
        work_dir: Path,
        per_domain: int,
        seed: int,
        gpu: int,
        target_steps: int,
    ) -> None:
        self.spec = spec
        self.rows = rows
        self.per_domain = int(per_domain)
        self.seed = int(seed)
        self.target_steps = int(target_steps)
        self.steps_per_epoch = 6 * self.per_domain
        _, self.runtime, _ = modules()
        self.runtime.seed_everything(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            opt = self.runtime.build_options(
                _lane(spec),
                dataroot=Path(train_view),
                checkpoint_dir=Path(work_dir) / "option_records",
                steps_per_epoch=self.steps_per_epoch,
                total_steps=target_steps,
                seed=seed,
                gpu=gpu,
            )
            dataset_a, dataset_b = self.runtime.build_datasets(opt, rows, per_domain)
            self.stream_a = self.runtime.SerializableDataStream(
                dataset_a, seed=seed + 101
            )
            self.stream_b = self.runtime.SerializableDataStream(
                dataset_b, seed=seed + 202
            )
            self.model = self.runtime.build_model(
                opt, self.stream_a.next(), self.stream_b.next()
            )
        self._fresh_extra = _cpu_clone(self.model.get_extra_training_state())
        self._base_class = next(
            cls for cls in type(self.model).__mro__ if cls.__name__ == "SBModel"
        )

    def close(self) -> None:
        del self.model, self.stream_a, self.stream_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_extra(self, state: dict | None, *, step: int) -> None:
        extra = _cpu_clone(self._fresh_extra if state is None else state)
        extra["search_global_step"] = int(step)
        extra["search_total_steps"] = int(self.target_steps)
        self.model.load_extra_training_state(extra)
        if self.spec.probe == "dt" and state is None:
            # ``load_extra_training_state`` intentionally does not interpret a
            # missing teacher as "delete an existing teacher".  Branch reset
            # does require that stronger semantics.
            self.model.dtcov.teacher = None
            self.model.dtcov.stats.store = {}
            self.model.dtcov.iter = 0

    @contextlib.contextmanager
    def _plain_operator(self):
        """Call the actual SB objective through a proposal-class container."""
        patched = ["compute_G_loss"]
        self.model.compute_G_loss = types.MethodType(
            self._base_class.compute_G_loss, self.model
        )
        if self.spec.probe == "hj":
            # SBModel.compute_G_loss dispatches this method dynamically.
            patched.append("calculate_NCE_loss")
            self.model.calculate_NCE_loss = types.MethodType(
                self._base_class.calculate_NCE_loss, self.model
            )
        try:
            yield
        finally:
            for name in patched:
                delattr(self.model, name)

    def branch(
        self,
        source: dict,
        *,
        proposal: bool,
    ) -> dict:
        step = int(source["step"])
        horizon = min(self.spec.horizon, self.target_steps - step)
        if horizon <= 0:
            raise ValueError("source is already at target")
        self.runtime.load_model_state(self.model, source["model"], load_extra=False)
        self.stream_a.load_state_dict(_cpu_clone(source["stream_a"]))
        self.stream_b.load_state_dict(_cpu_clone(source["stream_b"]))
        self.runtime.restore_rng(_cpu_clone(source["rng"]))
        self._reset_extra(
            source.get("proposal_costate") if proposal else None,
            step=step,
        )
        loss_sums: dict[str, float] = {}
        operator_context = (
            contextlib.nullcontext() if proposal else self._plain_operator()
        )
        with operator_context:
            for offset in range(horizon):
                zero_step = step + offset
                physical_epoch = 1 + zero_step // self.steps_per_epoch
                self.model.set_train_epoch(physical_epoch)
                if proposal:
                    _force_operator_active(self.model, self.spec.probe, zero_step)
                else:
                    self.model.set_search_step(zero_step, self.target_steps)
                self.model.set_input(self.stream_a.next(), self.stream_b.next())
                self.model.optimize_parameters()
                for key, value in self.model.get_current_losses().items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
                completed = zero_step + 1
                if completed % self.steps_per_epoch == 0:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.model.update_learning_rate()
        completed_step = step + horizon
        full_model_state = self.runtime.model_state(self.model)
        costate = _cpu_clone(full_model_state["extra"]) if proposal else None
        branch_state = {
            "step": completed_step,
            "target_steps": self.target_steps,
            "per_domain": self.per_domain,
            "steps_per_epoch": self.steps_per_epoch,
            "model": _strip_operator_extra(
                full_model_state, step=completed_step, target_steps=self.target_steps
            ),
            "rng": self.runtime.capture_rng(),
            "stream_a": self.stream_a.state_dict(),
            "stream_b": self.stream_b.state_dict(),
            "mean_losses": {
                key: value / float(horizon) for key, value in loss_sums.items()
            },
        }
        if proposal:
            branch_state["proposal_costate"] = costate
        return branch_state

    def future_native_consensus(self, plain: dict, proposal: dict) -> dict:
        """Test proposal correction against the next independent native gradient.

        The correction is measured relative to the plain H-step endpoint.  A
        positive score means the correction is a descent direction for the
        native UNSB generator objective on the next unpaired batch.  The batch
        and RNG are inspected transactionally: their advanced states are not
        committed by this audit.
        """
        self.runtime.load_model_state(self.model, plain["model"], load_extra=False)
        self.stream_a.load_state_dict(_cpu_clone(plain["stream_a"]))
        self.stream_b.load_state_dict(_cpu_clone(plain["stream_b"]))
        self.runtime.restore_rng(_cpu_clone(plain["rng"]))
        self._reset_extra(None, step=int(plain["step"]))
        self.model.set_train_epoch(1 + int(plain["step"]) // self.steps_per_epoch)
        self.model.set_search_step(int(plain["step"]), self.target_steps)
        with self._plain_operator():
            self.model.set_input(self.stream_a.next(), self.stream_b.next())
            self.model.forward()
            self.model.set_requires_grad(self.model.netD, False)
            self.model.set_requires_grad(self.model.netE, False)
            self.model.optimizer_G.zero_grad()
            if self.model.opt.netF == "mlp_sample":
                self.model.optimizer_F.zero_grad()
            native_loss = self.model.compute_G_loss()
            native_loss.backward()
        correction_sq = 0.0
        gradient_sq = 0.0
        correction_gradient = 0.0
        for network_name in ("G", "F"):
            network = self.runtime.inner(getattr(self.model, "net" + network_name))
            plain_network = plain["model"]["networks"][network_name]
            proposal_network = proposal["model"]["networks"][network_name]
            for name, parameter in network.named_parameters():
                if parameter.grad is None:
                    continue
                correction = (
                    proposal_network[name].to(parameter.device)
                    - plain_network[name].to(parameter.device)
                ).double()
                gradient = parameter.grad.detach().double()
                correction_sq += float(torch.sum(correction * correction).item())
                gradient_sq += float(torch.sum(gradient * gradient).item())
                correction_gradient += float(
                    torch.sum(correction * gradient).item()
                )
        denominator = (correction_sq * gradient_sq) ** 0.5
        descent_score = -correction_gradient
        return {
            "native_loss": float(native_loss.detach().item()),
            "correction_native_gradient_dot": correction_gradient,
            "descent_score": descent_score,
            "descent_cosine": (
                descent_score / denominator if denominator > 0.0 else 0.0
            ),
            "correction_norm": correction_sq ** 0.5,
            "native_gradient_norm": gradient_sq ** 0.5,
            "passes": bool(descent_score > 0.0 and correction_sq > 0.0),
            "paired_target_access": False,
        }

    def advance(self, source: dict, *, mode: str) -> dict:
        if mode not in {"full", "proposal_only", "observable_only"}:
            raise ValueError(f"unknown candidate mode: {mode}")
        history = list(source.get("decisions", []))
        parent_costate = _cpu_clone(source.get("proposal_costate"))
        if mode == "proposal_only":
            proposal = self.branch(source, proposal=True)
            proposal["schema"] = "clean-unsb-search003-candidate-state-v1"
            proposal["candidate"] = self.spec.to_dict()
            proposal["mode"] = mode
            proposal["decisions"] = history + [{
                "start_step": int(source["step"]),
                "end_step": int(proposal["step"]),
                "observable": self.spec.observable,
                "direction": self.spec.direction,
                "plain_value": None,
                "proposal_value": float(
                    proposal["mean_losses"][self.spec.observable]
                ),
                "signed_margin": None,
                "audited_choice": "not_evaluated_ablation",
                "future_native_consensus": None,
                "committed": "proposal",
                "paired_target_access": False,
            }]
            proposal["confirmation20_opened"] = False
            return proposal
        plain = self.branch(source, proposal=False)
        # The parent remains immutable.  Running proposal restores every item
        # from it, so the plain branch cannot contaminate the proposal branch.
        proposal = self.branch(source, proposal=True)
        plain_value = float(plain["mean_losses"][self.spec.observable])
        proposal_value = float(proposal["mean_losses"][self.spec.observable])
        audited_choice = proposal_selected(
            self.spec.direction, proposal_value, plain_value
        )
        future_consensus = (
            self.future_native_consensus(plain, proposal)
            if self.spec.future_native_consensus else None
        )
        safe_choice = audited_choice and (
            future_consensus is None or future_consensus["passes"]
        )
        selected_proposal = (
            False if mode == "observable_only" else safe_choice
        )
        selected = proposal if selected_proposal else plain
        selected["schema"] = "clean-unsb-search003-candidate-state-v1"
        selected["candidate"] = self.spec.to_dict()
        selected["mode"] = mode
        selected["proposal_costate"] = (
            _cpu_clone(proposal["proposal_costate"])
            if selected_proposal else parent_costate
        )
        history.append({
            "start_step": int(source["step"]),
            "end_step": int(selected["step"]),
            "observable": self.spec.observable,
            "direction": self.spec.direction,
            "plain_value": plain_value,
            "proposal_value": proposal_value,
            "signed_margin": self.spec.direction * (proposal_value - plain_value),
            "audited_choice": "proposal" if audited_choice else "plain",
            "future_native_consensus": future_consensus,
            "committed": "proposal" if selected_proposal else "plain",
            "paired_target_access": False,
        })
        selected["decisions"] = history
        selected["confirmation20_opened"] = False
        del plain, proposal
        return selected

    def load_for_evaluation(self, state: dict) -> None:
        self.runtime.load_model_state(self.model, state["model"], load_extra=False)


def _initial_candidate_state(e0: dict, spec: RecedingSpec, mode: str) -> dict:
    return {
        "schema": "clean-unsb-search003-candidate-state-v1",
        "candidate": spec.to_dict(),
        "mode": mode,
        "step": 0,
        "target_steps": int(e0["target_steps"]),
        "per_domain": int(e0["per_domain"]),
        "steps_per_epoch": int(e0["steps_per_epoch"]),
        "model": _cpu_clone(e0["model"]),
        "rng": _cpu_clone(e0["rng"]),
        "stream_a": _cpu_clone(e0["stream_a"]),
        "stream_b": _cpu_clone(e0["stream_b"]),
        "proposal_costate": None,
        "decisions": [],
        "confirmation20_opened": False,
    }


def _evaluate_state(
    engine: RecedingEngine,
    state: dict,
    *,
    rows: list[dict],
    data_root: Path,
    seed: int,
    start_per_domain: int,
    count_per_domain: int,
    include_lpips: bool,
) -> dict:
    _, runtime, evaluator = modules()
    rng = runtime.capture_rng()
    try:
        engine.load_for_evaluation(state)
        metrics = evaluator.evaluate(
            engine.model,
            rows=rows,
            data_root=data_root,
            start_per_domain=start_per_domain,
            count_per_domain=count_per_domain,
            eval_seed=seed,
            include_lpips=include_lpips,
        )
    finally:
        runtime.restore_rng(rng)
    metrics.update({
        "step": int(state["step"]),
        "candidate": engine.spec.to_dict(),
        "mode": state["mode"],
        "paired_metrics_available_to_controller": False,
        "confirmation20_opened": False,
    })
    return metrics


def run_candidate(
    *,
    spec: RecedingSpec,
    mode: str,
    output_dir: Path,
    rows: list[dict],
    train_view: Path,
    data_root: Path,
    per_domain: int,
    target_steps: int,
    eval_steps: tuple[int, ...],
    eval_start: int,
    eval_count: int,
    seed: int,
    gpu: int,
    include_lpips: bool = False,
) -> dict:
    if target_steps % spec.horizon != 0:
        raise ValueError("target_steps must be divisible by the audited horizon")
    if any(step % spec.horizon for step in eval_steps):
        raise ValueError("evaluation steps must be divisible by the audited horizon")
    lane_dir = Path(output_dir) / f"{spec.candidate_id}__{mode}"
    lane_dir.mkdir(parents=True, exist_ok=True)
    e0 = create_e0(
        path=Path(output_dir) / "e0.pt",
        rows=rows,
        train_view=train_view,
        per_domain=per_domain,
        seed=seed,
        gpu=gpu,
        target_steps=target_steps,
    )
    latest = lane_dir / "latest.pt"
    state = (
        torch.load(latest, map_location="cpu", weights_only=False)
        if latest.is_file() else _initial_candidate_state(e0, spec, mode)
    )
    if state["candidate"] != spec.to_dict() or state["mode"] != mode:
        raise RuntimeError("candidate checkpoint identity mismatch")
    if int(state["step"]) > target_steps:
        raise RuntimeError("candidate checkpoint is beyond requested target")
    state["target_steps"] = int(target_steps)
    state["model"]["extra"]["search_total_steps"] = int(target_steps)
    started = time.time()
    engine = RecedingEngine(
        spec=spec,
        rows=rows,
        train_view=train_view,
        work_dir=lane_dir,
        per_domain=per_domain,
        seed=seed,
        gpu=gpu,
        target_steps=target_steps,
    )
    try:
        while int(state["step"]) < target_steps:
            state = engine.advance(state, mode=mode)
            step = int(state["step"])
            if step in eval_steps or step == target_steps:
                _atomic_torch_save(lane_dir / f"step_{step}.pt", state)
                _atomic_torch_save(latest, state)
            if step in eval_steps:
                metric_path = lane_dir / f"metrics_step_{step}.json"
                if not metric_path.is_file():
                    metrics = _evaluate_state(
                        engine,
                        state,
                        rows=rows,
                        data_root=data_root,
                        seed=seed,
                        start_per_domain=eval_start,
                        count_per_domain=eval_count,
                        include_lpips=include_lpips,
                    )
                    _atomic_json(metric_path, metrics)
            if step % 80 == 0:
                recent = state["decisions"][-10:]
                accepted = sum(item["committed"] == "proposal" for item in recent)
                print(
                    f"{spec.candidate_id}/{mode} step={step}/{target_steps} "
                    f"recent_proposal={accepted}/{len(recent)}",
                    flush=True,
                )
    finally:
        engine.close()
    run_state = {
        "schema": "clean-unsb-search003-candidate-run-v1",
        "candidate": spec.to_dict(),
        "mode": mode,
        "final_step": int(state["step"]),
        "target_steps": int(target_steps),
        "per_domain": int(per_domain),
        "eval_steps": list(eval_steps),
        "decision_blocks": len(state["decisions"]),
        "proposal_blocks": sum(
            item["committed"] == "proposal" for item in state["decisions"]
        ),
        "paired_target_access_by_controller": False,
        "wall_seconds_this_call": time.time() - started,
        "confirmation20_opened": False,
    }
    _atomic_json(lane_dir / "RUN_STATE.json", run_state)
    return run_state
