"""Impulse-and-native-flow audit; diagnostic only, never a trainable policy."""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import types
from pathlib import Path

import numpy as np
import torch

from .catalog import AuditCheckpoint
from .geometry import initial_network_delta, model_gap_geometry
from .search001_compat import modules


class _NullDiagnostics:
    def log(self, **fields) -> None:
        del fields


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def exact_equal(first, second) -> bool:
    if torch.is_tensor(first) and torch.is_tensor(second):
        return first.dtype == second.dtype and first.shape == second.shape and torch.equal(first, second)
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return first.dtype == second.dtype and first.shape == second.shape and np.array_equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return tuple(first) == tuple(second) and all(exact_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(exact_equal(a, b) for a, b in zip(first, second))
    return first == second


def _lane(cell: AuditCheckpoint):
    protocol = modules()[0]
    return protocol.LaneSpec(
        name=f"search005_audit_{cell.checkpoint_id}",
        model=cell.model,
        family="search005_causal_probe",
    )


def _force_active(model, probe: str, step: int) -> None:
    model.set_search_step(step, max(step + 10_000_000, 10_000_000))
    if probe == "dt":
        model.opt.dtcov_lambda_schedule = "fixed"
        model.opt.dtcov_lambda = 0.001
        model.opt.dtcov_search_start_step = 0
        model.opt.dtcov_search_duration_steps = max(step + 10_000_000, 10_000_000)
    elif probe == "hj":
        model.opt.hj_enable = True
        model.opt.hj_search_start_step = 0
        model.opt.hj_search_duration_steps = max(step + 10_000_000, 10_000_000)
        model._hj_diag = _NullDiagnostics()
    elif probe == "hnek":
        from models.hnek.hnek_search import set_hnek_search_active

        set_hnek_search_active(model, True)


class PropagationEngine:
    """One proposal-class model transactionally executes every audit arm."""

    def __init__(
        self,
        *,
        cell: AuditCheckpoint,
        rows: list[dict],
        train_view: Path,
        work_dir: Path,
        seed: int,
        gpu: int,
        max_horizon: int,
    ) -> None:
        self.cell = cell
        self.rows = rows
        self.seed = int(seed)
        self.steps_per_epoch = 6 * int(cell.per_domain)
        self.target_steps = int(cell.step + max_horizon + 64)
        _, self.runtime, self.evaluator = modules()
        self.runtime.seed_everything(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            opt = self.runtime.build_options(
                _lane(cell),
                dataroot=Path(train_view),
                checkpoint_dir=Path(work_dir) / "option_records" / cell.checkpoint_id,
                steps_per_epoch=self.steps_per_epoch,
                total_steps=self.target_steps,
                seed=seed,
                gpu=gpu,
            )
            dataset_a, dataset_b = self.runtime.build_datasets(
                opt, rows, int(cell.per_domain)
            )
            self.stream_a = self.runtime.SerializableDataStream(dataset_a, seed=seed + 101)
            self.stream_b = self.runtime.SerializableDataStream(dataset_b, seed=seed + 202)
            self.model = self.runtime.build_model(
                opt, self.stream_a.next(), self.stream_b.next()
            )
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self._base_class = next(
            cls for cls in type(self.model).__mro__ if cls.__name__ == "SBModel"
        )

    def close(self) -> None:
        del self.model, self.stream_a, self.stream_b, self.dataset_a, self.dataset_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def source_state(self) -> dict:
        payload = torch.load(self.cell.method, map_location="cpu", weights_only=False)
        if payload.get("schema") != "clean-unsb-directional-v1":
            raise RuntimeError(f"checkpoint schema mismatch: {self.cell.method}")
        if int(payload["step"]) != int(self.cell.step):
            raise RuntimeError("checkpoint step mismatch")
        return {
            "schema": "clean-unsb-search005-audit-state-v1",
            "step": int(payload["step"]),
            "model": cpu_clone(payload["model"]),
            "rng": cpu_clone(payload["rng"]),
            "stream_a": cpu_clone(payload["stream_a"]),
            "stream_b": cpu_clone(payload["stream_b"]),
        }

    def load_state(self, state: dict, *, clock_state: dict | None = None) -> None:
        self.runtime.load_model_state(self.model, state["model"], load_extra=True)
        clock = state if clock_state is None else clock_state
        self.stream_a.load_state_dict(cpu_clone(clock["stream_a"]))
        self.stream_b.load_state_dict(cpu_clone(clock["stream_b"]))
        self.runtime.restore_rng(cpu_clone(clock["rng"]))
        self.current_step = int(state["step"])

    def capture(self, *, losses: dict[str, float] | None = None) -> dict:
        return {
            "schema": "clean-unsb-search005-audit-state-v1",
            "step": int(self.current_step),
            "model": cpu_clone(self.runtime.model_state(self.model)),
            "rng": self.runtime.capture_rng(),
            "stream_a": self.stream_a.state_dict(),
            "stream_b": self.stream_b.state_dict(),
            "mean_losses": losses or {},
        }

    @contextlib.contextmanager
    def operator(self, active: bool):
        if self.cell.probe == "hnek":
            from models.hnek.hnek_search import set_hnek_search_active

            set_hnek_search_active(self.model, active)
            yield
            return
        if active:
            yield
            return
        patched = ["compute_G_loss"]
        self.model.compute_G_loss = types.MethodType(
            self._base_class.compute_G_loss, self.model
        )
        if self.cell.probe == "hj":
            patched.append("calculate_NCE_loss")
            self.model.calculate_NCE_loss = types.MethodType(
                self._base_class.calculate_NCE_loss, self.model
            )
        try:
            yield
        finally:
            for name in patched:
                delattr(self.model, name)

    def run(
        self,
        state: dict,
        *,
        active: bool,
        steps: int,
        clock_state: dict | None = None,
    ) -> dict:
        self.load_state(state, clock_state=clock_state)
        sums: dict[str, float] = {}
        domains: list[str] = []
        bridge_times: list[int] = []
        with self.operator(active):
            for offset in range(int(steps)):
                step = int(state["step"]) + offset
                self.current_step = step
                self.model.set_train_epoch(1 + step // self.steps_per_epoch)
                if active:
                    _force_active(self.model, self.cell.probe, step)
                else:
                    self.model.set_search_step(step, self.target_steps)
                batch_a = self.stream_a.next()
                batch_b = self.stream_b.next()
                paths = list(batch_a.get("A_paths", []))
                if paths:
                    stem = Path(paths[0]).stem
                    domains.append(stem.split("__", 1)[0] if "__" in stem else "unknown")
                self.model.set_input(batch_a, batch_b)
                self.model.optimize_parameters()
                if hasattr(self.model, "time_idx"):
                    bridge_times.append(int(self.model.time_idx.reshape(-1)[0].item()))
                for key, value in self.model.get_current_losses().items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                completed = step + 1
                if completed % self.steps_per_epoch == 0:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.model.update_learning_rate()
            self.current_step = int(state["step"]) + int(steps)
            result = self.capture(
                losses={key: value / float(steps) for key, value in sums.items()}
            )
            result["batch_domains"] = domains
            result["bridge_times"] = bridge_times
            return result

    def evaluate(
        self,
        state: dict,
        *,
        data_root: Path,
        active: bool,
        count_per_domain: int,
    ) -> dict:
        parent_rng = self.runtime.capture_rng()
        try:
            self.load_state(state)
            with self.operator(active):
                return self.evaluator.evaluate(
                    self.model,
                    rows=self.rows,
                    data_root=Path(data_root),
                    start_per_domain=0,
                    count_per_domain=int(count_per_domain),
                    eval_seed=self.seed,
                    include_lpips=False,
                )
        finally:
            self.runtime.restore_rng(parent_rng)


def _compare_metrics(engine: PropagationEngine, method: dict, plain: dict, step: int) -> dict:
    result = engine.evaluator.compare(method, plain, step=step)
    result["available_to_algorithm"] = False
    result["computed_after_branch"] = True
    return result


def _clock_equal(first: dict, second: dict) -> bool:
    return (
        exact_equal(first["rng"], second["rng"])
        and exact_equal(first["stream_a"], second["stream_a"])
        and exact_equal(first["stream_b"], second["stream_b"])
    )


def hybrid_state(baseline: dict, pulse: dict, arm: str) -> dict:
    result = cpu_clone(baseline)
    if arm == "gf_parameters":
        for name in ("G", "F"):
            result["model"]["networks"][name] = cpu_clone(pulse["model"]["networks"][name])
    elif arm == "gf_parameters_moments":
        for name in ("G", "F"):
            result["model"]["networks"][name] = cpu_clone(pulse["model"]["networks"][name])
        for index in (0, 3):
            result["model"]["optimizers"][index] = cpu_clone(pulse["model"]["optimizers"][index])
    elif arm == "de_parameters_moments":
        for name in ("D", "E"):
            result["model"]["networks"][name] = cpu_clone(pulse["model"]["networks"][name])
        for index in (1, 2):
            result["model"]["optimizers"][index] = cpu_clone(pulse["model"]["optimizers"][index])
    elif arm == "all_networks":
        result["model"]["networks"] = cpu_clone(pulse["model"]["networks"])
    elif arm == "full_state":
        result["model"] = cpu_clone(pulse["model"])
    else:
        raise ValueError(f"unknown hybrid arm: {arm}")
    # Every attribution arm uses the same future data/RNG clock.
    result["rng"] = cpu_clone(baseline["rng"])
    result["stream_a"] = cpu_clone(baseline["stream_a"])
    result["stream_b"] = cpu_clone(baseline["stream_b"])
    return result


def run_propagation_audit(
    *,
    cell: AuditCheckpoint,
    rows: list[dict],
    train_view: Path,
    data_root: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    pulse_steps: int = 8,
    horizons: tuple[int, ...] = (8, 32, 200),
    eval_count: int = 10,
    attribution_steps: int = 32,
    run_attribution: bool = True,
) -> dict:
    if pulse_steps not in horizons or min(horizons) != pulse_steps:
        raise ValueError("horizons must start at pulse_steps")
    engine = PropagationEngine(
        cell=cell,
        rows=rows,
        train_view=train_view,
        work_dir=work_dir,
        seed=seed,
        gpu=gpu,
        max_horizon=max(max(horizons), pulse_steps + attribution_steps),
    )
    try:
        source = engine.source_state()
        baseline = engine.run(source, active=False, steps=pulse_steps)
        pulse = engine.run(source, active=True, steps=pulse_steps)
        print(f"  pulse complete: {cell.checkpoint_id} H={pulse_steps}", flush=True)
        initial_delta = initial_network_delta(baseline["model"], pulse["model"])
        baseline_metrics = engine.evaluate(
            baseline, data_root=data_root, active=False, count_per_domain=eval_count
        )
        pulse_native_view = engine.evaluate(
            pulse, data_root=data_root, active=False, count_per_domain=eval_count
        )
        pulse_active_view = engine.evaluate(
            pulse, data_root=data_root, active=True, count_per_domain=eval_count
        )
        trajectory = [{
            "horizon": int(pulse_steps),
            "native_flow_after_pulse_steps": 0,
            "network_gap": model_gap_geometry(
                baseline["model"], pulse["model"], initial_delta
            ),
            "pulse_native_view_delta": _compare_metrics(
                engine, pulse_native_view, baseline_metrics, cell.step + pulse_steps
            ),
            "pulse_active_view_delta": _compare_metrics(
                engine, pulse_active_view, baseline_metrics, cell.step + pulse_steps
            ),
            "future_clock_synchronized": False,
        }]
        completed = pulse_steps
        for horizon in sorted(h for h in horizons if h > pulse_steps):
            segment = int(horizon - completed)
            common_clock = baseline
            baseline_next = engine.run(
                baseline, active=False, steps=segment, clock_state=common_clock
            )
            pulse_next = engine.run(
                pulse, active=False, steps=segment, clock_state=common_clock
            )
            if not _clock_equal(baseline_next, pulse_next):
                raise AssertionError("native propagation clocks diverged")
            base_metrics = engine.evaluate(
                baseline_next, data_root=data_root, active=False,
                count_per_domain=eval_count,
            )
            pulse_metrics = engine.evaluate(
                pulse_next, data_root=data_root, active=False,
                count_per_domain=eval_count,
            )
            trajectory.append({
                "horizon": int(horizon),
                "native_flow_after_pulse_steps": int(horizon - pulse_steps),
                "network_gap": model_gap_geometry(
                    baseline_next["model"], pulse_next["model"], initial_delta
                ),
                "native_view_delta": _compare_metrics(
                    engine, pulse_metrics, base_metrics, cell.step + horizon
                ),
                "future_clock_synchronized": True,
            })
            print(f"  native propagation complete: {cell.checkpoint_id} H={horizon}", flush=True)
            baseline, pulse, completed = baseline_next, pulse_next, horizon

        attribution = []
        if run_attribution:
            # Recreate the pulse endpoints because ``baseline`` and ``pulse``
            # now point at the last propagation horizon.
            base_pulse = engine.run(source, active=False, steps=pulse_steps)
            method_pulse = engine.run(source, active=True, steps=pulse_steps)
            base_end = engine.run(
                base_pulse, active=False, steps=attribution_steps,
                clock_state=base_pulse,
            )
            base_eval = engine.evaluate(
                base_end, data_root=data_root, active=False,
                count_per_domain=eval_count,
            )
            for arm in (
                "gf_parameters",
                "gf_parameters_moments",
                "de_parameters_moments",
                "all_networks",
                "full_state",
            ):
                hybrid = hybrid_state(base_pulse, method_pulse, arm)
                end = engine.run(
                    hybrid, active=False, steps=attribution_steps,
                    clock_state=base_pulse,
                )
                if not _clock_equal(base_end, end):
                    raise AssertionError(f"attribution clock diverged: {arm}")
                metrics = engine.evaluate(
                    end, data_root=data_root, active=False,
                    count_per_domain=eval_count,
                )
                attribution.append({
                    "arm": arm,
                    "native_steps": int(attribution_steps),
                    "delta": _compare_metrics(
                        engine, metrics, base_eval,
                        cell.step + pulse_steps + attribution_steps,
                    ),
                    "network_gap": model_gap_geometry(
                        base_end["model"], end["model"]
                    ),
                    "future_clock_synchronized": True,
                })
                print(f"  attribution complete: {cell.checkpoint_id} arm={arm}", flush=True)
        return {
            "schema": "clean-unsb-search005-propagation-audit-v1",
            "checkpoint": cell.to_dict(),
            "pulse_steps": int(pulse_steps),
            "horizons": list(horizons),
            "trajectory": trajectory,
            "attribution_steps": int(attribution_steps),
            "component_attribution": attribution,
            "audit_only_not_candidate": True,
            "paired_metrics_available_to_algorithm": False,
            "confirmation20_opened": False,
        }
    finally:
        engine.close()


def run_component_attribution(
    *,
    cell: AuditCheckpoint,
    rows: list[dict],
    train_view: Path,
    data_root: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    pulse_steps: int = 8,
    attribution_steps: int = 32,
    eval_count: int = 10,
) -> dict:
    """Run only the diagnostic hybrid-state arms from a pulse endpoint."""
    engine = PropagationEngine(
        cell=cell,
        rows=rows,
        train_view=train_view,
        work_dir=work_dir,
        seed=seed,
        gpu=gpu,
        max_horizon=pulse_steps + attribution_steps,
    )
    try:
        source = engine.source_state()
        baseline = engine.run(source, active=False, steps=pulse_steps)
        pulse = engine.run(source, active=True, steps=pulse_steps)
        base_end = engine.run(
            baseline, active=False, steps=attribution_steps, clock_state=baseline
        )
        base_eval = engine.evaluate(
            base_end, data_root=data_root, active=False,
            count_per_domain=eval_count,
        )
        arms = []
        for arm in (
            "gf_parameters",
            "gf_parameters_moments",
            "de_parameters_moments",
            "all_networks",
            "full_state",
        ):
            hybrid = hybrid_state(baseline, pulse, arm)
            end = engine.run(
                hybrid, active=False, steps=attribution_steps,
                clock_state=baseline,
            )
            if not _clock_equal(base_end, end):
                raise AssertionError(f"attribution clock diverged: {arm}")
            metrics = engine.evaluate(
                end, data_root=data_root, active=False,
                count_per_domain=eval_count,
            )
            arms.append({
                "arm": arm,
                "delta": _compare_metrics(
                    engine, metrics, base_eval,
                    cell.step + pulse_steps + attribution_steps,
                ),
                "network_gap": model_gap_geometry(base_end["model"], end["model"]),
                "future_clock_synchronized": True,
            })
            print(f"  attribution {cell.checkpoint_id} arm={arm}", flush=True)
        return {
            "schema": "clean-unsb-search005-component-attribution-v1",
            "checkpoint": cell.to_dict(),
            "pulse_steps": int(pulse_steps),
            "native_steps": int(attribution_steps),
            "arms": arms,
            "hybrid_states_are_diagnostic_only": True,
            "paired_metrics_available_to_algorithm": False,
            "confirmation20_opened": False,
        }
    finally:
        engine.close()
