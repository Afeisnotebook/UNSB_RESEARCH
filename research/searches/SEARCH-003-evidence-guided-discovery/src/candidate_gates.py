"""Concrete engineering gates for discovered receding-horizon candidates."""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .receding import (
    RecedingEngine,
    RecedingSpec,
    _cpu_clone,
    _initial_candidate_state,
    _plain_lane,
    _strip_operator_extra,
    create_e0,
)
from .search001_compat import modules


def exact_equal(left, right, path: str = "root") -> tuple[bool, str | None]:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False, path
        return (True, None) if torch.equal(left.cpu(), right.cpu()) else (False, path)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return False, path
        return (True, None) if np.array_equal(left, right) else (False, path)
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return False, path
        if list(left) != list(right):
            return False, f"{path}.keys"
        for key in left:
            equal, mismatch = exact_equal(left[key], right[key], f"{path}.{key}")
            if not equal:
                return equal, mismatch
        return True, None
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False, path
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            equal, mismatch = exact_equal(l_item, r_item, f"{path}[{index}]")
            if not equal:
                return equal, mismatch
        return True, None
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True, None
    return ((True, None) if left == right else (False, path))


def _actual_plain_branch(
    *,
    source: dict,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    per_domain: int,
    seed: int,
    gpu: int,
    target_steps: int,
    horizon: int,
) -> dict:
    _, runtime, _ = modules()
    steps_per_epoch = 6 * int(per_domain)
    runtime.seed_everything(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = runtime.build_options(
            _plain_lane("search003_gate_plain"),
            dataroot=train_view,
            checkpoint_dir=work_dir / "plain_option_records",
            steps_per_epoch=steps_per_epoch,
            total_steps=target_steps,
            seed=seed,
            gpu=gpu,
        )
        dataset_a, dataset_b = runtime.build_datasets(opt, rows, per_domain)
        stream_a = runtime.SerializableDataStream(dataset_a, seed=seed + 101)
        stream_b = runtime.SerializableDataStream(dataset_b, seed=seed + 202)
        model = runtime.build_model(opt, stream_a.next(), stream_b.next())
    runtime.load_model_state(model, source["model"], load_extra=False)
    stream_a.load_state_dict(_cpu_clone(source["stream_a"]))
    stream_b.load_state_dict(_cpu_clone(source["stream_b"]))
    runtime.restore_rng(_cpu_clone(source["rng"]))
    start = int(source["step"])
    for offset in range(horizon):
        zero_step = start + offset
        model.set_train_epoch(1 + zero_step // steps_per_epoch)
        model.set_search_step(zero_step, target_steps)
        model.set_input(stream_a.next(), stream_b.next())
        model.optimize_parameters()
        completed = zero_step + 1
        if completed % steps_per_epoch == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                model.update_learning_rate()
    result = {
        "step": start + horizon,
        "model": _strip_operator_extra(
            runtime.model_state(model), step=start + horizon, target_steps=target_steps
        ),
        "rng": runtime.capture_rng(),
        "stream_a": stream_a.state_dict(),
        "stream_b": stream_b.state_dict(),
    }
    del model, stream_a, stream_b, dataset_a, dataset_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _transition_state(state: dict) -> dict:
    """Fields that must be exact after plain/reference or resume."""
    return {
        "step": state["step"],
        "model": state["model"],
        "rng": state["rng"],
        "stream_a": state["stream_a"],
        "stream_b": state["stream_b"],
    }


def run_candidate_gate(
    *,
    spec: RecedingSpec,
    output_dir: Path,
    rows: list[dict],
    train_view: Path,
    per_domain: int,
    seed: int,
    gpu: int,
) -> dict:
    """Test identity, immutable branching and full candidate resume."""
    target_steps = 16
    output_dir = Path(output_dir)
    gate_dir = output_dir / "candidate_gates" / spec.candidate_id
    e0 = create_e0(
        path=output_dir / "candidate_gates" / "e0.pt",
        rows=rows,
        train_view=train_view,
        per_domain=per_domain,
        seed=seed,
        gpu=gpu,
        target_steps=target_steps,
    )
    source = _initial_candidate_state(e0, spec, "full")
    source_before = _cpu_clone(source)
    actual_plain = _actual_plain_branch(
        source=source,
        rows=rows,
        train_view=train_view,
        work_dir=gate_dir,
        per_domain=per_domain,
        seed=seed,
        gpu=gpu,
        target_steps=target_steps,
        horizon=spec.horizon,
    )
    engine = RecedingEngine(
        spec=spec,
        rows=rows,
        train_view=train_view,
        work_dir=gate_dir,
        per_domain=per_domain,
        seed=seed,
        gpu=gpu,
        target_steps=target_steps,
    )
    try:
        reference = engine.branch(source, proposal=False)
        source_immutable, source_mismatch = exact_equal(source, source_before)
        zero_exact, zero_mismatch = exact_equal(
            _transition_state(reference), _transition_state(actual_plain)
        )
        continuous_first = engine.advance(source, mode="full")
        continuous_final = engine.advance(continuous_first, mode="full")
    finally:
        engine.close()

    checkpoint = gate_dir / "resume_boundary.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(continuous_first, checkpoint)
    restored_first = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resume_boundary_exact, boundary_mismatch = exact_equal(
        continuous_first, restored_first
    )
    resumed_engine = RecedingEngine(
        spec=spec,
        rows=rows,
        train_view=train_view,
        work_dir=gate_dir,
        per_domain=per_domain,
        seed=seed,
        gpu=gpu,
        target_steps=target_steps,
    )
    try:
        resumed_final = resumed_engine.advance(restored_first, mode="full")
    finally:
        resumed_engine.close()
    resume_exact, resume_mismatch = exact_equal(continuous_final, resumed_final)
    report = {
        "schema": "clean-unsb-search003-candidate-gate-v1",
        "candidate": spec.to_dict(),
        "source_immutable_after_virtual_branch": source_immutable,
        "source_immutable_mismatch": source_mismatch,
        "zero_intervention_plain_exact": zero_exact,
        "zero_intervention_mismatch": zero_mismatch,
        "checkpoint_roundtrip_exact": resume_boundary_exact,
        "checkpoint_roundtrip_mismatch": boundary_mismatch,
        "controller_resume_exact": resume_exact,
        "controller_resume_mismatch": resume_mismatch,
        "paired_target_access": False,
        "confirmation20_opened": False,
    }
    report["status"] = (
        "PASS" if all((source_immutable, zero_exact, resume_boundary_exact, resume_exact))
        else "FAIL"
    )
    path = gate_dir / "CANDIDATE_GATE.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return report
