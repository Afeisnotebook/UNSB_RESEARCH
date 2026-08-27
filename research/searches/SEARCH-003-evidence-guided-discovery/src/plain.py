"""Efficient matched plain continuation after zero-intervention equivalence."""

from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import shutil
import time
from pathlib import Path

import torch

from .receding import (
    _atomic_json,
    _atomic_torch_save,
    _cpu_clone,
    _plain_lane,
    _strip_operator_extra,
    create_e0,
)
from .search001_compat import modules


def _initial_from_e0(e0: dict, target_steps: int) -> dict:
    return {
        "schema": "clean-unsb-search003-plain-state-v1",
        "step": 0,
        "target_steps": int(target_steps),
        "per_domain": int(e0["per_domain"]),
        "steps_per_epoch": int(e0["steps_per_epoch"]),
        "model": _cpu_clone(e0["model"]),
        "rng": _cpu_clone(e0["rng"]),
        "stream_a": _cpu_clone(e0["stream_a"]),
        "stream_b": _cpu_clone(e0["stream_b"]),
        "confirmation20_opened": False,
    }


def run_plain(
    *,
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
    bootstrap_state: Path | None = None,
    bootstrap_metrics_dir: Path | None = None,
    include_lpips: bool = False,
) -> dict:
    """Run true SB plain, optionally bootstrapped from an exact zero lane."""
    output_dir = Path(output_dir)
    lane_dir = output_dir / "plain"
    lane_dir.mkdir(parents=True, exist_ok=True)
    e0 = create_e0(
        path=output_dir / "e0.pt",
        rows=rows,
        train_view=train_view,
        per_domain=per_domain,
        seed=seed,
        gpu=gpu,
        target_steps=target_steps,
    )
    latest = lane_dir / "latest.pt"
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
    elif bootstrap_state is not None and Path(bootstrap_state).is_file():
        inherited = torch.load(bootstrap_state, map_location="cpu", weights_only=False)
        state = {
            "schema": "clean-unsb-search003-plain-state-v1",
            "step": int(inherited["step"]),
            "target_steps": int(target_steps),
            "per_domain": int(inherited["per_domain"]),
            "steps_per_epoch": int(inherited["steps_per_epoch"]),
            "model": _cpu_clone(inherited["model"]),
            "rng": _cpu_clone(inherited["rng"]),
            "stream_a": _cpu_clone(inherited["stream_a"]),
            "stream_b": _cpu_clone(inherited["stream_b"]),
            "bootstrap": {
                "source": str(Path(bootstrap_state).resolve()),
                "justification": "candidate zero-intervention was byte-exact to actual plain",
            },
            "confirmation20_opened": False,
        }
        _atomic_torch_save(latest, state)
        _atomic_torch_save(lane_dir / f"step_{state['step']}.pt", state)
        if bootstrap_metrics_dir is not None:
            for step in eval_steps:
                source_metric = Path(bootstrap_metrics_dir) / f"metrics_step_{step}.json"
                target_metric = lane_dir / f"metrics_step_{step}.json"
                if source_metric.is_file() and not target_metric.exists():
                    shutil.copy2(source_metric, target_metric)
    else:
        state = _initial_from_e0(e0, target_steps)
    state["target_steps"] = int(target_steps)

    _, runtime, evaluator = modules()
    runtime.seed_everything(seed)
    steps_per_epoch = 6 * int(per_domain)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = runtime.build_options(
            _plain_lane("search003_matched_plain"),
            dataroot=train_view,
            checkpoint_dir=lane_dir / "option_records",
            steps_per_epoch=steps_per_epoch,
            total_steps=target_steps,
            seed=seed,
            gpu=gpu,
        )
        dataset_a, dataset_b = runtime.build_datasets(opt, rows, per_domain)
        stream_a = runtime.SerializableDataStream(dataset_a, seed=seed + 101)
        stream_b = runtime.SerializableDataStream(dataset_b, seed=seed + 202)
        model = runtime.build_model(opt, stream_a.next(), stream_b.next())
    runtime.load_model_state(model, state["model"], load_extra=False)
    stream_a.load_state_dict(_cpu_clone(state["stream_a"]))
    stream_b.load_state_dict(_cpu_clone(state["stream_b"]))
    runtime.restore_rng(_cpu_clone(state["rng"]))
    started = time.time()
    try:
        for zero_step in range(int(state["step"]), int(target_steps)):
            model.set_train_epoch(1 + zero_step // steps_per_epoch)
            model.set_search_step(zero_step, target_steps)
            model.set_input(stream_a.next(), stream_b.next())
            model.optimize_parameters()
            completed = zero_step + 1
            if completed % steps_per_epoch == 0:
                with contextlib.redirect_stdout(io.StringIO()):
                    model.update_learning_rate()
            if completed in eval_steps or completed == target_steps:
                state.update({
                    "step": completed,
                    "target_steps": int(target_steps),
                    "model": _strip_operator_extra(
                        runtime.model_state(model),
                        step=completed,
                        target_steps=target_steps,
                    ),
                    "rng": runtime.capture_rng(),
                    "stream_a": stream_a.state_dict(),
                    "stream_b": stream_b.state_dict(),
                })
                _atomic_torch_save(lane_dir / f"step_{completed}.pt", state)
                _atomic_torch_save(latest, state)
            if completed in eval_steps:
                metric_path = lane_dir / f"metrics_step_{completed}.json"
                if not metric_path.is_file():
                    saved_rng = runtime.capture_rng()
                    metrics = evaluator.evaluate(
                        model,
                        rows=rows,
                        data_root=data_root,
                        start_per_domain=eval_start,
                        count_per_domain=eval_count,
                        eval_seed=seed,
                        include_lpips=include_lpips,
                    )
                    runtime.restore_rng(saved_rng)
                    metrics.update({
                        "step": completed,
                        "lane": "plain",
                        "confirmation20_opened": False,
                    })
                    _atomic_json(metric_path, metrics)
            if completed % 100 == 0:
                print(f"plain step={completed}/{target_steps}", flush=True)
    finally:
        del model, stream_a, stream_b, dataset_a, dataset_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "schema": "clean-unsb-search003-plain-run-v1",
        "final_step": int(state["step"]),
        "target_steps": int(target_steps),
        "bootstrap": state.get("bootstrap"),
        "paired_target_access_by_training": False,
        "wall_seconds_this_call": time.time() - started,
        "confirmation20_opened": False,
    }
    _atomic_json(lane_dir / "RUN_STATE.json", report)
    return report
