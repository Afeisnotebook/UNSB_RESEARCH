"""Same-state multi-batch correction bias--variance audit."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .catalog import AuditCheckpoint
from .propagation import PropagationEngine, cpu_clone


def _floating_deltas(first: dict, second: dict):
    for name in first["networks"]:
        for key, value in first["networks"][name].items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                yield name, key, (
                    second["networks"][name][key].detach().cpu()
                    - value.detach().cpu()
                )


def _displacement_cosine(start_a: dict, end_a: dict, start_b: dict, end_b: dict) -> float:
    aa = 0.0
    bb = 0.0
    ab = 0.0
    by_key_b = {
        (name, key): delta for name, key, delta in _floating_deltas(start_b, end_b)
    }
    for name, key, first in _floating_deltas(start_a, end_a):
        second = by_key_b[(name, key)]
        first64 = first.double()
        second64 = second.double()
        aa += float(torch.sum(first64 * first64).item())
        bb += float(torch.sum(second64 * second64).item())
        ab += float(torch.sum(first64 * second64).item())
    denominator = math.sqrt(aa * bb)
    return max(-1.0, min(1.0, ab / denominator)) if denominator > 0.0 else 0.0


def _replicate_clock(engine: PropagationEngine, source: dict, replicate: int, seed: int) -> dict:
    engine.load_state(source)
    # Shift to a different deterministic unpaired batch without updating the model.
    for _ in range(int(replicate)):
        engine.stream_a.next()
        engine.stream_b.next()
    engine.runtime.seed_everything(int(seed + 10_003 * (replicate + 1)))
    clock = cpu_clone(source)
    clock["stream_a"] = engine.stream_a.state_dict()
    clock["stream_b"] = engine.stream_b.state_dict()
    clock["rng"] = engine.runtime.capture_rng()
    return clock


def run_variance_audit(
    *,
    cell: AuditCheckpoint,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    replicates: int = 8,
) -> dict:
    if replicates < 4:
        raise ValueError("at least four replicates are required")
    engine = PropagationEngine(
        cell=cell,
        rows=rows,
        train_view=train_view,
        work_dir=work_dir,
        seed=seed,
        gpu=gpu,
        max_horizon=4,
    )
    try:
        source = engine.source_state()
        means: dict[str, dict[str, torch.Tensor]] = {}
        second_moment: dict[str, float] = {}
        same_native_cosines = []
        future_native_cosines = []
        samples = []
        for replicate in range(int(replicates)):
            clock = _replicate_clock(engine, source, replicate, seed)
            plain = engine.run(source, active=False, steps=1, clock_state=clock)
            proposal = engine.run(source, active=True, steps=1, clock_state=clock)
            future = engine.run(plain, active=False, steps=1, clock_state=plain)
            correction_start = plain["model"]
            correction_end = proposal["model"]
            same_native_cosines.append(_displacement_cosine(
                source["model"], plain["model"], correction_start, correction_end
            ))
            future_native_cosines.append(_displacement_cosine(
                correction_start, correction_end, plain["model"], future["model"]
            ))
            correction_norm_sq = 0.0
            by_network_norm_sq: dict[str, float] = {}
            for name, key, delta in _floating_deltas(correction_start, correction_end):
                means.setdefault(name, {})
                if key not in means[name]:
                    means[name][key] = torch.zeros_like(delta)
                means[name][key].add_((delta - means[name][key]) / float(replicate + 1))
                norm_sq = float(torch.sum(delta.double() * delta.double()).item())
                second_moment[name] = second_moment.get(name, 0.0) + norm_sq / float(replicates)
                by_network_norm_sq[name] = by_network_norm_sq.get(name, 0.0) + norm_sq
                correction_norm_sq += norm_sq
            samples.append({
                "replicate": replicate,
                "domain": proposal.get("batch_domains", [None])[0],
                "bridge_time": proposal.get("bridge_times", [None])[0],
                "correction_norm": math.sqrt(correction_norm_sq),
                "network_correction_norm": {
                    name: math.sqrt(value) for name, value in by_network_norm_sq.items()
                },
                "same_batch_native_cosine": same_native_cosines[-1],
                "next_batch_native_cosine": future_native_cosines[-1],
            })
            print(
                f"  variance {cell.checkpoint_id} replicate={replicate + 1}/{replicates}",
                flush=True,
            )
        network_rows = {}
        total_mean_sq = 0.0
        total_second = 0.0
        for name, values in means.items():
            mean_sq = sum(
                float(torch.sum(value.double() * value.double()).item())
                for value in values.values()
            )
            second = float(second_moment[name])
            variance = max(0.0, second - mean_sq)
            total_mean_sq += mean_sq
            total_second += second
            network_rows[name] = {
                "mean_correction_norm": math.sqrt(mean_sq),
                "correction_second_moment": second,
                "variance_trace": variance,
                "variance_fraction": variance / second if second > 0.0 else 0.0,
                "mean_to_variance_ratio": mean_sq / variance if variance > 0.0 else None,
            }
        total_variance = max(0.0, total_second - total_mean_sq)
        variance_fraction = total_variance / total_second if total_second > 0.0 else 0.0
        return {
            "schema": "clean-unsb-search005-bias-variance-v1",
            "checkpoint": cell.to_dict(),
            "replicates": int(replicates),
            "samples": samples,
            "global": {
                "mean_correction_norm": math.sqrt(total_mean_sq),
                "correction_second_moment": total_second,
                "variance_trace": total_variance,
                "variance_fraction": variance_fraction,
                "mean_to_variance_ratio": (
                    total_mean_sq / total_variance if total_variance > 0.0 else None
                ),
                "mean_same_batch_native_cosine": sum(same_native_cosines) / len(same_native_cosines),
                "mean_next_batch_native_cosine": sum(future_native_cosines) / len(future_native_cosines),
                "variance_dominated": bool(variance_fraction >= 0.75),
            },
            "networks": network_rows,
            "paired_metrics_accessed": False,
            "audit_only_not_candidate": True,
            "confirmation20_opened": False,
        }
    finally:
        engine.close()
