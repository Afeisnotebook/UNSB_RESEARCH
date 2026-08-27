"""Matched local experiments for SEARCH-005 Generation 1."""

from __future__ import annotations

import json
import copy
import time
from pathlib import Path

import torch

from .candidate_runtime import ACMP, BCAVP, BCNRP, CNDRP, ELIPRC, FBCMP, NPOOA, PCOA, PHCRP, PHRSUP, PLAIN, CandidateSpec, create_e0, prepare_lane
from .search001_compat import modules


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lane_metrics(lane_dir: Path, step: int) -> dict | None:
    path = Path(lane_dir) / f"metrics_step_{step}.json"
    return _read_json(path) if path.is_file() else None


def run_lane(
    spec: CandidateSpec,
    *,
    rows: list[dict],
    train_view: Path,
    data_root: Path,
    stage_dir: Path,
    e0: dict,
    per_domain: int,
    total_steps: int,
    eval_steps: tuple[int, ...],
    eval_count: int,
    seed: int,
    gpu: int,
    stage_name: str = "generation1_micro",
    eval_start: int = 0,
) -> dict:
    """Run or exactly resume one lane from the shared e0 state."""
    _, runtime, evaluator = modules()
    model, stream_a, stream_b = prepare_lane(
        spec,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=stage_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    lane_dir = stage_dir / spec.lane_name
    lane_dir.mkdir(parents=True, exist_ok=True)
    latest = lane_dir / "latest.pt"
    start_step = 0
    if latest.is_file():
        payload = runtime.load_checkpoint(
            latest,
            model=model,
            spec=spec.lane_spec(),
            stream_a=stream_a,
            stream_b=stream_b,
        )
        start_step = int(payload["step"])

    started = time.time()
    losses = []
    completed_step = start_step
    for zero_step in range(start_step, total_steps):
        model.set_train_epoch(1 + zero_step // (6 * per_domain))
        model.set_search_step(zero_step, total_steps)
        model.set_input(stream_a.next(), stream_b.next())
        model.optimize_parameters()
        completed = zero_step + 1
        completed_step = completed
        if completed % (6 * per_domain) == 0:
            model.update_learning_rate()
        if completed % 50 == 0 or completed in eval_steps:
            row = {"step": completed, **model.get_current_losses()}
            if hasattr(model, "_search005_cndrp_last"):
                row["operator"] = copy.deepcopy(model._search005_cndrp_last)
            if hasattr(model, "_search005_acmp_last"):
                row["operator"] = copy.deepcopy(model._search005_acmp_last)
            if hasattr(model, "_search005_bcavp_last"):
                row["operator"] = copy.deepcopy(model._search005_bcavp_last)
            if hasattr(model, "_search005_phcrp_last"):
                row["operator"] = copy.deepcopy(model._search005_phcrp_last)
            if hasattr(model, "_search005_phrsup_last"):
                row["operator"] = copy.deepcopy(model._search005_phrsup_last)
            if hasattr(model, "_search005_pcoa_last"):
                row["operator"] = copy.deepcopy(model._search005_pcoa_last)
            if hasattr(model, "_search005_npooa_last"):
                row["operator"] = copy.deepcopy(model._search005_npooa_last)
            losses.append(row)
            print(
                f"MICRO {spec.candidate_id} step={completed}/{total_steps} "
                f"G={float(row.get('G', float('nan'))):.6f}",
                flush=True,
            )
        if completed in eval_steps or completed == total_steps:
            runtime.save_checkpoint(
                lane_dir / f"step_{completed}.pt",
                model=model,
                spec=spec.lane_spec(),
                step=completed,
                target_steps=total_steps,
                stream_a=stream_a,
                stream_b=stream_b,
                metadata={
                    "search": "SEARCH-005",
                    "candidate_id": spec.candidate_id,
                    "stage": stage_name,
                    "confirmation20_opened": False,
                },
            )
            runtime.save_checkpoint(
                latest,
                model=model,
                spec=spec.lane_spec(),
                step=completed,
                target_steps=total_steps,
                stream_a=stream_a,
                stream_b=stream_b,
                metadata={
                    "search": "SEARCH-005",
                    "candidate_id": spec.candidate_id,
                    "stage": stage_name,
                    "confirmation20_opened": False,
                },
            )
        if completed in eval_steps:
            metric_path = lane_dir / f"metrics_step_{completed}.json"
            if not metric_path.is_file():
                metrics = evaluator.evaluate(
                    model,
                    rows=rows,
                    data_root=data_root,
                    start_per_domain=eval_start,
                    count_per_domain=eval_count,
                    eval_seed=seed,
                    include_lpips=False,
                )
                metrics.update({
                    "step": completed,
                    "candidate_id": spec.candidate_id,
                    "spec": spec.lane_spec().to_dict(),
                })
                runtime.write_json(metric_path, metrics)
                print(
                    f"MICRO {spec.candidate_id} eval={completed} "
                    f"PSNR={metrics['macro_psnr']:.6f}",
                    flush=True,
                )

    runtime.write_json(lane_dir / "RUN_STATE.json", {
        "schema": "clean-unsb-search005-run-state-v1",
        "candidate_id": spec.candidate_id,
        "start_step_this_call": start_step,
        "final_step": completed_step,
        "target_steps": total_steps,
        "per_domain": per_domain,
        "eval_steps": list(eval_steps),
        "wall_seconds_this_call": time.time() - started,
        "losses_this_call": losses,
        "confirmation20_opened": False,
    })
    del model
    torch.cuda.empty_cache()
    return {
        "candidate_id": spec.candidate_id,
        "final_step": completed_step,
        "metrics": {
            str(step): _lane_metrics(lane_dir, step)
            for step in eval_steps
            if _lane_metrics(lane_dir, step) is not None
        },
    }


def run_candidate_micro(
    spec: CandidateSpec,
    *,
    rows: list[dict],
    train_view: Path,
    data_root: Path,
    output: Path,
    gate_path: Path,
    seed: int,
    gpu: int,
    total_steps: int = 800,
    eval_steps: tuple[int, ...] = (400, 800),
    stage_name: str = "generation1_micro",
    result_schema: str = "clean-unsb-search005-generation1-micro-v1",
    per_domain: int = 25,
    eval_count: int = 10,
    eval_start: int = 0,
) -> dict:
    """Matched plain/candidate micro run; paired metrics are post-hoc only."""
    gate = _read_json(gate_path)
    if gate.get("candidate_id") != spec.candidate_id or gate.get("passed") is not True:
        raise RuntimeError(f"{spec.candidate_id} engineering gate is not open")
    if max(eval_steps) > total_steps or min(eval_steps) <= 0:
        raise ValueError("invalid micro evaluation schedule")

    stage_dir = Path(output) / stage_name
    e0 = create_e0(
        stage_dir / "e0.pt",
        rows=rows,
        train_view=train_view,
        option_dir=stage_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    plain = run_lane(
        PLAIN,
        rows=rows,
        train_view=train_view,
        data_root=data_root,
        stage_dir=stage_dir,
        e0=e0,
        per_domain=per_domain,
        total_steps=total_steps,
        eval_steps=eval_steps,
        eval_count=eval_count,
        seed=seed,
        gpu=gpu,
        stage_name=stage_name,
        eval_start=eval_start,
    )
    candidate = run_lane(
        spec,
        rows=rows,
        train_view=train_view,
        data_root=data_root,
        stage_dir=stage_dir,
        e0=e0,
        per_domain=per_domain,
        total_steps=total_steps,
        eval_steps=eval_steps,
        eval_count=eval_count,
        seed=seed,
        gpu=gpu,
        stage_name=stage_name,
        eval_start=eval_start,
    )

    _, _, evaluator = modules()
    trajectory = []
    for step in eval_steps:
        plain_metric = plain["metrics"].get(str(step))
        candidate_metric = candidate["metrics"].get(str(step))
        if plain_metric is not None and candidate_metric is not None:
            trajectory.append(evaluator.compare(candidate_metric, plain_metric, step=step))
    result = {
        "schema": result_schema,
        "candidate_id": spec.candidate_id,
        "shared_e0": str((stage_dir / "e0.pt").resolve()),
        "seed": seed,
        "train_per_domain": per_domain,
        "discovery_per_domain": eval_count,
        "discovery_start_per_domain": eval_start,
        "total_steps": total_steps,
        "trajectory": trajectory,
        "plain_absolute": {
            step: plain["metrics"][step]["macro_psnr"] for step in plain["metrics"]
        },
        "candidate_absolute": {
            step: candidate["metrics"][step]["macro_psnr"]
            for step in candidate["metrics"]
        },
        "paired_metrics_used_for_training_or_control": False,
        "uses_fixed_window": False,
        "confirmation20_opened": False,
    }
    modules()[1].write_json(stage_dir / "RESULT.json", result)
    return result


def run_eliprc_micro(**kwargs) -> dict:
    return run_candidate_micro(ELIPRC, **kwargs)


def run_cndrp_micro(**kwargs) -> dict:
    return run_candidate_micro(CNDRP, **kwargs)


def run_acmp_micro(**kwargs) -> dict:
    return run_candidate_micro(ACMP, **kwargs)


def run_fbcmp_micro(**kwargs) -> dict:
    return run_candidate_micro(FBCMP, **kwargs)


def run_bcavp_micro(**kwargs) -> dict:
    return run_candidate_micro(BCAVP, **kwargs)


def run_phcrp_micro(**kwargs) -> dict:
    return run_candidate_micro(PHCRP, **kwargs)


def run_phrsup_micro(**kwargs) -> dict:
    return run_candidate_micro(PHRSUP, **kwargs)


def run_bcnrp_micro(**kwargs) -> dict:
    return run_candidate_micro(BCNRP, **kwargs)


def run_pcoa_micro(**kwargs) -> dict:
    return run_candidate_micro(PCOA, **kwargs)


def run_npooa_micro(**kwargs) -> dict:
    return run_candidate_micro(NPOOA, **kwargs)


def run_pcoa_small_view(**kwargs) -> dict:
    """Independent 2400-step PCOA trajectory; never resumes the 800-step micro lane."""
    return run_candidate_micro(
        PCOA,
        stage_name="generation1_small_2400",
        result_schema="clean-unsb-search005-generation1-small-view-v1",
        **kwargs,
    )


def run_npooa_small_view(**kwargs) -> dict:
    """Independent 2400-step Generation-2 trajectory."""
    return run_candidate_micro(
        NPOOA,
        stage_name="generation2_small_2400",
        result_schema="clean-unsb-search005-generation2-small-view-v1",
        **kwargs,
    )


def run_pcoa_full_view(**kwargs) -> dict:
    """Frozen full100 matched entrypoint; not evidence that PCOA was promoted."""
    return run_candidate_micro(
        PCOA,
        stage_name="generation3_full_pcoa",
        result_schema="clean-unsb-search005-full-view-v1",
        per_domain=100,
        eval_count=70,
        eval_start=10,
        **kwargs,
    )
