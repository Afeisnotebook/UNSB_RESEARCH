#!/usr/bin/env python3
"""Run the frozen clean-UNSB directional search and always emit one candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parents[2]
if str(SEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.evaluate import compare, evaluate  # noqa: E402
from src.protocol import LaneSpec, classify, frozen_lanes, ranked, synthesize  # noqa: E402
from src.runtime import (  # noqa: E402
    SerializableDataStream,
    build_datasets,
    build_model,
    build_options,
    capture_rng,
    load_checkpoint,
    load_model_state,
    model_state,
    read_manifest,
    restore_rng,
    save_checkpoint,
    seed_everything,
    write_json,
)


MANIFEST_SHA256 = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def verify_protocol(args, rows: list[dict]) -> dict:
    actual_hash = sha256(args.manifest)
    if actual_hash != MANIFEST_SHA256 and not args.allow_manifest_mismatch:
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_hash}")
    counts = {}
    for domain in sorted({row["domain"] for row in rows}):
        counts[domain] = {
            split: sum(row["domain"] == domain and row["split"] == split for row in rows)
            for split in ("train", "discovery", "confirmation")
        }
        if counts[domain] != {"train": 100, "discovery": 80, "confirmation": 20}:
            raise RuntimeError(f"split contract mismatch for {domain}: {counts[domain]}")
    required = [args.data_root / row[key] for row in rows for key in ("input_relpath", "target_relpath")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"raw paired data is incomplete, first missing: {missing[0]}")
    return {
        "git_commit": git_commit(),
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": actual_hash,
        "domain_split_counts": counts,
        "confirmation20_opened": False,
    }


def create_e0(
    *,
    args,
    rows,
    stage_dir: Path,
    per_domain: int,
    target_steps: int,
) -> dict:
    path = stage_dir / "e0.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    spec = LaneSpec("plain")
    steps_per_epoch = 6 * per_domain
    seed_everything(args.seed)
    opt = build_options(
        spec,
        dataroot=args.train_view,
        checkpoint_dir=stage_dir / "option_records",
        steps_per_epoch=steps_per_epoch,
        total_steps=target_steps,
        seed=args.seed,
        gpu=args.gpu,
    )
    dataset_a, dataset_b = build_datasets(opt, rows, per_domain)
    stream_a = SerializableDataStream(dataset_a, seed=args.seed + 101)
    stream_b = SerializableDataStream(dataset_b, seed=args.seed + 202)
    ddi_a, ddi_b = stream_a.next(), stream_b.next()
    model = build_model(opt, ddi_a, ddi_b)
    payload = {
        "schema": "clean-unsb-e0-v1",
        "model": model_state(model),
        "rng": capture_rng(),
        "stream_a": stream_a.state_dict(),
        "stream_b": stream_b.state_dict(),
        "per_domain": per_domain,
        "steps_per_epoch": steps_per_epoch,
        "target_steps": target_steps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    del model
    torch.cuda.empty_cache()
    return payload


def prepare_lane(
    *,
    args,
    rows,
    stage_dir: Path,
    spec: LaneSpec,
    per_domain: int,
    target_steps: int,
    schedule_steps: int,
    e0: dict,
):
    steps_per_epoch = 6 * per_domain
    seed_everything(args.seed)
    opt = build_options(
        spec,
        dataroot=args.train_view,
        checkpoint_dir=stage_dir / "option_records",
        steps_per_epoch=steps_per_epoch,
        total_steps=schedule_steps,
        seed=args.seed,
        gpu=args.gpu,
    )
    dataset_a, dataset_b = build_datasets(opt, rows, per_domain)
    stream_a = SerializableDataStream(dataset_a, seed=args.seed + 101)
    stream_b = SerializableDataStream(dataset_b, seed=args.seed + 202)
    model = build_model(opt, stream_a.next(), stream_b.next())
    load_model_state(model, e0["model"], load_extra=False)
    stream_a.load_state_dict(e0["stream_a"])
    stream_b.load_state_dict(e0["stream_b"])
    restore_rng(e0["rng"])
    model.set_search_step(0, schedule_steps)
    if "lbst" in spec.mechanisms:
        model.reset_lbst_teacher()
    return model, stream_a, stream_b


def run_lane(
    *,
    args,
    protocol,
    rows,
    stage_dir: Path,
    spec: LaneSpec,
    per_domain: int,
    target_steps: int,
    eval_steps: list[int],
    eval_start: int,
    eval_count: int,
    include_lpips: bool,
    schedule_steps: int | None = None,
    initial_checkpoint: Path | None = None,
    stop_after: int | None = None,
) -> dict:
    schedule_steps = int(schedule_steps or target_steps)
    e0 = create_e0(
        args=args, rows=rows, stage_dir=stage_dir, per_domain=per_domain,
        target_steps=schedule_steps,
    )
    model, stream_a, stream_b = prepare_lane(
        args=args, rows=rows, stage_dir=stage_dir, spec=spec,
        per_domain=per_domain, target_steps=target_steps,
        schedule_steps=schedule_steps, e0=e0,
    )
    lane_dir = stage_dir / spec.name
    lane_dir.mkdir(parents=True, exist_ok=True)
    latest = lane_dir / "latest.pt"
    start_step = 0
    source = latest if latest.is_file() else initial_checkpoint
    if source is not None and Path(source).is_file():
        payload = load_checkpoint(
            Path(source), model=model, spec=spec, stream_a=stream_a, stream_b=stream_b
        )
        start_step = int(payload["step"])
    end_step = min(target_steps, stop_after) if stop_after is not None else target_steps
    started = time.time()
    losses = []
    completed_step = start_step
    last_saved_step = start_step if latest.is_file() else -1
    for zero_step in range(start_step, end_step):
        if args.deadline and time.time() >= args.deadline:
            break
        physical_epoch = 1 + zero_step // (6 * per_domain)
        model.set_train_epoch(physical_epoch)
        model.set_search_step(zero_step, schedule_steps)
        model.set_input(stream_a.next(), stream_b.next())
        model.optimize_parameters()
        completed = zero_step + 1
        completed_step = completed
        if completed % (6 * per_domain) == 0:
            model.update_learning_rate()
        if completed % args.log_every == 0 or completed in eval_steps:
            losses.append({"step": completed, **model.get_current_losses()})
        should_save = completed in eval_steps or completed == end_step
        if should_save:
            save_checkpoint(
                lane_dir / f"step_{completed}.pt",
                model=model, spec=spec, step=completed, target_steps=target_steps,
                stream_a=stream_a, stream_b=stream_b,
                metadata={"protocol": protocol, "schedule_steps": schedule_steps},
            )
            last_saved_step = completed
            save_checkpoint(
                latest,
                model=model, spec=spec, step=completed, target_steps=target_steps,
                stream_a=stream_a, stream_b=stream_b,
                metadata={"protocol": protocol, "schedule_steps": schedule_steps},
            )
        if completed in eval_steps:
            metric_path = lane_dir / f"metrics_step_{completed}.json"
            if not metric_path.is_file():
                metrics = evaluate(
                    model, rows=rows, data_root=args.data_root,
                    start_per_domain=eval_start, count_per_domain=eval_count,
                    eval_seed=args.seed, include_lpips=include_lpips,
                )
                metrics["step"] = completed
                metrics["spec"] = spec.to_dict()
                write_json(metric_path, metrics)
    if completed_step > last_saved_step:
        save_checkpoint(
            lane_dir / f"step_{completed_step}.pt",
            model=model, spec=spec, step=completed_step, target_steps=target_steps,
            stream_a=stream_a, stream_b=stream_b,
            metadata={"protocol": protocol, "schedule_steps": schedule_steps},
        )
        save_checkpoint(
            latest,
            model=model, spec=spec, step=completed_step, target_steps=target_steps,
            stream_a=stream_a, stream_b=stream_b,
            metadata={"protocol": protocol, "schedule_steps": schedule_steps},
        )
    final_step = completed_step
    write_json(
        lane_dir / "RUN_STATE.json",
        {
            "spec": spec.to_dict(), "start_step": start_step, "final_step": final_step,
            "target_steps": target_steps, "schedule_steps": schedule_steps,
            "wall_seconds_this_call": time.time() - started, "losses": losses,
            "confirmation20_opened": False,
        },
    )
    del model
    torch.cuda.empty_cache()
    return lane_summary(stage_dir, spec, eval_steps)


def read_metrics(stage_dir: Path, spec: LaneSpec, step: int) -> dict | None:
    path = stage_dir / spec.name / f"metrics_step_{step}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def lane_summary(
    stage_dir: Path, spec: LaneSpec, eval_steps: list[int], plain_spec: LaneSpec | None = None
) -> dict:
    plain_spec = plain_spec or LaneSpec("plain")
    trajectory = []
    for step in eval_steps:
        method = read_metrics(stage_dir, spec, step)
        plain = read_metrics(stage_dir, plain_spec, step)
        if method is None or plain is None:
            continue
        trajectory.append(compare(method, plain, step=step))
    deltas = [row["macro_psnr_delta"] for row in trajectory]
    return {
        "name": spec.name,
        "spec": spec.to_dict(),
        "trajectory": trajectory,
        "peak_to_final_rollback": (max(deltas) - deltas[-1] if deltas else float("inf")),
    }


def write_stage_summary(stage_dir: Path, summaries: list[dict]) -> list[dict]:
    ordering = ranked(summaries)
    write_json(
        stage_dir / "RANKING.json",
        {
            "ranking": ordering,
            "ordering_rule": [
                "late_three_macro_psnr_delta", "final_psnr", "positive_domains",
                "worst_domain", "peak_to_final_rollback",
            ],
            "confirmation20_opened": False,
        },
    )
    return ordering


def stage1(args, protocol, rows) -> tuple[list[LaneSpec], list[dict]]:
    stage_dir = args.output / "stage1_direction_screen"
    target = args.stage1_steps
    eval_steps = args.stage1_eval
    lanes = frozen_lanes()
    summaries = {}
    for spec in lanes:
        summaries[spec.name] = run_lane(
            args=args, protocol=protocol, rows=rows, stage_dir=stage_dir, spec=spec,
            per_domain=args.stage1_train_per_domain, target_steps=target,
            eval_steps=eval_steps, eval_start=0, eval_count=args.stage1_eval_per_domain,
            include_lpips=False,
        )
    base_rank = ranked([summaries[spec.name] for spec in lanes if spec.name != "plain"])
    spec_by_name = {spec.name: spec for spec in lanes}
    new_rank = [row for row in base_rank if row["spec"]["family"] == "new"]
    legacy_rank = [row for row in base_rank if row["spec"]["family"] == "legacy"]
    new_best = spec_by_name[new_rank[0]["name"]]
    new_second = spec_by_name[new_rank[1]["name"]]
    legacy_best = spec_by_name[legacy_rank[0]["name"]]
    synths = [
        synthesize("new_new_synthesis", new_best, new_second),
        synthesize("legacy_new_synthesis", legacy_best, new_best),
    ]
    for spec in synths:
        summaries[spec.name] = run_lane(
            args=args, protocol=protocol, rows=rows, stage_dir=stage_dir, spec=spec,
            per_domain=args.stage1_train_per_domain, target_steps=target,
            eval_steps=eval_steps, eval_start=0, eval_count=args.stage1_eval_per_domain,
            include_lpips=False,
        )
    all_specs = lanes + synths
    ordering = write_stage_summary(
        stage_dir, [summaries[spec.name] for spec in all_specs if spec.name != "plain"]
    )
    write_json(
        stage_dir / "SYNTHESIS_SELECTION.json",
        {
            "best_new": new_best.to_dict(), "second_new": new_second.to_dict(),
            "best_legacy": legacy_best.to_dict(),
            "new_new_synthesis": synths[0].to_dict(),
            "legacy_new_synthesis": synths[1].to_dict(),
        },
    )
    return all_specs, ordering


def load_stage1_specs(args) -> tuple[list[LaneSpec], list[dict]]:
    stage_dir = args.output / "stage1_direction_screen"
    ranking = json.loads((stage_dir / "RANKING.json").read_text(encoding="utf-8"))["ranking"]
    specs = [LaneSpec(**{
        **row["spec"], "mechanisms": tuple(row["spec"]["mechanisms"])
    }) for row in ranking]
    return specs, ranking


def stage2(args, protocol, rows, stage1_specs, stage1_rank) -> tuple[list[LaneSpec], list[dict]]:
    stage_dir = args.output / "stage2_full_view"
    selected_names = [row["name"] for row in stage1_rank[:3]]
    by_name = {spec.name: spec for spec in stage1_specs}
    specs = [LaneSpec("plain")] + [by_name[name] for name in selected_names]
    summaries = []
    for spec in specs:
        result = run_lane(
            args=args, protocol=protocol, rows=rows, stage_dir=stage_dir, spec=spec,
            per_domain=args.stage2_train_per_domain, target_steps=args.stage2_steps,
            eval_steps=args.stage2_eval, eval_start=args.stage2_eval_start,
            eval_count=args.stage2_eval_per_domain, include_lpips=True,
        )
        if spec.name != "plain":
            summaries.append(result)
    ordering = write_stage_summary(stage_dir, summaries)
    return specs, ordering


def load_stage2_specs(args) -> tuple[list[LaneSpec], list[dict]]:
    stage_dir = args.output / "stage2_full_view"
    ranking = json.loads((stage_dir / "RANKING.json").read_text(encoding="utf-8"))["ranking"]
    specs = [LaneSpec("plain")] + [
        LaneSpec(**{**row["spec"], "mechanisms": tuple(row["spec"]["mechanisms"])})
        for row in ranking
    ]
    return specs, ranking


def stage3(args, protocol, rows, stage2_specs, stage2_rank) -> tuple[dict, list[dict]]:
    stage2_dir = args.output / "stage2_full_view"
    stage3_dir = args.output / "stage3_extension"
    winner_name = stage2_rank[0]["name"]
    by_name = {spec.name: spec for spec in stage2_specs}
    winner = by_name[winner_name]
    specs = [LaneSpec("plain"), winner]
    planned_eval_steps = [
        step for step in range(args.stage2_steps + args.stage3_eval_interval, args.stage3_steps + 1, args.stage3_eval_interval)
    ]
    completed_eval_steps = []
    last_pair_seconds = None
    # Extend in matched 2k-update pairs. A time limit is checked only between
    # pairs, so plain can never consume budget that the candidate does not get.
    for milestone in planned_eval_steps:
        if args.deadline is not None:
            remaining = args.deadline - time.time()
            if remaining <= 0 or (
                last_pair_seconds is not None and remaining < last_pair_seconds
            ):
                break
        pair_started = time.time()
        saved_deadline = args.deadline
        args.deadline = None
        try:
            for spec in specs:
                initial = stage2_dir / spec.name / f"step_{args.stage2_steps}.pt"
                run_lane(
                    args=args, protocol=protocol, rows=rows, stage_dir=stage3_dir, spec=spec,
                    per_domain=args.stage2_train_per_domain, target_steps=milestone,
                    schedule_steps=args.stage2_steps, eval_steps=[milestone],
                    eval_start=args.stage2_eval_start, eval_count=args.stage2_eval_per_domain,
                    include_lpips=True, initial_checkpoint=initial,
                )
        finally:
            args.deadline = saved_deadline
        completed_eval_steps.append(milestone)
        last_pair_seconds = time.time() - pair_started
    trajectory = []
    # Stage 2's last checkpoints plus every extension checkpoint are all matched.
    for step in args.stage2_eval:
        method = read_metrics(stage2_dir, winner, step)
        plain = read_metrics(stage2_dir, LaneSpec("plain"), step)
        if method and plain:
            trajectory.append(compare(method, plain, step=step))
    for step in completed_eval_steps:
        method = read_metrics(stage3_dir, winner, step)
        plain = read_metrics(stage3_dir, LaneSpec("plain"), step)
        if method and plain:
            trajectory.append(compare(method, plain, step=step))
    summary = {
        "name": winner.name, "spec": winner.to_dict(), "trajectory": trajectory,
        "peak_to_final_rollback": max(row["macro_psnr_delta"] for row in trajectory)
        - trajectory[-1]["macro_psnr_delta"],
    }
    write_stage_summary(stage3_dir, [summary])
    all_rank = [summary] + stage2_rank[1:]
    status = classify(summary, all_rank)
    alternatives = [row for row in stage2_rank if row["name"] != winner.name][:2]
    candidate = {
        "schema": "clean-unsb-candidate-v1",
        "status": status,
        "candidate": summary,
        "alternatives": alternatives,
        "code": {
            "git_commit": protocol["git_commit"],
            "runner": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "configuration": winner.to_dict(),
        "cost": {
            "estimated_generator_compute_multiplier": winner.estimated_g_flops_multiplier,
            "local_final_step": trajectory[-1]["step"],
        },
        "risks": [
            "single development seed only",
            "local 128x128 screen is not a final conclusion",
            "LPIPS may be unavailable if its local pretrained weights are absent",
        ],
        "reproduce_4090": {
            "seed": 2026,
            "milestones": [30000, 60000, 120000],
            "command": (
                "python research/searches/SEARCH-001-clean-directional/run_search.py "
                "--stage verify4090 --candidate runs/directional_search_20260826/CANDIDATE.json "
                "--gpu 0"
            ),
            "algorithm_changes_from_intermediate_psnr": "forbidden",
            "next_seeds_after_positive": [2027, 2028],
        },
        "confirmation20_opened": False,
    }
    write_json(args.output / "CANDIDATE.json", candidate)
    return candidate, alternatives


def recursive_equal(left, right, path="root") -> None:
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(f"ndarray mismatch at {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"keys mismatch at {path}")
        for key in left:
            recursive_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            recursive_equal(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")


def engineering_gate(args, protocol, rows) -> dict:
    gate_root = args.output / "engineering_gate"
    gate_args = copy.copy(args)
    gate_args.stage1_steps = 2
    gate_args.stage1_eval = [2]
    gate_args.stage1_train_per_domain = 2
    gate_args.stage1_eval_per_domain = 1
    plain = LaneSpec("plain")
    # Twin run: separate roots force independent construction from the same e0 contract.
    results = []
    checkpoints = []
    for name in ("twin_a", "twin_b"):
        root = gate_root / name
        spec = LaneSpec("plain")
        result = run_lane(
            args=gate_args, protocol=protocol, rows=rows, stage_dir=root, spec=spec,
            per_domain=2, target_steps=2, eval_steps=[2], eval_start=0,
            eval_count=1, include_lpips=False,
        )
        results.append(result)
        checkpoints.append(torch.load(root / "plain" / "step_2.pt", map_location="cpu", weights_only=False))
    for key in ("model", "rng", "stream_a", "stream_b"):
        recursive_equal(checkpoints[0][key], checkpoints[1][key], f"twin.{key}")
    metrics_a = json.loads((gate_root / "twin_a" / "plain" / "metrics_step_2.json").read_text())
    metrics_b = json.loads((gate_root / "twin_b" / "plain" / "metrics_step_2.json").read_text())
    recursive_equal(metrics_a, metrics_b, "twin.metrics")

    # Resume gate: 2 uninterrupted versus 1 + exact full-state resume to 2.
    resume_root = gate_root / "resume"
    run_lane(
        args=gate_args, protocol=protocol, rows=rows, stage_dir=resume_root, spec=plain,
        per_domain=2, target_steps=2, eval_steps=[2], eval_start=0,
        eval_count=1, include_lpips=False, stop_after=1,
    )
    run_lane(
        args=gate_args, protocol=protocol, rows=rows, stage_dir=resume_root, spec=plain,
        per_domain=2, target_steps=2, eval_steps=[2], eval_start=0,
        eval_count=1, include_lpips=False,
    )
    resumed = torch.load(resume_root / "plain" / "step_2.pt", map_location="cpu", weights_only=False)
    for key in ("model", "rng", "stream_a", "stream_b"):
        recursive_equal(checkpoints[0][key], resumed[key], f"resume.{key}")

    # PTQ exact mass and repeatability.
    ptq_root = gate_root / "ptq_contract"
    e0 = create_e0(args=gate_args, rows=rows, stage_dir=ptq_root, per_domain=2, target_steps=50)
    model, stream_a, stream_b = prepare_lane(
        args=gate_args, rows=rows, stage_dir=ptq_root,
        spec=LaneSpec("ptq", mechanisms=("ptq",), family="new"),
        per_domain=2, target_steps=50, schedule_steps=50, e0=e0,
    )
    ptq_values = [model._ptq_index(step, 5) for step in range(50)]
    if [ptq_values.count(index) for index in range(5)] != [25, 12, 6, 4, 3]:
        raise AssertionError("PTQ exact block mass failed")
    del model, stream_a, stream_b
    torch.cuda.empty_cache()

    report = {
        "status": "PASS",
        "plain_twin_exact": True,
        "resume_exact": True,
        "evaluation_repeat_exact": True,
        "ptq_block_counts": [25, 12, 6, 4, 3],
        "manifest_sha256": protocol["manifest_sha256"],
        "confirmation20_opened": False,
    }
    write_json(gate_root / "ENGINEERING_GATE.json", report)
    return report


def verify4090(args, protocol, rows) -> None:
    payload = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    value = payload["configuration"]
    spec = LaneSpec(**{**value, "mechanisms": tuple(value["mechanisms"])})
    stage_dir = args.output / "verify4090"
    milestones = [30000, 60000, 120000]
    for lane in (LaneSpec("plain"), spec):
        run_lane(
            args=args, protocol=protocol, rows=rows, stage_dir=stage_dir, spec=lane,
            per_domain=100, target_steps=120000, eval_steps=milestones,
            eval_start=0, eval_count=80, include_lpips=True,
        )
    summary = lane_summary(stage_dir, spec, milestones)
    write_stage_summary(stage_dir, [summary])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["gate", "stage1", "stage2", "stage3", "all", "verify4090"], default="all")
    parser.add_argument("--manifest", type=Path, default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\DATA_MANIFEST.csv"))
    parser.add_argument("--train-view", type=Path, default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\data_views_v2\allinone_100"))
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\UNSB_abl\full_dataset"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs" / "directional_search_20260826")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=16.0)
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--stage1-steps", type=int, default=1200)
    parser.add_argument("--stage1-eval", type=int, nargs="+", default=[400, 800, 1200])
    parser.add_argument("--stage1-train-per-domain", type=int, default=25)
    parser.add_argument("--stage1-eval-per-domain", type=int, default=10)
    parser.add_argument("--stage2-steps", type=int, default=4000)
    parser.add_argument("--stage2-eval", type=int, nargs="+", default=[1000, 2000, 3000, 4000])
    parser.add_argument("--stage2-train-per-domain", type=int, default=100)
    parser.add_argument("--stage2-eval-start", type=int, default=10)
    parser.add_argument("--stage2-eval-per-domain", type=int, default=70)
    parser.add_argument("--stage3-steps", type=int, default=12000)
    parser.add_argument("--stage3-eval-interval", type=int, default=2000)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    args.train_view = args.train_view.resolve()
    args.data_root = args.data_root.resolve()
    args.deadline = None
    return args


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest)
    protocol = verify_protocol(args, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "PROTOCOL_LOCK.json", protocol)
    if args.stage == "gate":
        print(json.dumps(engineering_gate(args, protocol, rows), indent=2))
        return 0
    if args.stage == "verify4090":
        if args.candidate is None:
            raise SystemExit("--candidate is required for verify4090")
        verify4090(args, protocol, rows)
        return 0
    if args.stage in ("stage1", "all"):
        stage1_specs, stage1_rank = stage1(args, protocol, rows)
    else:
        stage1_specs, stage1_rank = load_stage1_specs(args)
    if args.stage == "stage1":
        return 0
    if args.stage in ("stage2", "all"):
        stage2_specs, stage2_rank = stage2(args, protocol, rows, stage1_specs, stage1_rank)
    else:
        stage2_specs, stage2_rank = load_stage2_specs(args)
    if args.stage == "stage2":
        return 0
    args.deadline = time.time() + args.max_hours * 3600 if args.max_hours > 0 else None
    candidate, _ = stage3(args, protocol, rows, stage2_specs, stage2_rank)
    print(json.dumps(candidate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
