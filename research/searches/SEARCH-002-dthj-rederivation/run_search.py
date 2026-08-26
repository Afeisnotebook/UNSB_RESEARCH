#!/usr/bin/env python3
"""Efficient matched search for the DT/HJ-derived LTTR candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parents[2]
SEARCH001 = REPO_ROOT / "research" / "searches" / "SEARCH-001-clean-directional"
for path in (SEARCH001, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_spec = importlib.util.spec_from_file_location("search001_runner", SEARCH001 / "run_search.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load SEARCH-001 runtime")
search001 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search001)

from src.protocol import LaneSpec, classify  # noqa: E402
from src.runtime import read_manifest, write_json  # noqa: E402


def lanes() -> list[LaneSpec]:
    return [
        LaneSpec("plain"),
        LaneSpec("lttr_tangent", model="lttr", family="dthj_derived", estimated_g_flops_multiplier=3.0),
        LaneSpec("lttr_pulse", model="lttr", family="dthj_derived", estimated_g_flops_multiplier=3.0),
        LaneSpec("lttr_direction", model="lttr", family="dthj_derived", estimated_g_flops_multiplier=3.0),
    ]


def run_gate(args, protocol, rows) -> dict:
    root = args.output / "engineering_gate"
    safe = lanes()[-1]
    gate_args = argparse.Namespace(**vars(args))
    gate_args.deadline = None
    gate_args.log_every = 100000
    checkpoints = []
    for name in ("twin_a", "twin_b"):
        stage = root / name
        search001.run_lane(
            args=gate_args, protocol=protocol, rows=rows, stage_dir=stage, spec=safe,
            per_domain=2, target_steps=4, schedule_steps=4, eval_steps=[4],
            eval_start=0, eval_count=1, include_lpips=False,
        )
        checkpoints.append(search001.torch.load(
            stage / safe.name / "step_4.pt", map_location="cpu", weights_only=False
        ))
    for key in ("model", "rng", "stream_a", "stream_b"):
        search001.recursive_equal(checkpoints[0][key], checkpoints[1][key], f"lttr_twin.{key}")

    resume = root / "resume"
    search001.run_lane(
        args=gate_args, protocol=protocol, rows=rows, stage_dir=resume, spec=safe,
        per_domain=2, target_steps=4, schedule_steps=4, eval_steps=[4],
        eval_start=0, eval_count=1, include_lpips=False, stop_after=2,
    )
    search001.run_lane(
        args=gate_args, protocol=protocol, rows=rows, stage_dir=resume, spec=safe,
        per_domain=2, target_steps=4, schedule_steps=4, eval_steps=[4],
        eval_start=0, eval_count=1, include_lpips=False,
    )
    resumed = search001.torch.load(
        resume / safe.name / "step_4.pt", map_location="cpu", weights_only=False
    )
    for key in ("model", "rng", "stream_a", "stream_b"):
        search001.recursive_equal(checkpoints[0][key], resumed[key], f"lttr_resume.{key}")
    report = {
        "status": "PASS",
        "lttr_twin_exact": True,
        "lttr_resume_exact": True,
        "teacher_present_after_activation": (
            checkpoints[0]["model"]["extra"]["lttr"]["teacher"] is not None
        ),
        "confirmation20_opened": False,
    }
    write_json(root / "ENGINEERING_GATE.json", report)
    return report


def run_screen(args, protocol, rows) -> list[dict]:
    root = args.output / "screen"
    summaries = []
    selected = set(args.screen_lanes or [lane.name for lane in lanes()])
    for lane in lanes():
        if lane.name not in selected:
            continue
        summary = search001.run_lane(
            args=args, protocol=protocol, rows=rows, stage_dir=root, spec=lane,
            per_domain=args.screen_train_per_domain, target_steps=args.screen_steps,
            eval_steps=args.screen_eval, eval_start=0,
            eval_count=args.screen_eval_per_domain, include_lpips=False,
        )
        if lane.name != "plain":
            summaries.append(summary)
    ordering = search001.write_stage_summary(root, summaries)
    write_json(root / "MECHANISM_SCOPE.json", {
        "old_DT_reused_as_hyperparameter_template": False,
        "old_HJ_reused_as_hyperparameter_template": False,
        "derived_objects": [
            "per-image antithetic latent tangent chart",
            "frozen first-use endpoint response reference",
            "one-sided high-risk endpoint direction reversal barrier",
        ],
        "confirmation20_opened": False,
    })
    return ordering


def load_ranking(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["ranking"]


def run_hj_validation(args, protocol, rows) -> dict:
    """Re-evaluate the existing matched HJ/plain checkpoints on unseen discovery70."""
    source_root = args.legacy_search_output / "stage1_direction_screen"
    output_root = args.output / "hj_checkpoint_validation"
    e0 = search001.torch.load(source_root / "e0.pt", map_location="cpu", weights_only=False)
    metrics = {}
    for spec in (LaneSpec("plain"), LaneSpec("hj_anchor", model="hj", family="legacy")):
        model, stream_a, stream_b = search001.prepare_lane(
            args=args,
            rows=rows,
            stage_dir=output_root / "build",
            spec=spec,
            per_domain=25,
            target_steps=1200,
            schedule_steps=1200,
            e0=e0,
        )
        search001.load_checkpoint(
            source_root / spec.name / "step_1200.pt",
            model=model,
            spec=spec,
            stream_a=stream_a,
            stream_b=stream_b,
        )
        value = search001.evaluate(
            model,
            rows=rows,
            data_root=args.data_root,
            start_per_domain=10,
            count_per_domain=70,
            eval_seed=args.seed,
            include_lpips=True,
        )
        value["checkpoint"] = str(source_root / spec.name / "step_1200.pt")
        value["spec"] = spec.to_dict()
        write_json(output_root / f"{spec.name}_discovery70.json", value)
        metrics[spec.name] = value
        del model
        search001.torch.cuda.empty_cache()
    comparison = search001.compare(metrics["hj_anchor"], metrics["plain"], step=1200)
    report = {
        "purpose": "independent expansion of the screen discovery10 signal",
        "comparison": comparison,
        "confirmation20_opened": False,
        "protocol": protocol,
    }
    write_json(output_root / "HJ_CHECKPOINT_VALIDATION.json", report)
    return report


def run_hj_handoff(args, protocol, rows) -> dict:
    """Continue the validated HJ checkpoint with the projection window closed."""
    source_root = args.legacy_search_output / "stage1_direction_screen"
    output_root = args.output / "hj_finite_handoff"
    metrics = {}
    specs = (LaneSpec("plain"), LaneSpec("hj_anchor", model="hj", family="legacy"))
    for spec in specs:
        search001.run_lane(
            args=args,
            protocol=protocol,
            rows=rows,
            stage_dir=output_root,
            spec=spec,
            per_domain=25,
            target_steps=args.handoff_steps,
            schedule_steps=1200,
            eval_steps=[args.handoff_steps],
            eval_start=10,
            eval_count=70,
            include_lpips=True,
            initial_checkpoint=source_root / spec.name / "step_1200.pt",
        )
        metrics[spec.name] = search001.read_metrics(output_root, spec, args.handoff_steps)
    comparison = search001.compare(
        metrics["hj_anchor"], metrics["plain"], step=args.handoff_steps
    )
    report = {
        "algorithm": "finite-horizon HJ then plain handoff",
        "active_optimizer_steps": "[240,1200)",
        "handoff_step": 1200,
        "comparison": comparison,
        "source_checkpoints": {
            spec.name: str(source_root / spec.name / "step_1200.pt") for spec in specs
        },
        "confirmation20_opened": False,
        "protocol": protocol,
    }
    write_json(output_root / f"HJ_HANDOFF_STEP_{args.handoff_steps}.json", report)
    return report


def spec_from_row(row: dict) -> LaneSpec:
    value = dict(row["spec"])
    value["mechanisms"] = tuple(value["mechanisms"])
    return LaneSpec(**value)


def run_full(args, protocol, rows) -> list[dict]:
    screen_rank = load_ranking(args.output / "screen" / "RANKING.json")
    winner = spec_from_row(screen_rank[0])
    root = args.output / "full"
    summaries = []
    for lane in (LaneSpec("plain"), winner):
        summary = search001.run_lane(
            args=args, protocol=protocol, rows=rows, stage_dir=root, spec=lane,
            per_domain=100, target_steps=args.full_steps, eval_steps=args.full_eval,
            eval_start=10, eval_count=70, include_lpips=True,
        )
        if lane.name != "plain":
            summaries.append(summary)
    return search001.write_stage_summary(root, summaries)


def write_candidate(args, protocol, trajectory: list[dict], spec: LaneSpec) -> dict:
    summary = {
        "name": spec.name,
        "spec": spec.to_dict(),
        "trajectory": trajectory,
        "peak_to_final_rollback": max(row["macro_psnr_delta"] for row in trajectory)
        - trajectory[-1]["macro_psnr_delta"],
    }
    status = classify(summary, [summary])
    payload = {
        "schema": "clean-unsb-candidate-v1",
        "status": status,
        "candidate": summary,
        "configuration": spec.to_dict(),
        "code": {"git_commit": protocol["git_commit"], "runner": str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/")},
        "mathematical_lineage": {
            "DT": "replace batch/domain covariance z-score with a per-image antithetic latent-tangent chart",
            "HJ": "replace indirect PatchNCE finite-difference surgery with a one-sided endpoint direction reversal barrier",
        },
        "confirmation20_opened": False,
    }
    write_json(args.output / "CANDIDATE.json", payload)
    return payload


def run_extend(args, protocol, rows) -> dict:
    full_rank = load_ranking(args.output / "full" / "RANKING.json")
    winner = spec_from_row(full_rank[0])
    full_root = args.output / "full"
    extension_root = args.output / "extension"
    milestones = list(range(args.full_steps + args.extend_interval, args.extend_steps + 1, args.extend_interval))
    completed = []
    for milestone in milestones:
        for lane in (LaneSpec("plain"), winner):
            search001.run_lane(
                args=args, protocol=protocol, rows=rows, stage_dir=extension_root,
                spec=lane, per_domain=100, target_steps=milestone,
                schedule_steps=args.full_steps, eval_steps=[milestone],
                eval_start=10, eval_count=70, include_lpips=True,
                initial_checkpoint=full_root / lane.name / f"step_{args.full_steps}.pt",
            )
        completed.append(milestone)
    trajectory = []
    for step in args.full_eval:
        method = search001.read_metrics(full_root, winner, step)
        plain = search001.read_metrics(full_root, LaneSpec("plain"), step)
        if method and plain:
            trajectory.append(search001.compare(method, plain, step=step))
    for step in completed:
        method = search001.read_metrics(extension_root, winner, step)
        plain = search001.read_metrics(extension_root, LaneSpec("plain"), step)
        if method and plain:
            trajectory.append(search001.compare(method, plain, step=step))
    return write_candidate(args, protocol, trajectory, winner)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["gate", "screen", "hj_validate", "hj_handoff", "full", "extend", "all"],
        default="screen",
    )
    parser.add_argument("--manifest", type=Path, default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\DATA_MANIFEST.csv"))
    parser.add_argument("--train-view", type=Path, default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\data_views_v2\allinone_100"))
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\UNSB_abl\full_dataset"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs" / "dthj_rederivation_20260827")
    parser.add_argument(
        "--legacy-search-output",
        type=Path,
        default=Path(r"E:\UNSB_Expl\runs\directional_search_20260826"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--screen-steps", type=int, default=1200)
    parser.add_argument("--screen-eval", type=int, nargs="+", default=[400, 800, 1200])
    parser.add_argument("--screen-train-per-domain", type=int, default=25)
    parser.add_argument("--screen-eval-per-domain", type=int, default=10)
    parser.add_argument(
        "--screen-lanes",
        nargs="+",
        choices=[lane.name for lane in lanes()],
        help="Run only the named screening lanes; useful for stopping falsified lanes without spending their remaining budget.",
    )
    parser.add_argument("--full-steps", type=int, default=4000)
    parser.add_argument("--full-eval", type=int, nargs="+", default=[1000, 2000, 3000, 4000])
    parser.add_argument("--extend-steps", type=int, default=8000)
    parser.add_argument("--extend-interval", type=int, default=2000)
    parser.add_argument("--handoff-steps", type=int, default=1600)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    protocol = search001.verify_protocol(args, rows)
    protocol["search"] = "SEARCH-002-dthj-rederivation"
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "PROTOCOL_LOCK.json", protocol)
    if args.stage in {"gate", "all"}:
        run_gate(args, protocol, rows)
    if args.stage in {"screen", "all"}:
        run_screen(args, protocol, rows)
    if args.stage in {"hj_validate", "all"}:
        report = run_hj_validation(args, protocol, rows)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.stage in {"hj_handoff", "all"}:
        report = run_hj_handoff(args, protocol, rows)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.stage in {"full", "all"}:
        run_full(args, protocol, rows)
    if args.stage in {"extend", "all"}:
        candidate = run_extend(args, protocol, rows)
        print(json.dumps(candidate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"wall_seconds={time.time() - started:.1f}")
