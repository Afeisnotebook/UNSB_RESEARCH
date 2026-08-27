#!/usr/bin/env python3
"""Execute SEARCH-003's evidence-first discovery funnel.

The runner is intentionally staged.  ``atlas`` is the first scientific stage;
no trainable candidate is available until ``discover`` writes a derivation card
from target-blind reversal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parents[2]
if str(SEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.atlas import append_rows, audit_cell, normalize_atlas  # noqa: E402
from src.analyze import analyze_atlas  # noqa: E402
from src.adjudicate import adjudicate_generation0, generation0_markdown  # noqa: E402
from src.derive import derive_generation1  # noqa: E402
from src.catalog import preserved_catalog  # noqa: E402
from src.candidate_gates import run_candidate_gate  # noqa: E402
from src.protocol import Search003Protocol  # noqa: E402
from src.plain import run_plain  # noqa: E402
from src.receding import CANDIDATES, run_candidate  # noqa: E402
from src.revise import revise_generation2  # noqa: E402
from src.report import freeze_local_report  # noqa: E402
from src.search001_compat import modules  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def lock_protocol(args) -> tuple[Search003Protocol, list[dict], dict]:
    protocol = Search003Protocol(seed=args.seed)
    actual_hash = sha256(args.manifest)
    if actual_hash != protocol.manifest_sha256:
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_hash}")
    runtime = modules()[1]
    rows = runtime.read_manifest(args.manifest)
    counts = {
        domain: {
            split: sum(row["domain"] == domain and row["split"] == split for row in rows)
            for split in ("train", "discovery", "confirmation")
        }
        for domain in sorted({row["domain"] for row in rows})
    }
    if any(value != {"train": 100, "discovery": 80, "confirmation": 20}
           for value in counts.values()):
        raise RuntimeError(f"data split contract mismatch: {counts}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if head != protocol.source_commit:
        # The source anchor is the accepted parent.  Later SEARCH-003 commits
        # are allowed only when the parent remains in ancestry.
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", protocol.source_commit, head],
            cwd=REPO_ROOT,
        )
        if ancestor.returncode != 0:
            raise RuntimeError(f"canonical source {protocol.source_commit} is not an ancestor")
    lock = {
        **protocol.to_dict(),
        "runtime_git_commit": head,
        "manifest": str(args.manifest),
        "manifest_sha256": actual_hash,
        "train_view": str(args.train_view),
        "data_root": str(args.data_root),
        "runs_root": str(args.runs_root),
        "domain_split_counts": counts,
        "paired_metrics_available_only_after_branch": True,
        "confirmation20_opened": False,
    }
    write_json(args.output / "PROTOCOL_LOCK.json", lock)
    return protocol, rows, lock


def existing_atlas_keys(path: Path) -> set[tuple[str, str, int, int, str, str]]:
    if not path.is_file():
        return set()
    result = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            operator_costate = row.get("operator_costate")
            if operator_costate is None:
                operator_costate = (
                    "matched_historical_costate"
                    if row["source_state"] == row["probe"]
                    else "legacy_transplanted_method_costate"
                )
            result.add((
                row["probe"], row["stage"], int(row["step"]),
                int(row.get("horizon", 1)), row["source_state"], operator_costate,
            ))
    return result


def run_gate(args, rows, lock) -> dict:
    catalog = preserved_catalog(args.runs_root)
    inventory = [{
        "probe": row.probe,
        "stage": row.stage,
        "step": row.step,
        "per_domain": row.per_domain,
        "plain": str(row.plain),
        "method": str(row.method),
        "decisive": row.decisive,
    } for row in catalog]
    write_json(args.output / "CHECKPOINT_INVENTORY.json", {
        "schema": "clean-unsb-search003-checkpoint-inventory-v1",
        "checkpoints": inventory,
        "confirmation20_opened": False,
    })
    inherited_path = (
        args.runs_root / "directional_gate_20260826" / "engineering_gate"
        / "ENGINEERING_GATE.json"
    )
    inherited = (
        json.loads(inherited_path.read_text(encoding="utf-8"))
        if inherited_path.is_file() else None
    )
    inherited_pass = bool(
        inherited
        and inherited.get("plain_twin_exact") is True
        and inherited.get("resume_exact") is True
        and inherited.get("evaluation_repeat_exact") is True
        and inherited.get("manifest_sha256") == lock["manifest_sha256"]
        and inherited.get("confirmation20_opened") is False
    )
    candidate_gate_paths = sorted(
        (args.output / "candidate_gates").glob("*/CANDIDATE_GATE.json")
    )
    candidate_gates = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in candidate_gate_paths
    ]
    expected_candidate_ids = set(CANDIDATES)
    passed_candidate_ids = {
        gate["candidate"]["candidate_id"]
        for gate in candidate_gates if gate.get("status") == "PASS"
    }
    candidates_pass = expected_candidate_ids <= passed_candidate_ids
    report = {
        # Do not call the entire SEARCH-003 gate PASS before a discovered
        # controller exists: zero-intervention and controller-state resume can
        # only be checked against that concrete implementation.
        "status": (
            "PASS" if inherited_pass and candidates_pass else
            "PASS_BASELINE_INHERITED_CANDIDATE_GATES_PENDING"
            if inherited_pass else "BLOCKED_MISSING_BASELINE_GATE"
        ),
        "source_anchor_in_ancestry": True,
        "manifest_locked": True,
        "checkpoint_cells": len(catalog),
        "checkpoint_files": len(catalog) * 2,
        "state_observation_paired_access": False,
        "plain_twin_exact": inherited.get("plain_twin_exact") if inherited else None,
        "resume_exact": inherited.get("resume_exact") if inherited else None,
        "evaluation_repeat_exact": (
            inherited.get("evaluation_repeat_exact") if inherited else None
        ),
        "inherited_gate": str(inherited_path),
        "candidate_zero_intervention_exact": (
            all(gate["zero_intervention_plain_exact"] for gate in candidate_gates)
            if candidates_pass else "pending_generation1_implementation"
        ),
        "candidate_controller_resume_exact": (
            all(gate["controller_resume_exact"] for gate in candidate_gates)
            if candidates_pass else "pending_generation1_implementation"
        ),
        "candidate_virtual_branch_parent_immutable": (
            all(gate["source_immutable_after_virtual_branch"] for gate in candidate_gates)
            if candidates_pass else "pending_generation1_implementation"
        ),
        "candidate_gates": [str(path) for path in candidate_gate_paths],
        "confirmation20_opened": False,
        "protocol_lock": lock,
    }
    write_json(args.output / "ENGINEERING_GATE.json", report)
    return report


def select_cells(args):
    cells = preserved_catalog(args.runs_root)
    if args.probes:
        wanted = set(args.probes)
        cells = [cell for cell in cells if cell.probe in wanted]
    if args.checkpoint_stages:
        wanted_stages = set(args.checkpoint_stages)
        cells = [cell for cell in cells if cell.stage in wanted_stages]
    if args.steps:
        wanted_steps = set(args.steps)
        cells = [cell for cell in cells if cell.step in wanted_steps]
    cells.sort(key=lambda cell: (not cell.decisive, cell.probe, cell.stage, cell.step))
    if args.max_cells > 0:
        cells = cells[:args.max_cells]
    return cells


def run_atlas(args, rows) -> dict:
    path = args.output / "REVERSAL_ATLAS.jsonl"
    existing = existing_atlas_keys(path)
    selected = select_cells(args)
    completed = 0
    skipped = 0
    jobs = [(cell, horizon) for cell in selected for horizon in args.horizons]
    for index, (cell, horizon) in enumerate(jobs, start=1):
        requested_states = tuple(
            cell.probe if state == "method" else state
            for state in (args.source_states or ("plain", "method"))
        )
        expected = set()
        for source_state in requested_states:
            expected.add((
                cell.probe, cell.stage, cell.step, horizon, source_state,
                "reinitialized_from_source_state"
                if source_state == "plain" else "matched_historical_costate",
            ))
        if expected <= existing:
            skipped += 1
            continue
        print(
            f"ATLAS {index}/{len(jobs)} probe={cell.probe} "
            f"stage={cell.stage} step={cell.step} horizon={horizon}",
            flush=True,
        )
        result = audit_cell(
            cell,
            rows=rows,
            train_view=args.train_view,
            work_dir=args.output / "atlas_work",
            seed=args.seed,
            gpu=args.gpu,
            horizon=horizon,
            data_root=args.data_root,
            evaluate_after=args.evaluate_branches,
            source_states=requested_states,
        )
        fresh = [
            row for row in result
            if (
                row["probe"], row["stage"], int(row["step"]),
                int(row.get("horizon", 1)), row["source_state"],
                row["operator_costate"],
            ) not in existing
        ]
        append_rows(path, fresh)
        existing.update(
            (
                row["probe"], row["stage"], int(row["step"]),
                int(row.get("horizon", 1)), row["source_state"],
                row["operator_costate"],
            )
            for row in fresh
        )
        completed += 1
    report = {
        "schema": "clean-unsb-search003-atlas-run-v1",
        "selected_cells": len(selected),
        "selected_jobs": len(jobs),
        "horizons": list(args.horizons),
        "post_branch_development_evaluation": bool(args.evaluate_branches),
        "completed_this_call": completed,
        "skipped_existing": skipped,
        "atlas_rows": len(existing),
        "paired_metrics_accessed_by_controller": False,
        "paired_development_evaluated_after_branch": bool(args.evaluate_branches),
        "confirmation20_opened": False,
    }
    write_json(args.output / "ATLAS_RUN_STATE.json", report)
    return report


def run_analysis(args, protocol) -> dict:
    atlas_path = args.output / "REVERSAL_ATLAS.jsonl"
    if not atlas_path.is_file():
        raise FileNotFoundError("run --stage atlas before analysis")
    result = analyze_atlas(
        atlas_path,
        preserved_catalog(args.runs_root),
        protocol,
    )
    write_json(args.output / "REVERSAL_ANALYSIS.json", result)
    summary = {
        "atlas_rows": len(result["rows"]),
        "probes": result["probe_summaries"],
        "shared_signal_gate_passed": result["shared_signal_gate_passed"],
        "eligible_shared_signals": result["eligible_shared_signals"],
        "eligible_method_signals": result["eligible_method_signals"],
        "eligible_method_signals_by_horizon": result[
            "eligible_method_signals_by_horizon"
        ],
        "confirmation20_opened": False,
    }
    write_json(args.output / "REVERSAL_ANALYSIS_SUMMARY.json", summary)
    return summary


def run_adjudication(args) -> dict:
    atlas_path = args.output / "REVERSAL_ATLAS.jsonl"
    analysis_path = args.output / "REVERSAL_ANALYSIS.json"
    if not analysis_path.is_file():
        raise FileNotFoundError("run --stage analyze before adjudication")
    result = adjudicate_generation0(atlas_path, analysis_path)
    write_json(args.output / "GENERATION0_ADJUDICATION.json", result)
    (args.output / "GENERATION0_ADJUDICATION.md").write_text(
        generation0_markdown(result), encoding="utf-8"
    )
    return {
        "complete": result["complete"],
        "missing_decisive_cells": result["missing_decisive_cells"],
        "probe_verdicts": {
            probe: value["verdict"] for probe, value in result["probes"].items()
        },
        "shared_signal_gate_passed": result["target_blind_signal_gate"][
            "shared_signal_gate_passed"
        ],
        "confirmation20_opened": False,
    }


def run_normalize(args) -> dict:
    path = args.output / "REVERSAL_ATLAS.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = normalize_atlas(path)
    result.update({
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
    })
    write_json(args.output / "ATLAS_NORMALIZATION.json", result)
    return result


def run_derive(args) -> dict:
    result = derive_generation1(
        args.output / "REVERSAL_ANALYSIS.json",
        args.output / "GENERATION0_ADJUDICATION.json",
        args.output / "REVERSAL_ATLAS.jsonl",
        args.output,
    )
    return {
        "candidate_count": result["candidate_count"],
        "candidate_ids": [card["id"] for card in result["cards"]],
        "route_closures": result["route_closures"],
        "confirmation20_opened": False,
    }


def run_generation1_candidate(args, rows) -> dict:
    if not args.candidate:
        raise ValueError("--candidate is required for --stage candidate")
    spec = CANDIDATES[args.candidate]
    eval_steps = tuple(
        step for step in args.candidate_eval_steps
        if 0 < step <= args.candidate_steps
    )
    stage_dir = (
        args.output
        / f"generation1_small{args.candidate_train_per_domain}_seed{args.seed}"
    )
    return run_candidate(
        spec=spec,
        mode=args.candidate_mode,
        output_dir=stage_dir,
        rows=rows,
        train_view=args.train_view,
        data_root=args.data_root,
        per_domain=args.candidate_train_per_domain,
        target_steps=args.candidate_steps,
        eval_steps=eval_steps,
        eval_start=0,
        eval_count=args.candidate_eval_per_domain,
        seed=args.seed,
        gpu=args.gpu,
        include_lpips=args.candidate_include_lpips,
    )


def run_generation1_candidate_gate(args, rows) -> dict:
    if not args.candidate:
        raise ValueError("--candidate is required for --stage candidate_gate")
    return run_candidate_gate(
        spec=CANDIDATES[args.candidate],
        output_dir=args.output,
        rows=rows,
        train_view=args.train_view,
        per_domain=args.candidate_train_per_domain,
        seed=args.seed,
        gpu=args.gpu,
    )


def run_generation2_revision(args) -> dict:
    small_dir = (
        args.output
        / f"generation1_small{args.candidate_train_per_domain}_seed{args.seed}"
    )
    result = revise_generation2(args.output, small_dir)
    return {
        "revision_count": result["revision_count"],
        "candidate_ids": [
            item["revision"]["id"] for item in result["revisions"]
        ],
        "confirmation20_opened": False,
    }


def run_matched_plain(args, rows) -> dict:
    stage_dir = (
        args.output
        / f"generation1_small{args.candidate_train_per_domain}_seed{args.seed}"
    )


def run_final_report(args) -> dict:
    small_dir = (
        args.output
        / f"generation1_small{args.candidate_train_per_domain}_seed{args.seed}"
    )
    result = freeze_local_report(args.output, small_dir, REPO_ROOT)
    return {
        "candidate_id": result["winner"]["candidate_id"],
        "variant": result["winner"]["variant"],
        "classification": result["winner"]["classification"],
        "promotion_passed": result["winner"]["promotion_passed"],
        "route1_sustained_candidate_found": result["stop"][
            "route1_sustained_candidate_found"
        ],
        "confirmation20_opened": False,
    }
    zero_dir = stage_dir / "G1-DT-RHGC8__observable_only"
    return run_plain(
        output_dir=stage_dir,
        rows=rows,
        train_view=args.train_view,
        data_root=args.data_root,
        per_domain=args.candidate_train_per_domain,
        target_steps=args.candidate_steps,
        eval_steps=tuple(
            step for step in args.candidate_eval_steps
            if 0 < step <= args.candidate_steps
        ),
        eval_start=0,
        eval_count=args.candidate_eval_per_domain,
        seed=args.seed,
        gpu=args.gpu,
        bootstrap_state=zero_dir / "latest.pt",
        bootstrap_metrics_dir=zero_dir,
        include_lpips=args.candidate_include_lpips,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=(
            "gate", "atlas", "normalize", "analyze", "adjudicate", "derive",
            "candidate_gate", "candidate", "revise", "plain", "report", "all",
        ), default="all"
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\DATA_MANIFEST.csv"),
    )
    parser.add_argument(
        "--train-view", type=Path,
        default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\data_views_v2\allinone_100"),
    )
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\UNSB_abl\full_dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path(r"E:\UNSB_Expl\runs"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(r"E:\UNSB_Expl\runs\evidence_guided_discovery_20260827"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--probes", nargs="*")
    parser.add_argument("--checkpoint-stages", nargs="*")
    parser.add_argument("--steps", type=int, nargs="*")
    parser.add_argument("--source-states", nargs="*", choices=("plain", "method"))
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1])
    parser.add_argument("--evaluate-branches", action="store_true")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument(
        "--candidate-mode",
        choices=("full", "proposal_only", "observable_only"),
        default="full",
    )
    parser.add_argument("--candidate-steps", type=int, default=800)
    parser.add_argument("--candidate-train-per-domain", type=int, default=25)
    parser.add_argument("--candidate-eval-per-domain", type=int, default=10)
    parser.add_argument(
        "--candidate-eval-steps", type=int, nargs="+",
        default=[400, 800, 1200, 1600, 2000, 2400],
    )
    parser.add_argument("--candidate-include-lpips", action="store_true")
    args = parser.parse_args()
    for name in ("manifest", "train_view", "data_root", "runs_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol, rows, lock = lock_protocol(args)
    if args.stage in {"gate", "all"}:
        report = run_gate(args, rows, lock)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"atlas", "all"}:
        report = run_atlas(args, rows)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"normalize", "all"}:
        report = run_normalize(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"analyze", "all"}:
        report = run_analysis(args, protocol)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"adjudicate", "all"}:
        report = run_adjudication(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"derive", "all"}:
        report = run_derive(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "candidate":
        report = run_generation1_candidate(args, rows)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "candidate_gate":
        report = run_generation1_candidate_gate(args, rows)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "revise":
        report = run_generation2_revision(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "plain":
        report = run_matched_plain(args, rows)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "report":
        report = run_final_report(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
