#!/usr/bin/env python3
"""Execute SEARCH-004's staged gap-aware handoff discovery funnel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parents[2]
if str(SEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyze import (  # noqa: E402
    adjudicate_mechanisms,
    analyze_checkpoint,
    extension_pass,
    read_json,
)
from src.catalog import audit_catalog, exact_historical_endpoint  # noqa: E402
from src.engine import HandoffEngine, atomic_json  # noqa: E402
from src.gates import run_engineering_gate  # noqa: E402
from src.long_continuation import run_long_continuation  # noqa: E402
from src.protocol import ARMS, Search004Protocol  # noqa: E402
from src.search001_compat import modules  # noqa: E402
from src.state import validate_checkpoint_payload  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_protocol(args) -> tuple[Search004Protocol, list[dict], dict]:
    protocol = Search004Protocol(seed=args.seed)
    actual_hash = sha256(args.manifest)
    if actual_hash != protocol.manifest_sha256:
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_hash}")
    if not args.train_view.is_dir() or not args.data_root.is_dir():
        raise FileNotFoundError((args.train_view, args.data_root))
    rows = modules()[1].read_manifest(args.manifest)
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
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    for anchor in (protocol.repository_anchor, protocol.canonical_anchor):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", anchor, head], cwd=REPO_ROOT
        )
        if result.returncode != 0:
            raise RuntimeError(f"required source anchor is not in ancestry: {anchor}")
    lock = {
        **protocol.to_dict(),
        "runtime_git_commit": head,
        "manifest": str(args.manifest),
        "train_view": str(args.train_view),
        "data_root": str(args.data_root),
        "runs_root": str(args.runs_root),
        "domain_split_counts": counts,
        "paired_metrics_available_only_after_branch": True,
        "confirmation20_opened": False,
    }
    atomic_json(args.output / "PROTOCOL_LOCK.json", lock)
    return protocol, rows, lock


def write_inventory(args) -> dict:
    catalog = audit_catalog(args.runs_root)
    rows = []
    for checkpoint in catalog:
        entry = checkpoint.to_dict()
        entry["plain_completeness"] = validate_checkpoint_payload(
            torch.load(checkpoint.plain, map_location="cpu", weights_only=False)
        )
        entry["method_completeness"] = validate_checkpoint_payload(
            torch.load(checkpoint.method, map_location="cpu", weights_only=False)
        )
        rows.append(entry)
    value = {
        "schema": "clean-unsb-search004-checkpoint-inventory-v1",
        "checkpoints": rows,
        "all_exist": True,
        "confirmation20_opened": False,
    }
    atomic_json(args.output / "CHECKPOINT_INVENTORY.json", value)
    return value


def _load_results(checkpoint_dir: Path, horizon: int) -> dict[str, dict]:
    results = {}
    for path in checkpoint_dir.glob(f"*_h{horizon}.json"):
        value = read_json(path)
        results[value["arm"]] = value
    return results


def _enforce_disk_cap(output: Path, cap_gib: float) -> None:
    used = sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file())
    if used > float(cap_gib) * (1024 ** 3):
        raise RuntimeError(
            f"SEARCH-004 disk cap exceeded: {used / (1024 ** 3):.3f} GiB > {cap_gib} GiB"
        )


def _run_one(
    engine: HandoffEngine,
    *,
    arm: str,
    horizon: int,
    checkpoint_dir: Path,
    protocol: Search004Protocol,
    args,
    extension: bool,
    save_state: bool,
) -> dict:
    path = checkpoint_dir / f"{arm}_h{horizon}.json"
    if path.is_file():
        existing = read_json(path)
        geometry_complete = existing.get("diagnostic_schema") in {
            "future-unpaired-gradient-geometry-v1",
            "exact-search003-selector-v1",
        }
        # The 800-step result uses the same first 32 updates as the audited
        # 200-step arm, so its expensive discovery70/LPIPS endpoint remains
        # valid.  Only h200 files need regeneration when diagnostics evolve.
        if extension or geometry_complete:
            return existing
        archived = path.with_name(path.stem + ".pre_geometry.json")
        if not archived.exists():
            path.replace(archived)
    _enforce_disk_cap(args.output, protocol.disk_cap_gib)
    print(
        f"SEARCH004 checkpoint={engine.checkpoint.checkpoint_id} arm={arm} horizon={horizon}",
        flush=True,
    )
    result = engine.run_arm(
        arm=arm,
        horizon=horizon,
        protocol=protocol,
        data_root=args.data_root,
        eval_count=70 if extension else 10,
        eval_start=10 if extension else 0,
        include_lpips=extension,
        save_state=(checkpoint_dir / f"{arm}_h{horizon}.pt") if save_state else None,
        evaluation_horizons=(horizon,) if extension else (32, 200),
    )
    atomic_json(path, result)
    _enforce_disk_cap(args.output, protocol.disk_cap_gib)
    return result


def _ingest_hj_positive_control(args, checkpoint, checkpoint_dir: Path) -> list[dict]:
    """Reuse the preregistered SEARCH-002 HJ hard-handoff continuation."""
    if checkpoint.checkpoint_id != "HJ-1200":
        return []
    root = args.runs_root / "dthj_rederivation_20260826" / "hj_finite_handoff"
    imported = []
    for continuation, subdir in (
        (400, ("plain", "hj_anchor")),
        (800, ("plain", "hj_anchor")),
    ):
        absolute_step = checkpoint.step + continuation
        p_metric = read_json(root / subdir[0] / f"metrics_step_{absolute_step}.json")
        a_metric = read_json(root / subdir[1] / f"metrics_step_{absolute_step}.json")
        imported.append({
            "stage": "external_search002_positive_control",
            "checkpoint_id": checkpoint.checkpoint_id,
            "continuation": continuation,
            "P_macro_psnr": p_metric["macro_psnr"],
            "A_macro_psnr": a_metric["macro_psnr"],
            "delta_A_vs_P": a_metric["macro_psnr"] - p_metric["macro_psnr"],
            "P_macro_ssim": p_metric["macro_ssim"],
            "A_macro_ssim": a_metric["macro_ssim"],
            "P_macro_lpips": p_metric["macro_lpips"],
            "A_macro_lpips": a_metric["macro_lpips"],
            "source": str(root),
            "reexecuted": False,
            "confirmation20_opened": False,
        })
        if continuation == 800:
            for arm, metric in (
                ("P_common_plain", p_metric),
                ("A_hard_disable", a_metric),
            ):
                destination = checkpoint_dir / f"{arm}_h800.json"
                if not destination.is_file():
                    atomic_json(destination, {
                        "schema": "clean-unsb-search004-continuation-v1",
                        "diagnostic_schema": "imported-search002-positive-control-v1",
                        "checkpoint": checkpoint.to_dict(),
                        "arm": arm,
                        "horizon": 800,
                        "diagnostics": [],
                        "final_component_diagnostics": {},
                        "evaluations": [{"horizon": 800, **metric}],
                        "parent_immutable": True,
                        "continuation_semantics": (
                            "common_plain" if arm == "P_common_plain" else "hard_disable_native"
                        ),
                        "imported_from": str(root),
                        "paired_target_access_by_operator": False,
                        "wall_seconds": 0.0,
                        "confirmation20_opened": False,
                    })
    atomic_json(checkpoint_dir / "SEARCH002_POSITIVE_CONTROL.json", {
        "schema": "clean-unsb-search004-imported-positive-control-v1",
        "rows": imported,
        "confirmation20_opened": False,
    })
    return imported


def run_generation0(args, protocol: Search004Protocol, rows: list[dict]) -> dict:
    catalog = audit_catalog(args.runs_root)
    if args.checkpoint_id:
        catalog = [row for row in catalog if row.checkpoint_id == args.checkpoint_id]
        if not catalog:
            raise ValueError(f"unknown checkpoint: {args.checkpoint_id}")
    analyses = []
    all_extensions = []
    for checkpoint in catalog:
        checkpoint_dir = args.output / "generation0" / checkpoint.checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _ingest_hj_positive_control(args, checkpoint, checkpoint_dir)
        engine = HandoffEngine(
            checkpoint=checkpoint,
            rows=rows,
            train_view=args.train_view,
            work_dir=args.output / "generation0_work",
            seed=args.seed,
            gpu=args.gpu,
            max_horizon=protocol.extension_horizon,
        )
        try:
            compatibility_path = checkpoint_dir / "PARENT_COMPATIBILITY.json"
            if (
                compatibility_path.is_file()
                and read_json(compatibility_path).get("schema")
                != "clean-unsb-search004-parent-compatibility-v2"
            ):
                archived = compatibility_path.with_name(
                    "PARENT_COMPATIBILITY.pre_informative_filter.json"
                )
                if not archived.exists():
                    compatibility_path.replace(archived)
            if not compatibility_path.is_file():
                engine.prepare_arm("A_hard_disable", protocol)
                atomic_json(
                    compatibility_path,
                    engine.audit_parent_compatibility(
                        protocol.confidence_min_observations
                    ),
                )
            for arm in ARMS:
                _run_one(
                    engine, arm=arm, horizon=200, checkpoint_dir=checkpoint_dir,
                    protocol=protocol, args=args, extension=False, save_state=False,
                )
            results = _load_results(checkpoint_dir, 200)
            analysis = analyze_checkpoint(checkpoint, results, protocol)
            f_row = next(row for row in analysis["rows"] if row["arm"] == "F_g_only_transplant")
            if checkpoint.family == "hj" and not f_row["promote_800"]:
                _run_one(
                    engine, arm="G_gf_transplant", horizon=200,
                    checkpoint_dir=checkpoint_dir, protocol=protocol,
                    args=args, extension=False, save_state=False,
                )
                results = _load_results(checkpoint_dir, 200)
                analysis = analyze_checkpoint(checkpoint, results, protocol)
            atomic_json(checkpoint_dir / "ANALYSIS.json", analysis)
            analyses.append(analysis)
            extension_arms = list(dict.fromkeys(
                ["P_common_plain", "A_hard_disable", "U_uninterrupted"]
                + list(analysis["promoted_arms"])
            ))
            extension_results = {}
            for arm in extension_arms:
                destination = checkpoint_dir / f"{arm}_h800.json"
                endpoint = exact_historical_endpoint(checkpoint, arm, 800)
                if (
                    not destination.is_file()
                    and endpoint is not None
                    and endpoint.is_file()
                ):
                    atomic_json(
                        destination,
                        engine.evaluate_exact_historical_endpoint(
                            arm=arm,
                            horizon=800,
                            endpoint=endpoint,
                            data_root=args.data_root,
                        ),
                    )
                extension_results[arm] = _run_one(
                    engine, arm=arm, horizon=800, checkpoint_dir=checkpoint_dir,
                    protocol=protocol, args=args, extension=True,
                    save_state=arm in analysis["promoted_arms"],
                )
            p = extension_results["P_common_plain"]
            a = extension_results["A_hard_disable"]
            u = extension_results["U_uninterrupted"]
            checkpoint_extensions = [
                extension_pass(analysis, extension_results[arm], p, a, u)
                for arm in analysis["promoted_arms"]
            ]
            all_extensions.extend(checkpoint_extensions)
            atomic_json(checkpoint_dir / "EXTENSION_ANALYSIS.json", {
                "checkpoint_id": checkpoint.checkpoint_id,
                "rows": checkpoint_extensions,
                "confirmation20_opened": False,
            })
        finally:
            engine.close()
    # Aggregate every completed checkpoint, including prior resumable calls.
    analyses = [
        read_json(path) for path in sorted((args.output / "generation0").glob("*/ANALYSIS.json"))
    ]
    all_extensions = [
        row
        for path in sorted((args.output / "generation0").glob("*/EXTENSION_ANALYSIS.json"))
        for row in read_json(path)["rows"]
    ]
    compatibilities = [
        read_json(path)
        for path in sorted((args.output / "generation0").glob("*/PARENT_COMPATIBILITY.json"))
    ]
    adjudication = adjudicate_mechanisms(
        analyses, all_extensions, compatibilities
    )
    atomic_json(args.output / "STATE_COMPONENT_ABLATION.json", {
        "schema": "clean-unsb-search004-state-component-ablation-v1",
        "checkpoint_analyses": analyses,
        "extensions": all_extensions,
        "parent_compatibilities": compatibilities,
        "confirmation20_opened": False,
    })
    atomic_json(args.output / "GENERATION0_ADJUDICATION.json", adjudication)
    atlas = args.output / "HANDOFF_CAUSAL_ATLAS.jsonl"
    temporary = atlas.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for analysis in analyses:
            for row in analysis["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in all_extensions:
            handle.write(json.dumps({"stage": "extension800", **row}, ensure_ascii=False) + "\n")
        for row in compatibilities:
            handle.write(json.dumps({"stage": "parent_compatibility", **row}, ensure_ascii=False) + "\n")
        for path in sorted((args.output / "generation0").glob("*/SEARCH002_POSITIVE_CONTROL.json")):
            for row in read_json(path)["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(atlas)
    return adjudication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("lock", "gate", "generation0", "long", "all"),
        default="all",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\DATA_MANIFEST.csv"))
    parser.add_argument("--train-view", type=Path, default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\data_views_v2\allinone_100"))
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\UNSB_abl\full_dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path(r"E:\UNSB_Expl\runs"))
    parser.add_argument("--output", type=Path, default=Path(r"E:\UNSB_Expl\runs\gap_aware_handoff_20260827"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint-id", type=str)
    parser.add_argument("--long-horizon", type=int, default=2000)
    parser.add_argument("--long-eval-interval", type=int, default=400)
    parser.add_argument(
        "--long-arms", type=str,
        default="P_common_plain,A_hard_disable,H_native_moment_projection,K_gf_state_transplant",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol, rows, _ = lock_protocol(args)
    write_inventory(args)
    if args.stage == "lock":
        return 0
    if args.stage in {"gate", "all"}:
        gate = run_engineering_gate(
            rows=rows, train_view=args.train_view, data_root=args.data_root,
            runs_root=args.runs_root, output_dir=args.output,
            seed=args.seed, gpu=args.gpu, protocol=protocol,
        )
        atomic_json(args.output / "ENGINEERING_GATE.json", gate)
        print(json.dumps({"engineering_gate": gate["status"], "checks": gate["checks"]}, indent=2), flush=True)
        if gate["status"] != "PASS":
            raise RuntimeError("SEARCH-004 engineering gate failed")
        if args.stage == "gate":
            return 0
    if args.stage in {"generation0", "all"}:
        gate_path = args.output / "ENGINEERING_GATE.json"
        if not gate_path.is_file() or read_json(gate_path)["status"] != "PASS":
            raise RuntimeError("generation0 requires a passing engineering gate")
        adjudication = run_generation0(args, protocol, rows)
        print(json.dumps(adjudication, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "long":
        if not args.checkpoint_id:
            raise ValueError("--stage long requires --checkpoint-id")
        gate_path = args.output / "ENGINEERING_GATE.json"
        if not gate_path.is_file() or read_json(gate_path).get("status") != "PASS":
            raise RuntimeError("long continuation requires a passing current engineering gate")
        catalog = audit_catalog(args.runs_root)
        checkpoint = next(
            (row for row in catalog if row.checkpoint_id == args.checkpoint_id), None
        )
        if checkpoint is None:
            raise ValueError(f"unknown checkpoint: {args.checkpoint_id}")
        analysis = run_long_continuation(
            checkpoint=checkpoint,
            rows=rows,
            train_view=args.train_view,
            data_root=args.data_root,
            output=args.output,
            protocol=protocol,
            seed=args.seed,
            gpu=args.gpu,
            horizon=args.long_horizon,
            arms=tuple(value.strip() for value in args.long_arms.split(",") if value.strip()),
            eval_interval=args.long_eval_interval,
        )
        print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
