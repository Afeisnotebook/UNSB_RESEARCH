#!/usr/bin/env python3
"""Execute SEARCH-005's diagnosis-before-derivation funnel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parents[2]
if str(SEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.catalog import causal_catalog  # noqa: E402
from src.evidence_seed import seed_from_search003  # noqa: E402
from src.experiment import run_acmp_micro, run_bcnrp_micro, run_cndrp_micro, run_eliprc_micro, run_fbcmp_micro, run_npooa_micro, run_npooa_small_view, run_pcoa_full_view, run_pcoa_micro, run_pcoa_small_view, run_phcrp_micro, run_phrsup_micro  # noqa: E402
from src.derive import write_cards  # noqa: E402
from src.invariants import run_acmp_invariants, run_bcavp_invariants, run_bcnrp_invariants, run_cndrp_invariants, run_eliprc_invariants, run_fbcmp_invariants, run_npooa_invariants, run_pcoa_invariants, run_phcrp_invariants, run_phrsup_invariants  # noqa: E402
from src.propagation import run_component_attribution, run_propagation_audit  # noqa: E402
from src.revise import fbcmp_card, markdown as revision_markdown  # noqa: E402
from src.search001_compat import modules  # noqa: E402
from src.synthesize import build_causal_matrix, markdown  # noqa: E402
from src.variance import run_variance_audit  # noqa: E402


MANIFEST_SHA256 = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
SOURCE_ANCHOR = "3674a390e9fde997ec1261660c2a96f2a7d49aa6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def lock(args) -> tuple[list[dict], list]:
    actual = sha256(args.manifest)
    if actual != MANIFEST_SHA256:
        raise RuntimeError(f"manifest SHA256 mismatch: {actual}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_ANCHOR, head], cwd=REPO_ROOT
    ).returncode != 0:
        raise RuntimeError("SEARCH-005 source anchor is not in runtime ancestry")
    runtime = modules()[1]
    rows = runtime.read_manifest(args.manifest)
    counts = {
        domain: {
            split: sum(
                row["domain"] == domain and row["split"] == split for row in rows
            )
            for split in ("train", "discovery", "confirmation")
        }
        for domain in sorted({row["domain"] for row in rows})
    }
    if any(value != {"train": 100, "discovery": 80, "confirmation": 20}
           for value in counts.values()):
        raise RuntimeError(f"manifest split contract mismatch: {counts}")
    catalog = causal_catalog(args.runs_root)
    atomic_json(args.output / "PROTOCOL_LOCK.json", {
        "schema": "clean-unsb-search005-protocol-lock-v1",
        "source_anchor": SOURCE_ANCHOR,
        "runtime_commit": head,
        "manifest": str(args.manifest),
        "manifest_sha256": actual,
        "train_view": str(args.train_view),
        "data_root": str(args.data_root),
        "seed": int(args.seed),
        "domain_split_counts": counts,
        "route": "route1_mathematical_operator_discovery",
        "candidate_generated": False,
        "whole_state_branch_selection_as_candidate": False,
        "confirmation20_opened": False,
    })
    atomic_json(args.output / "CAUSAL_CHECKPOINT_CATALOG.json", {
        "schema": "clean-unsb-search005-causal-catalog-v1",
        "checkpoints": [cell.to_dict() for cell in catalog],
        "purpose": "diagnostic probes, not candidate lanes",
        "confirmation20_opened": False,
    })
    return rows, catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=(
            "seed", "propagation", "variance", "attribution", "synthesize", "derive",
            "invariants",
            "micro",
            "small",
            "small2",
            "full",
            "revise",
        ),
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\DATA_MANIFEST.csv"),
    )
    parser.add_argument(
        "--train-view",
        type=Path,
        default=Path(r"E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\data_views_v2\allinone_100"),
    )
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\UNSB_abl\full_dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path(r"E:\UNSB_Expl\runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\UNSB_Expl\runs\long_horizon_operator_discovery_20260827"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint-ids", nargs="*", default=[])
    parser.add_argument("--pulse-steps", type=int, default=8)
    parser.add_argument("--horizons", nargs="+", type=int, default=(8, 32, 200))
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--attribution-steps", type=int, default=32)
    parser.add_argument("--no-attribution", action="store_true")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--micro-steps", type=int, default=800)
    parser.add_argument("--micro-eval", nargs="+", type=int, default=(400, 800))
    parser.add_argument("--small-steps", type=int, default=2400)
    parser.add_argument(
        "--small-eval", nargs="+", type=int,
        default=(400, 800, 1200, 1600, 2000, 2400),
    )
    parser.add_argument("--full-steps", type=int, default=12000)
    parser.add_argument(
        "--full-eval", nargs="+", type=int,
        default=(1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000),
    )
    args = parser.parse_args()

    rows, catalog = lock(args)
    if args.stage == "invariants":
        gate_dir = SEARCH_ROOT / "artifacts" / "ENGINEERING_GATES"
        jobs = (
            ("G1-HNEK-ELIPRC", run_eliprc_invariants),
            ("G1-DT-CNDRP", run_cndrp_invariants),
            ("G1-HJ-ACMP", run_acmp_invariants),
            ("G2-HJ-FBCMP", run_fbcmp_invariants),
            ("G1-DT-HNEK-BCAVP", run_bcavp_invariants),
            ("G1-HNEK-PHCRP", run_phcrp_invariants),
            ("G2-HNEK-PHRSUP", run_phrsup_invariants),
            ("G2-DT-BCNRP", run_bcnrp_invariants),
            ("G1-GAME-PCOA", run_pcoa_invariants),
            ("G2-GAME-NPOOA", run_npooa_invariants),
        )
        for candidate_id, function in jobs:
            target = gate_dir / f"{candidate_id}.json"
            if target.is_file() and json.loads(target.read_text(encoding="utf-8")).get("passed") is True:
                print(f"invariants {candidate_id} skip passed", flush=True)
                continue
            result = function(
                rows=rows, train_view=args.train_view, output=args.output,
                seed=args.seed, gpu=args.gpu,
            )
            atomic_json(target, result)
            print(f"invariants {result['candidate_id']} passed={result['passed']}", flush=True)
        return 0
    if args.stage == "micro":
        jobs = (
            ("G1-HNEK-ELIPRC", run_eliprc_micro),
            ("G1-DT-CNDRP", run_cndrp_micro),
            ("G1-HJ-ACMP", run_acmp_micro),
            ("G2-HJ-FBCMP", run_fbcmp_micro),
            ("G1-HNEK-PHCRP", run_phcrp_micro),
            ("G2-HNEK-PHRSUP", run_phrsup_micro),
            ("G2-DT-BCNRP", run_bcnrp_micro),
            ("G1-GAME-PCOA", run_pcoa_micro),
            ("G2-GAME-NPOOA", run_npooa_micro),
        )
        for candidate_id, function in jobs:
            target = SEARCH_ROOT / "artifacts" / "MICRO_RESULTS" / f"{candidate_id}.json"
            if target.is_file():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if int(existing.get("total_steps", 0)) >= args.micro_steps:
                    print(f"micro {candidate_id} skip completed", flush=True)
                    continue
            result = function(
                rows=rows, train_view=args.train_view, data_root=args.data_root,
                output=args.output,
                gate_path=(SEARCH_ROOT / "artifacts" / "ENGINEERING_GATES" / f"{candidate_id}.json"),
                seed=args.seed, gpu=args.gpu, total_steps=args.micro_steps,
                eval_steps=tuple(args.micro_eval),
            )
            atomic_json(target, result)
            final = result["trajectory"][-1]
            print(
                f"micro {result['candidate_id']} final_delta="
                f"{final['macro_psnr_delta']:+.6f}", flush=True,
            )
        return 0
    if args.stage == "small":
        candidate_id = "G1-GAME-PCOA"
        target = SEARCH_ROOT / "artifacts" / "SMALL_VIEW_RESULTS" / f"{candidate_id}.json"
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if int(existing.get("total_steps", 0)) >= args.small_steps:
                print(f"small {candidate_id} skip completed", flush=True)
                return 0
        result = run_pcoa_small_view(
            rows=rows, train_view=args.train_view, data_root=args.data_root,
            output=args.output,
            gate_path=(SEARCH_ROOT / "artifacts" / "ENGINEERING_GATES" / f"{candidate_id}.json"),
            seed=args.seed, gpu=args.gpu, total_steps=args.small_steps,
            eval_steps=tuple(args.small_eval),
        )
        atomic_json(target, result)
        final = result["trajectory"][-1]
        print(
            f"small {result['candidate_id']} final_delta="
            f"{final['macro_psnr_delta']:+.6f}", flush=True,
        )
        return 0
    if args.stage == "small2":
        candidate_id = "G2-GAME-NPOOA"
        target = SEARCH_ROOT / "artifacts" / "SMALL_VIEW_RESULTS" / f"{candidate_id}.json"
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if int(existing.get("total_steps", 0)) >= args.small_steps:
                print(f"small2 {candidate_id} skip completed", flush=True)
                return 0
        result = run_npooa_small_view(
            rows=rows, train_view=args.train_view, data_root=args.data_root,
            output=args.output,
            gate_path=(SEARCH_ROOT / "artifacts" / "ENGINEERING_GATES" / f"{candidate_id}.json"),
            seed=args.seed, gpu=args.gpu, total_steps=args.small_steps,
            eval_steps=tuple(args.small_eval),
        )
        atomic_json(target, result)
        final = result["trajectory"][-1]
        print(
            f"small2 {result['candidate_id']} final_delta="
            f"{final['macro_psnr_delta']:+.6f}", flush=True,
        )
        return 0
    if args.stage == "full":
        candidate_id = "G1-GAME-PCOA"
        target = SEARCH_ROOT / "artifacts" / "FULL_VIEW_RESULTS" / f"{candidate_id}.json"
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if int(existing.get("total_steps", 0)) >= args.full_steps:
                print(f"full {candidate_id} skip completed", flush=True)
                return 0
        result = run_pcoa_full_view(
            rows=rows, train_view=args.train_view, data_root=args.data_root,
            output=args.output,
            gate_path=(SEARCH_ROOT / "artifacts" / "ENGINEERING_GATES" / f"{candidate_id}.json"),
            seed=args.seed, gpu=args.gpu, total_steps=args.full_steps,
            eval_steps=tuple(args.full_eval),
        )
        atomic_json(target, result)
        final = result["trajectory"][-1]
        print(
            f"full {result['candidate_id']} final_delta="
            f"{final['macro_psnr_delta']:+.6f}", flush=True,
        )
        return 0
    if args.stage == "revise":
        card = fbcmp_card(
            SEARCH_ROOT / "artifacts" / "MICRO_RESULTS" / "G1-HJ-ACMP.json"
        )
        card_dir = SEARCH_ROOT / "artifacts" / "DERIVATION_CARDS"
        atomic_json(card_dir / f"{card['candidate_id']}.json", card)
        (card_dir / f"{card['candidate_id']}.md").write_text(
            revision_markdown(card), encoding="utf-8"
        )
        atomic_json(SEARCH_ROOT / "artifacts" / "GENERATION2_DERIVATION.json", {
            "schema": "clean-unsb-search005-generation2-derivation-v1",
            "candidate_ids": [card["candidate_id"]],
            "revision_trigger": card["causal_failure_class"],
            "uses_fixed_window": False,
            "paired_target_access": False,
            "confirmation20_opened": False,
        })
        print(f"revised {card['candidate_id']}", flush=True)
        return 0
    if args.stage == "seed":
        result = seed_from_search003(
            REPO_ROOT / "research" / "searches" / "SEARCH-003-evidence-guided-discovery"
            / "artifacts" / "REVERSAL_ANALYSIS.json",
            catalog,
        )
        atomic_json(args.output / "CAUSAL_EVIDENCE_SEED.json", result)
        print(
            f"seed rows={len(result['selected_rows'])} "
            f"contradictions={len(result['contradictions'])}", flush=True
        )
        return 0

    if args.stage == "synthesize":
        seed_path = args.output / "CAUSAL_EVIDENCE_SEED.json"
        if not seed_path.is_file():
            raise FileNotFoundError(seed_path)
        matrix = build_causal_matrix(
            args.output / "propagation",
            seed_path,
            args.output / "variance",
            args.output / "attribution",
        )
        artifact_dir = SEARCH_ROOT / "artifacts"
        atomic_json(artifact_dir / "CAUSAL_MATRIX.json", matrix)
        (artifact_dir / "CAUSAL_MATRIX.md").write_text(
            markdown(matrix), encoding="utf-8"
        )
        print(
            f"matrix rows={len(matrix['propagation_rows'])} "
            f"candidate_generation_allowed={matrix['candidate_generation_allowed']}",
            flush=True,
        )
        return 0

    if args.stage == "derive":
        artifact_dir = SEARCH_ROOT / "artifacts"
        result = write_cards(
            artifact_dir / "CAUSAL_MATRIX.json",
            artifact_dir / "DERIVATION_CARDS",
        )
        atomic_json(artifact_dir / "GENERATION1_DERIVATION.json", result)
        print(f"derived {result['candidate_ids']}", flush=True)
        return 0

    if args.stage == "variance":
        selected = catalog
        if args.checkpoint_ids:
            wanted = set(args.checkpoint_ids)
            selected = [cell for cell in catalog if cell.checkpoint_id in wanted]
            missing = wanted - {cell.checkpoint_id for cell in selected}
            if missing:
                raise ValueError(f"unknown checkpoint ids: {sorted(missing)}")
        for index, cell in enumerate(selected, start=1):
            target = args.output / "variance" / f"{cell.checkpoint_id}.json"
            if target.is_file():
                print(f"VARIANCE skip existing {cell.checkpoint_id}", flush=True)
                continue
            print(f"VARIANCE {index}/{len(selected)} {cell.checkpoint_id}", flush=True)
            result = run_variance_audit(
                cell=cell,
                rows=rows,
                train_view=args.train_view,
                work_dir=args.output / "work",
                seed=args.seed,
                gpu=args.gpu,
                replicates=args.replicates,
            )
            atomic_json(target, result)
            print(
                f"VARIANCE done {cell.checkpoint_id} "
                f"fraction={result['global']['variance_fraction']:.6f}", flush=True
            )
        return 0

    if args.stage == "attribution":
        selected = catalog
        if args.checkpoint_ids:
            wanted = set(args.checkpoint_ids)
            selected = [cell for cell in catalog if cell.checkpoint_id in wanted]
            missing = wanted - {cell.checkpoint_id for cell in selected}
            if missing:
                raise ValueError(f"unknown checkpoint ids: {sorted(missing)}")
        for index, cell in enumerate(selected, start=1):
            target = args.output / "attribution" / f"{cell.checkpoint_id}.json"
            if target.is_file():
                print(f"ATTRIBUTION skip existing {cell.checkpoint_id}", flush=True)
                continue
            print(f"ATTRIBUTION {index}/{len(selected)} {cell.checkpoint_id}", flush=True)
            result = run_component_attribution(
                cell=cell,
                rows=rows,
                train_view=args.train_view,
                data_root=args.data_root,
                work_dir=args.output / "work",
                seed=args.seed,
                gpu=args.gpu,
                pulse_steps=args.pulse_steps,
                attribution_steps=args.attribution_steps,
                eval_count=args.eval_count,
            )
            atomic_json(target, result)
        return 0

    selected = catalog
    if args.checkpoint_ids:
        wanted = set(args.checkpoint_ids)
        selected = [cell for cell in catalog if cell.checkpoint_id in wanted]
        missing = wanted - {cell.checkpoint_id for cell in selected}
        if missing:
            raise ValueError(f"unknown checkpoint ids: {sorted(missing)}")
    for index, cell in enumerate(selected, start=1):
        target = args.output / "propagation" / f"{cell.checkpoint_id}.json"
        if target.is_file():
            print(f"PROPAGATION skip existing {cell.checkpoint_id}", flush=True)
            continue
        print(
            f"PROPAGATION {index}/{len(selected)} {cell.checkpoint_id} "
            f"horizons={list(args.horizons)}", flush=True
        )
        result = run_propagation_audit(
            cell=cell,
            rows=rows,
            train_view=args.train_view,
            data_root=args.data_root,
            work_dir=args.output / "work",
            seed=args.seed,
            gpu=args.gpu,
            pulse_steps=args.pulse_steps,
            horizons=tuple(args.horizons),
            eval_count=args.eval_count,
            attribution_steps=args.attribution_steps,
            run_attribution=not args.no_attribution,
        )
        atomic_json(target, result)
        final = result["trajectory"][-1]
        delta = final.get("native_view_delta", final.get("pulse_native_view_delta"))
        print(
            f"PROPAGATION done {cell.checkpoint_id} "
            f"final_delta={delta['macro_psnr_delta']:+.6f}", flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
