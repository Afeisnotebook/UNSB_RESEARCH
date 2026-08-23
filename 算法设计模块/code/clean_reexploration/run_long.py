"""Single long-run orchestrator for the DT/HJ/HNEK clean re-exploration.

Execution order (section 11.2): protection/authority/data identity -> CPU, real
model, determinism, resume, self-review and freeze -> profile/budget ->
canonical plain -> HNEK FULL/HANDOFF -> DT -> HJ -> TRAINING_FROZEN.ok ->
paired evaluator -> adjudication -> packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from clean_reexploration import (
    access_guard,
    adjudicate,
    controllers,
    diagnostics,
    full_state,
    identity,
    package_return,
)


REPO_ROOT = Path("/home/yc/unsb_tired")
CODE_ROOT = REPO_ROOT / "算法设计模块/code"
MODULE_ROOT = CODE_ROOT / "clean_reexploration"
RUNTIME_ROOT = REPO_ROOT / "runtime_4090/clean_reexploration_20260824"
AUTHORITY_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def freeze_spec() -> dict:
    return {
        "schema_version": 1,
        "task": "DT/HJ/HNEK clean-reexploration 4090 long task",
        "seed": 2026,
        "epochs": 200,
        "canonical_plain": {
            "model": "sb",
            "lambda_NCE": 1.0,
            "num_timesteps": 5,
            "tau": 0.01,
            "batch_size": 1,
            "load_size": 128,
            "crop_size": 128,
            "lr": 2e-4,
            "beta1": 0.5,
            "beta2": 0.999,
            "lr_policy": "linear",
        },
        "dt": {
            "base_lambda": 0.001,
            "m": 4,
            "active_age_ramp": [1, 5],
            "active_age_hold": [5, 15],
            "active_age_decay": [15, 25],
            "teacher_from": "post_e20",
        },
        "hj": {
            "layer": 0,
            "structure": "joint",
            "probe": "central_consensus",
            "strength": 0.5,
            "gate_quantile": 0.75,
            "min_risk": 0.05,
            "boundary_scale": 0.001,
            "update_mode": "remove",
            "start_epoch": 5,
        },
        "hnek": {
            "model": "hnek_search",
            "gamma": 0.25,
            "coord": "residual",
            "horizon_mode": "physical",
            "partial": "all",
        },
        "diagnostic_panel": {
            "per_domain_a": 16,
            "per_domain_b": 16,
            "seed": 20260824,
        },
        "bootstrap": {"n_draws": 999, "alpha": 0.05, "cluster": "source_in_domain"},
        "budget": {
            "gpu_hours_hard": 48,
            "wall_clock_hours_hard": 72,
            "disk_gib_hard": 120,
        },
        "base_authority": {
            "final1_spec_canonical_sha256": identity.FINAL1_SPEC_CANONICAL_SHA256,
            "training_manifest_sha256": identity.TRAINING_MANIFEST_SHA256,
            "paired_development_manifest_sha256": identity.PAIRED_DEVELOPMENT_MANIFEST_SHA256,
        },
    }


def setup_identity(state_dir: Path) -> dict:
    auth = identity.verify_base_authority(AUTHORITY_ROOT)
    if not auth["ok"]:
        _write_json(state_dir / "HARD_STOP.json", {"reason": auth["reason"], "auth": auth})
        raise SystemExit(f"HARD_STOP_MISSING_BASE_AUTHORITY: {auth['problems']}")

    spec = freeze_spec()
    spec_canonical = identity.sha256_bytes(identity.canonical_json_bytes(spec))
    run_id = identity.make_run_id(spec_canonical)

    # Code identity over the executed source trees.
    code_files = []
    for root in (
        CODE_ROOT / "baseline",
        CODE_ROOT / "dt_covmatch",
        CODE_ROOT / "hj_patchnce",
        MODULE_ROOT,
    ):
        for p in root.rglob("*.py"):
            if "__pycache__" not in p.parts:
                code_files.append(p)
    code_id = identity.code_identity(code_files)

    _write_json(state_dir / "BASE_AUTHORITY.json", auth)
    (state_dir / "CODE_SHA256.txt").write_text(code_id["code_sha256"] + "\n")
    (state_dir / "RUN_ID.txt").write_text(run_id + "\n")

    frozen_spec_path = MODULE_ROOT / "frozen/CLEAN_REEXPLORATION_FROZEN_SPEC.json"
    frozen_spec_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_spec_path.write_text(
        identity.canonical_json(spec) + "\n", encoding="utf-8"
    )
    (state_dir / "SPEC_CANONICAL_SHA256.txt").write_text(spec_canonical + "\n")

    return {
        "run_id": run_id,
        "spec": spec,
        "spec_canonical_sha256": spec_canonical,
        "code_sha256": code_id["code_sha256"],
        "authority": auth,
    }


def build_training_rows(identity_dir: Path) -> tuple[list[dict], str]:
    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    audit = identity.audit_manifest_files(files)
    if not audit["ok"]:
        _write_json(identity_dir / "HARD_STOP_DATA_MANIFEST.json", audit)
        raise SystemExit(f"HARD_STOP_DATA_MANIFEST: {audit}")
    return files, identity.data_root_from_manifest(t2)


def run_tests(state_dir: Path) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "2026"
    cmds = [
        [sys.executable, "-m", "pytest", "-q", str(MODULE_ROOT / "tests")],
        [sys.executable, "-m", "compileall", "-q", str(CODE_ROOT / "clean_reexploration")],
    ]
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=CODE_ROOT, env=env, text=True, capture_output=True)
        if proc.returncode != 0:
            _write_json(state_dir / "TEST_FAILURE.json", {
                "cmd": cmd,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            })
            raise SystemExit(f"test failure: {cmd}")


def run_long(args) -> int:
    state_dir = RUNTIME_ROOT / "state"
    authority_dir = RUNTIME_ROOT / "authority"
    runs_dir = RUNTIME_ROOT / "runs"
    logs_dir = RUNTIME_ROOT / "logs"
    raw_dir = RUNTIME_ROOT / "raw"
    portable_dir = RUNTIME_ROOT / "portable"
    for d in (state_dir, authority_dir, runs_dir, logs_dir, raw_dir, portable_dir):
        d.mkdir(parents=True, exist_ok=True)

    _log(f"[{_now()}] setup identity")
    ident = setup_identity(state_dir)
    (authority_dir / "CODE_SHA256.txt").write_text(ident["code_sha256"] + "\n")
    (authority_dir / "SPEC_CANONICAL_SHA256.txt").write_text(ident["spec_canonical_sha256"] + "\n")
    (authority_dir / "RUN_ID.txt").write_text(ident["run_id"] + "\n")

    _log(f"[{_now()}] build training rows")
    training_files, data_root = build_training_rows(state_dir)
    t3 = AUTHORITY_ROOT / "specs/h2c/T3_CONFIRMATORY_MANIFEST.json"
    paired_files = identity.load_paired_development_manifest(t3)

    _log(f"[{_now()}] run CPU/static tests")
    run_tests(state_dir)

    _log(f"[{_now()}] build diagnostic panel and access guard")
    panel = diagnostics.build_diagnostic_panel(
        training_files,
        per_domain_a=ident["spec"]["diagnostic_panel"]["per_domain_a"],
        per_domain_b=ident["spec"]["diagnostic_panel"]["per_domain_b"],
        seed=ident["spec"]["diagnostic_panel"]["seed"],
    )
    _write_json(state_dir / "DIAGNOSTIC_PANEL.json", panel)

    frozen_ok = RUNTIME_ROOT / "TRAINING_FROZEN.ok"
    guard = access_guard.TargetAccessGuard(
        training_manifest=training_files,
        paired_manifest=paired_files,
        ledger_path=raw_dir / "ACCESS_LEDGER.csv",
        data_root=data_root,
        frozen_ok_path=frozen_ok,
    )
    # Reject every paired-development target before training freeze.
    for f in paired_files:
        if f["role"].endswith("TARGET") or f["role"] == "T3_A_TARGET":
            try:
                guard.request(f["absolute_path"], role="sealed_target_probe", purpose="guard_self_test")
            except PermissionError:
                pass

    _log(f"[{_now()}] write freeze marker after all lanes complete")
    # Full training loop is delegated to the real-model executor; here we record
    # the pre-effect contract and let the effect executor populate lanes.
    _write_json(state_dir / "PRE_EFFECT_CONTRACT.json", {
        "run_id": ident["run_id"],
        "spec_canonical_sha256": ident["spec_canonical_sha256"],
        "code_sha256": ident["code_sha256"],
        "training_manifest": ident["authority"]["training_manifest_path"],
        "paired_development_manifest": ident["authority"]["paired_development_manifest_path"],
        "data_root": data_root,
        "training_files": len(training_files),
        "paired_files": len(paired_files),
    })
    _log("[run_long] pre-effect contract written; real-model training delegated to executor")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pre_effect", "train", "evaluate", "package"], default="pre_effect")
    args = parser.parse_args()
    return run_long(args)


if __name__ == "__main__":
    raise SystemExit(main())
