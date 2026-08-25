from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from .common import DOMAINS, dump_json, sha256_file
except ImportError:  # direct script execution
    from common import DOMAINS, dump_json, sha256_file


def run_job(job: dict, *, baseline_root: Path, checkpoints_root: Path, log_root: Path) -> dict:
    name = job["name"]
    epochs = job["epochs"]
    expected = [checkpoints_root / name / f"{epoch}_net_G.pth" for epoch in range(1, epochs + 1)]
    if all(path.exists() for path in expected):
        return {
            **job,
            "status": "cached",
            "elapsed_seconds": 0.0,
            "checkpoint_hashes": {path.name: sha256_file(path) for path in expected},
        }

    command = [
        sys.executable,
        str(baseline_root / "train.py"),
        "--dataroot", str(job["dataroot"]),
        "--name", name,
        "--checkpoints_dir", str(checkpoints_root),
        "--model", "sb",
        "--seed", "2041",
        "--gpu_ids", "0",
        "--dataset_mode", "unaligned",
        "--direction", "AtoB",
        "--batch_size", "1",
        "--num_threads", "0",
        "--load_size", "128",
        "--crop_size", "128",
        "--preprocess", "resize_and_crop",
        "--no_flip",
        "--lr", "0.0001",
        "--lambda_GAN", "1.0",
        "--lambda_SB", "1.0",
        "--lambda_NCE", "1.0",
        "--tau", "0.01",
        "--num_timesteps", "5",
        "--pool_size", "0",
        "--n_epochs", str(epochs),
        "--n_epochs_decay", "0",
        "--save_epoch_freq", "1",
        "--save_latest_freq", "999999999",
        "--max_dataset_size", str(job["dataset_size"]),
        "--display_id", "0",
        "--no_html",
        "--print_freq", "100",
    ]
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{name}.log"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=baseline_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.time() - start
    if proc.returncode != 0 or not all(path.exists() for path in expected):
        raise RuntimeError(f"training failed for {name}; rc={proc.returncode}; see {log_path}")
    return {
        **job,
        "status": "complete",
        "elapsed_seconds": elapsed,
        "command": command,
        "log_path": str(log_path),
        "checkpoint_hashes": {path.name: sha256_file(path) for path in expected},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    path_map = json.loads((run_root / "state" / "PATH_MAP.json").read_text(encoding="utf-8"))
    baseline_root = Path(path_map["baseline_root"])
    checkpoints_root = Path(path_map["checkpoints_root"])
    log_root = run_root / "logs"
    jobs = [
        {
            "name": f"single_{domain}_s2041_e5",
            "regime": "single",
            "domain": domain,
            "epochs": 5,
            "dataset_size": 100,
            "dataroot": Path(path_map["single_data_root"]) / domain,
        }
        for domain in DOMAINS
    ]
    jobs.append(
        {
            "name": "aio5_plain_s2041_e1",
            "regime": "aio",
            "domain": "ALL5",
            "epochs": 1,
            "dataset_size": 500,
            "dataroot": Path(path_map["aio_data_root"]),
        }
    )
    state = {"seed": 2041, "status": "running", "jobs": [], "started": time.time()}
    dump_json(run_root / "state" / "TRAINING_STATE.json", state)
    try:
        for job in jobs:
            result = run_job(
                job,
                baseline_root=baseline_root,
                checkpoints_root=checkpoints_root,
                log_root=log_root,
            )
            result["dataroot"] = str(result["dataroot"])
            state["jobs"].append(result)
            dump_json(run_root / "state" / "TRAINING_STATE.json", state)
        state["status"] = "complete"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        dump_json(run_root / "state" / "TRAINING_STATE.json", state)
        raise
    state["finished"] = time.time()
    state["elapsed_seconds"] = state["finished"] - state["started"]
    state["pairing_used_for_training"] = False
    state["target_content_read"] = False
    dump_json(run_root / "state" / "TRAINING_STATE.json", state)
    print(json.dumps({"status": state["status"], "elapsed_seconds": state["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
