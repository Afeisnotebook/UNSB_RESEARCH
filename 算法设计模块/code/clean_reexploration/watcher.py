"""Background watcher: finalize automatically when training completes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path("/home/yc/unsb_tired")
CODE_ROOT = REPO_ROOT / "算法设计模块/code"
RUNTIME_ROOT = REPO_ROOT / "runtime_4090/clean_reexploration_20260824"
PY = "/home/yc/anaconda3/envs/unsb_cov/bin/python"


def _run(args: list[str], timeout: int = 86400) -> int:
    env = dict(__import__("os").environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "2026"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    proc = subprocess.run(
        [PY, *args], cwd=str(CODE_ROOT), env=env, text=True, capture_output=True, timeout=timeout
    )
    if proc.returncode != 0:
        print(json.dumps({"step": args, "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}, ensure_ascii=False))
    return proc.returncode


def main() -> int:
    frozen = RUNTIME_ROOT / "TRAINING_FROZEN.ok"
    done = RUNTIME_ROOT / "FINALIZED.ok"
    while True:
        if done.exists():
            print(json.dumps({"status": "already_finalized"}))
            return 0
        if frozen.exists():
            # 1) HNEK HANDOFF determination + fork (if triggered).
            _run(["clean_reexploration/train_executor.py", "--hnek-handoff", "--backend", "STRICT_CUDNN",
                  "--run-id", (RUNTIME_ROOT / "authority" / "RUN_ID.txt").read_text().strip()])
            # 2) Finalize (evaluate -> adjudicate -> package -> ZIP/SHA).
            rc = _run(["clean_reexploration/finalize.py", "--epochs", "auto", "--replicates", "4"])
            if rc == 0:
                done.write_text(json.dumps({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
            print(json.dumps({"status": "finalized", "rc": rc}))
            return 0
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
