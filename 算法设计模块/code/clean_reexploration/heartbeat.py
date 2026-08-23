"""Five-minute heartbeat writer for the long-running 4090 task."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path("/home/yc/unsb_tired")
RUNTIME_ROOT = REPO_ROOT / "runtime_4090/clean_reexploration_20260824"
RUNS_ROOT = RUNTIME_ROOT / "runs"
LOG = RUNTIME_ROOT / "logs" / "run_long.log"
HEARTBEAT = RUNTIME_ROOT / "logs" / "heartbeat.json"
LANES = ["canonical_plain", "hnek_full", "dt", "hj"]


def _gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10,
        ).stdout.strip().split(",")
        return {"mem_used_mib": int(out[0]), "mem_total_mib": int(out[1]), "util_pct": int(out[2])}
    except Exception:
        return {}


def _disk() -> int:
    try:
        out = subprocess.run(["du", "-s", "-m", str(RUNTIME_ROOT)], text=True, capture_output=True, timeout=30).stdout.split()[0]
        return int(out)
    except Exception:
        return -1


def _last_epoch(lane: str) -> int:
    d = RUNS_ROOT / lane
    if not d.is_dir():
        return 0
    epochs = []
    for p in d.glob("full_state_e*.pt"):
        try:
            epochs.append(int(p.stem.replace("full_state_e", "")))
        except ValueError:
            continue
    return max(epochs) if epochs else 0


def _current_lane() -> str:
    for lane in LANES:
        if (RUNS_ROOT / lane / "full_state_e200.pt").is_file():
            continue
        if (RUNS_ROOT / lane).is_dir():
            return lane
    return LANES[-1]


def heartbeat_once() -> dict:
    lane = _current_lane()
    rec = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "train",
        "lane": lane,
        "last_epoch": {l: _last_epoch(l) for l in LANES},
        "gpu": _gpu(),
        "disk_mib": _disk(),
        "training_frozen": (RUNTIME_ROOT / "TRAINING_FROZEN.ok").exists(),
    }
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        while True:
            heartbeat_once()
            time.sleep(300)
    else:
        print(json.dumps(heartbeat_once(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
