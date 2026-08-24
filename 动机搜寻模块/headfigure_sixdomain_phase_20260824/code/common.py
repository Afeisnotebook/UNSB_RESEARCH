from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RainDS-syn",
    "RSCityscapes",
    "SnowTrafficData",
]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def dump_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bridge_times(num_timesteps: int = 5) -> np.ndarray:
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be >= 2")
    increments = np.array([0.0] + [1.0 / (index + 1) for index in range(num_timesteps - 1)])
    times = np.cumsum(increments)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    return np.concatenate([np.zeros(1), times])
