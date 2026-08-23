#!/usr/bin/env python3
"""Generate the measurement plan JSON consumed by run_measure.py."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
CKPT = PROJECT_ROOT / "checkpoints"


def ckpt(name: str, epoch: int) -> str:
    return str(CKPT / name / f"{epoch}_net_G.pth")


def main() -> int:
    common = {
        "bridge_times": [1, 2, 3],
        "m": 64,
        "ngf": 64,
        "tau": 0.01,
        "num_timesteps": 5,
        "region_patch": 32,
        "seed": 2026,
    }
    measurements = []
    for domain in DOMAINS:
        name = f"single_{domain}_s2026"
        for epoch in range(1, 21):
            measurements.append(
                {
                    "method": name,
                    "epoch": epoch,
                    "ckpt": ckpt(name, epoch),
                    "images_key": "c_subset",
                    "domain": domain,
                    "tag": f"{name}__e{epoch:02d}",
                }
            )
    for epoch in range(1, 21):
        measurements.append(
            {
                "method": "aio_plain",
                "epoch": epoch,
                "ckpt": ckpt("aio_plain_s2026", epoch),
                "images_key": "c_subset",
                "domain": None,
                "tag": f"aio_plain__e{epoch:02d}",
            }
        )
    for epoch in range(1, 6):
        measurements.append(
            {
                "method": "aio_dt",
                "epoch": epoch,
                "ckpt": ckpt("aio_dt_s2026", epoch),
                "images_key": "c_subset",
                "domain": None,
                "tag": f"aio_dt__e{epoch:02d}",
            }
        )

    panel_b_methods = {}
    for domain in DOMAINS:
        name = f"single_{domain}_s2026"
        panel_b_methods[name] = {"ckpt": ckpt(name, 20)}
    panel_b_methods["aio_plain"] = {"ckpt": ckpt("aio_plain_s2026", 20)}
    panel_b_methods["aio_dt"] = {"ckpt": ckpt("aio_dt_s2026", 5)}

    plan = {
        "common": common,
        "measurements": measurements,
        "panel_b": {"methods": panel_b_methods},
    }
    out = PROJECT_ROOT / "MEASURE_PLAN.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(measurements)} measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
