"""Portable CPU-only self-test for the harness core.

The historical version read manifests and checkpoints from ``/home/yc``.
That made a supposedly CPU-only self-test impossible to run from a fresh
clone. This version builds metadata-only fixtures in a temporary directory;
it exercises the same identity, overlap, checkpoint, RNG and bootstrap
contracts without requiring images, checkpoints or a GPU.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import checkpoint, config, data, determinism, metrics  # noqa: E402


DT_DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RainDS-syn",
    "RSCityscapes",
    "SnowTrafficData",
]
VALO_DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]


def _write_manifest(
    path: Path,
    *,
    domains: list[str],
    train_count: int,
    test_count: int,
    identity_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=data.MANIFEST_COLUMNS)
        writer.writeheader()
        for domain in domains:
            for split, count in (("train", train_count), ("test", test_count)):
                for index in range(count):
                    filename = f"{domain}__{split}_{index:04d}.png"
                    root = f"/fixture/{identity_prefix}/{domain}/{split}/{index:04d}"
                    writer.writerow(
                        {
                            "dataset": domain,
                            "split": split,
                            "filename": filename,
                            "input_view": f"{root}_input.png",
                            "target_view": f"{root}_target.png",
                            "source_input": f"{root}_source_input.png",
                            "source_target": f"{root}_source_target.png",
                        }
                    )


def _write_checkpoint_fixture(
    checkpoints_dir: Path,
    name: str,
    *,
    epoch: int,
    include_training_state: bool,
) -> None:
    root = checkpoints_dir / name
    root.mkdir(parents=True, exist_ok=True)
    for part in checkpoint.NET_PARTS:
        (root / f"{epoch}_{part}.pth").write_bytes(part.encode("ascii"))
    if include_training_state:
        (root / f"{epoch}_{checkpoint.STATE_PART}.pth").write_bytes(b"state")


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS " if ok else "FAIL ") + name + ((" | " + detail) if detail else ""))
    return ok


def main() -> int:
    results: list[bool] = []

    with TemporaryDirectory(prefix="unsb_harness_self_test_") as temp_dir:
        fixture_root = Path(temp_dir)
        dt_manifest = fixture_root / "dt" / "manifest.csv"
        valo_manifest = fixture_root / "valo" / "manifest.csv"
        checkpoints_dir = fixture_root / "checkpoints"

        _write_manifest(
            dt_manifest,
            domains=DT_DOMAINS,
            train_count=160,
            test_count=40,
            identity_prefix="dt",
        )
        _write_manifest(
            valo_manifest,
            domains=VALO_DOMAINS,
            train_count=160,
            test_count=16,
            identity_prefix="valo",
        )

        dt_report = data.audit_manifest(
            dt_manifest,
            expected_domain_split={
                domain: {"train": 160, "test": 40} for domain in DT_DOMAINS
            },
        )
        results.append(check("dt train160/test40 identity", dt_report["domain_split_ok"]))
        results.append(
            check(
                "dt row count == 1200",
                dt_report["n_rows"] == 1200,
                f"n_rows={dt_report['n_rows']}",
            )
        )
        results.append(
            check(
                "dt no duplicate source_target",
                dt_report["duplicate_source_targets"] == 0,
                f"dup={dt_report['duplicate_source_targets']}",
            )
        )

        valo_report = data.audit_manifest(
            valo_manifest,
            expected_domain_split={
                domain: {"train": 160, "test": 16} for domain in VALO_DOMAINS
            },
        )
        results.append(check("val-O identity", valo_report["domain_split_ok"]))
        results.append(
            check(
                "val-O test rows == 80",
                valo_report["split_counts"].get("test") == 80,
                f"test={valo_report['split_counts'].get('test')}",
            )
        )

        overlap = data.zero_overlap(
            dt_manifest, valo_manifest, split="test", key="source_target"
        )
        results.append(
            check(
                "dt test40 vs val-O test zero overlap",
                overlap["overlap"] == 0,
                f"overlap={overlap['overlap']}",
            )
        )

        _write_checkpoint_fixture(
            checkpoints_dir,
            "hj_true_constant",
            epoch=200,
            include_training_state=True,
        )
        _write_checkpoint_fixture(
            checkpoints_dir,
            "dt_best",
            epoch=200,
            include_training_state=False,
        )

        hj_true = checkpoint.audit_checkpoint(checkpoints_dir, "hj_true_constant")
        results.append(
            check(
                "HJ fixture full-state complete at e200",
                hj_true["exists"]
                and 200 in hj_true["epochs_net_complete"]
                and 200 in hj_true["epochs_state_complete"],
            )
        )

        dt_best = checkpoint.audit_checkpoint(checkpoints_dir, "dt_best")
        results.append(
            check(
                "DT fixture has net tuple but no training_state at e200",
                dt_best["exists"]
                and 200 in dt_best["epochs_net_complete"]
                and not dt_best["epochs_state_complete"],
            )
        )

    frozen = config.freeze_config(
        name="smoke",
        phase="cpu",
        args={"seed": 2026, "split": "test"},
        meta={"note": "self-test"},
    )
    frozen2 = config.freeze_config(
        name="smoke",
        phase="cpu",
        args={"seed": 2026, "split": "test"},
        meta={"note": "self-test"},
    )
    results.append(
        check(
            "config canonical hash stable (ignoring timestamp)",
            frozen["args"] == frozen2["args"]
            and config.canonical_json(frozen["args"])
            == config.canonical_json(frozen2["args"]),
        )
    )

    determinism.seed_everything(2026)
    before = np.random.get_state()[1].copy()
    with determinism.rng_scope():
        _ = np.random.rand(100)
    after = np.random.get_state()[1].copy()
    results.append(check("rng_scope restores numpy state", np.array_equal(before, after)))

    s1 = determinism.sub_seed(
        "run", 2026, "epoch", 150, "pair", 0, "domain", 0, "sample", 0
    )
    s2 = determinism.sub_seed(
        "run", 2027, "epoch", 50, "pair", 0, "domain", 0, "sample", 0
    )
    results.append(check("sub_seed no collision", s1 != s2, f"{s1} != {s2}"))

    a = np.linspace(20.0, 21.0, 80)
    b = np.linspace(19.0, 20.0, 80)
    boot = metrics.paired_bootstrap(a, b, n_bootstrap=2000, seed=2026)
    results.append(
        check(
            "paired bootstrap recovers +1.0 mean",
            boot["n"] == 80 and abs(boot["mean"] - 1.0) < 1e-6,
            f"mean={boot['mean']:.6f}",
        )
    )

    failed = sum(1 for result in results if not result)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
