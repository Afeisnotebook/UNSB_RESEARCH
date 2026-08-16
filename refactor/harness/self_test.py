"""CPU-only self-test for the harness core (no pixels, no GPU training)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import checkpoint, config, data, determinism, metrics  # noqa: E402


DT_DATA = Path("/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired/manifest.csv")
HJ_VALO = Path(
    "/home/yc/UNSB_Patch/datasets/final6train_valO5x16_offset560_unpaired/manifest.csv"
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS " if ok else "FAIL ") + name + ((" | " + detail) if detail else ""))
    return ok


def main() -> int:
    results: list[bool] = []

    dt_report = data.audit_manifest(
        DT_DATA,
        expected_domain_split={
            d: {"train": 160, "test": 40}
            for d in [
                "FoggyCityscapes",
                "LowLightTrafficData",
                "RainCityscapes",
                "RainDS-syn",
                "RSCityscapes",
                "SnowTrafficData",
            ]
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
        HJ_VALO,
        expected_domain_split={
            d: {"train": 160, "test": 16}
            for d in [
                "FoggyCityscapes",
                "LowLightTrafficData",
                "RainCityscapes",
                "RSCityscapes",
                "SnowTrafficData",
            ]
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

    overlap = data.zero_overlap(DT_DATA, HJ_VALO, split="test", key="source_target")
    results.append(
        check(
            "dt test40 vs val-O test zero overlap",
            overlap["overlap"] == 0,
            f"overlap={overlap['overlap']}",
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
            and config.canonical_json(frozen["args"]) == config.canonical_json(frozen2["args"]),
        )
    )

    determinism.seed_everything(2026)
    before = np.random.get_state()[1].copy()
    with determinism.rng_scope():
        _ = np.random.rand(100)
    after = np.random.get_state()[1].copy()
    results.append(
        check(
            "rng_scope restores numpy state",
            np.array_equal(before, after),
        )
    )

    s1 = determinism.sub_seed("run", 2026, "epoch", 150, "pair", 0, "domain", 0, "sample", 0)
    s2 = determinism.sub_seed("run", 2027, "epoch", 50, "pair", 0, "domain", 0, "sample", 0)
    results.append(check("sub_seed no collision", s1 != s2, f"{s1} != {s2}"))

    hj_true = checkpoint.audit_checkpoint(
        Path("/home/yc/UNSB_Patch/runs/checkpoints_patchnce_layer0_handoff_medium"),
        "fin6_patchnce_l0h_true_constant_e200_b16_r128_s2026",
    )
    results.append(
        check(
            "HJ true_constant full-state complete at e200",
            hj_true["exists"]
            and 200 in hj_true["epochs_net_complete"]
            and 200 in hj_true["epochs_state_complete"],
            f"net={200 in hj_true.get('epochs_net_complete', [])}, state={200 in hj_true.get('epochs_state_complete', [])}",
        )
    )

    dt_best = checkpoint.audit_checkpoint(
        Path("/home/yc/UNSB_Cov5/runs/checkpoints_final6srv_b16"),
        "fin6srv_b16_dtcov_grouped_ramp5hold15decay25_l001_all6_plain_e200_s2026",
    )
    results.append(
        check(
            "DT best branch has net tuple at e200 (no training_state)",
            dt_best["exists"]
            and 200 in dt_best["epochs_net_complete"]
            and not dt_best["epochs_state_complete"],
            f"net={200 in dt_best.get('epochs_net_complete', [])}, state_epochs={dt_best.get('epochs_state_complete')}",
        )
    )

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

    failed = sum(1 for r in results if not r)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
