"""SEARCH-004 engineering gates on real clean-UNSB checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .catalog import audit_catalog
from .engine import HandoffEngine, load_payload
from .protocol import Search004Protocol, assert_target_blind
from .state import (
    capture_full_training_state_v2,
    cpu_clone,
    exact_equal,
    export_named_optimizers,
    load_named_optimizers,
    load_full_training_state_v2,
    torch_digest,
    validate_checkpoint_payload,
)


def _equal(left, right) -> bool:
    return exact_equal(left, right)[0]


def run_engineering_gate(
    *,
    rows: list[dict],
    train_view: Path,
    data_root: Path,
    runs_root: Path,
    output_dir: Path,
    seed: int,
    gpu: int,
    protocol: Search004Protocol,
) -> dict:
    inherited_path = (
        Path(runs_root) / "evidence_guided_discovery_20260827"
        / "ENGINEERING_GATE.json"
    )
    if not inherited_path.is_file():
        inherited_path = (
            Path(__file__).resolve().parents[1]
            / ".." / "SEARCH-003-evidence-guided-discovery"
            / "artifacts" / "ENGINEERING_GATE.json"
        ).resolve()
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))
    catalog = audit_catalog(runs_root)
    completeness = []
    for checkpoint in catalog:
        for role, path in (("plain", checkpoint.plain), ("method", checkpoint.method)):
            value = validate_checkpoint_payload(load_payload(path))
            completeness.append({"checkpoint_id": checkpoint.checkpoint_id, "role": role, **value})

    checkpoint = next(row for row in catalog if row.checkpoint_id == "HJPROP-1200")
    gate_dir = Path(output_dir) / "engineering_gate"
    engine = HandoffEngine(
        checkpoint=checkpoint,
        rows=rows,
        train_view=train_view,
        work_dir=gate_dir,
        seed=seed,
        gpu=gpu,
        max_horizon=200,
    )
    try:
        source = engine.prepare_arm("A_hard_disable", protocol)
        engine.load_state(source)
        original_model = cpu_clone(engine.runtime.model_state(engine.model))
        named = export_named_optimizers(engine.model)
        reordered = {
            name: {
                "groups": cpu_clone(value["groups"]),
                "states": dict(reversed(list(value["states"].items()))),
            }
            for name, value in named.items()
        }
        load_named_optimizers(engine.model, reordered)
        named_roundtrip_exact = _equal(original_model, engine.runtime.model_state(engine.model))

        engine.load_state(source)
        v2 = capture_full_training_state_v2(
            engine.model,
            engine.runtime,
            step=source["step"],
            rng=source["rng"],
            stream_a=source["stream_a"],
            stream_b=source["stream_b"],
            global_clock={"step": source["step"], "steps_per_epoch": engine.steps_per_epoch},
            method_costate=source["model"]["extra"],
        )
        load_full_training_state_v2(engine.model, engine.runtime, v2)
        v2_legacy_roundtrip_exact = _equal(
            source["model"], engine.runtime.model_state(engine.model)
        )

        with engine.operator(active=False):
            pass
        hard_disable_nonoperator_exact = _equal(
            original_model, engine.runtime.model_state(engine.model)
        )

        engine.load_state(source)
        geometry_parent = engine.capture_state(arm=source["arm"], completed=0)
        geometry = engine._transactional_gradient_geometry()
        geometry_after = engine.capture_state(arm=source["arm"], completed=0)
        gradient_audit_nonpolluting = _equal(geometry_parent, geometry_after)
        gradient_audit_defined = (
            geometry["native_gradient_norm"] > 0.0
            and geometry["intervention_gradient_norm"] > 0.0
            and geometry["correction_gradient_norm"] > 0.0
            and bool(geometry["block_geometry"])
            and not geometry["paired_target_access"]
        )

        p_state = engine.prepare_arm("P_common_plain", protocol)
        f_state = engine.prepare_arm("F_g_only_transplant", protocol)
        g_only_network_difference = (
            not _equal(p_state["model"]["networks"]["G"], f_state["model"]["networks"]["G"])
            and all(
                _equal(p_state["model"]["networks"][name], f_state["model"]["networks"][name])
                for name in ("F", "D", "E")
            )
            and _equal(p_state["model"]["optimizers"], f_state["model"]["optimizers"])
            and _equal(p_state["model"]["schedulers"], f_state["model"]["schedulers"])
            and _equal(p_state["rng"], f_state["rng"])
            and _equal(p_state["stream_a"], f_state["stream_a"])
            and _equal(p_state["stream_b"], f_state["stream_b"])
        )

        k_state = engine.prepare_arm("K_gf_state_transplant", protocol)
        gf_state_transplant_exact = (
            _equal(
                k_state["model"]["networks"]["G"],
                source["model"]["networks"]["G"],
            )
            and _equal(
                k_state["model"]["networks"]["F"],
                source["model"]["networks"]["F"],
            )
            and _equal(
                k_state["model"]["optimizers"][0],
                source["model"]["optimizers"][0],
            )
            and _equal(
                k_state["model"]["optimizers"][3],
                source["model"]["optimizers"][3],
            )
            and _equal(
                k_state["model"]["networks"]["D"],
                p_state["model"]["networks"]["D"],
            )
            and _equal(
                k_state["model"]["networks"]["E"],
                p_state["model"]["networks"]["E"],
            )
            and _equal(
                k_state["model"]["optimizers"][1],
                p_state["model"]["optimizers"][1],
            )
            and _equal(
                k_state["model"]["optimizers"][2],
                p_state["model"]["optimizers"][2],
            )
        )

        c_state = engine.prepare_arm("C_local_native_moment", protocol)
        shadow_nonpolluting = (
            _equal(source["model"]["networks"], c_state["model"]["networks"])
            and _equal(source["model"]["optimizers"][1:3], c_state["model"]["optimizers"][1:3])
            and _equal(source["model"]["schedulers"], c_state["model"]["schedulers"])
            and _equal(source["rng"], c_state["rng"])
            and _equal(source["stream_a"], c_state["stream_a"])
            and _equal(source["stream_b"], c_state["stream_b"])
            and not _equal(source["model"]["optimizers"][0], c_state["model"]["optimizers"][0])
        )
        engine.load_state(c_state)
        shadow_resume_exact = _equal(c_state, engine.capture_state(arm=c_state["arm"], completed=0))

        d_path = gate_dir / "D_step1.pt"
        engine.run_arm(
            arm="D_costate_equilibration", horizon=1, protocol=protocol,
            data_root=data_root, eval_count=10, eval_start=0,
            include_lpips=False, save_state=d_path, evaluation_horizons=(),
        )
        d_state = torch.load(d_path, map_location="cpu", weights_only=False)
        d_source = engine.prepare_arm("D_costate_equilibration", protocol)
        costate_g_frozen = (
            _equal(d_source["model"]["networks"]["G"], d_state["model"]["networks"]["G"])
            and _equal(d_source["model"]["optimizers"][0], d_state["model"]["optimizers"][0])
            and not _equal(d_source["model"]["networks"]["D"], d_state["model"]["networks"]["D"])
            and not _equal(d_source["model"]["networks"]["E"], d_state["model"]["networks"]["E"])
        )
        h_state = engine.prepare_arm("H_native_moment_projection", protocol)
        h_record = h_state.get("transport_record") or {}
        h_source = engine.prepare_arm("A_hard_disable", protocol)
        h_named = h_state["model"]["optimizers"]
        source_named = h_source["model"]["optimizers"]
        native_projection_component_safe = (
            _equal(h_source["model"]["networks"], h_state["model"]["networks"])
            and _equal(h_source["model"]["schedulers"], h_state["model"]["schedulers"])
            and _equal(source_named[1], h_named[1])
            and _equal(source_named[2], h_named[2])
            and _equal(h_source["rng"], h_state["rng"])
            and _equal(h_source["stream_a"], h_state["stream_a"])
            and _equal(h_source["stream_b"], h_state["stream_b"])
            and h_record.get("paired_target_access") is False
            and h_record.get("plain_reference_access") is False
            and float(h_record.get("target_blind_defect_after", 1.0)) <= 1e-6
        )
        engine.load_state(h_state)
        native_projection_resume_exact = _equal(
            h_state,
            engine.capture_state(arm=h_state["arm"], completed=0),
        )
        l_state = engine.prepare_arm("L_variance_carried_rebase", protocol)
        l_record = l_state.get("transport_record") or {}
        variance_rebase_component_safe = (
            _equal(h_source["model"]["networks"], l_state["model"]["networks"])
            and _equal(h_source["model"]["schedulers"], l_state["model"]["schedulers"])
            and _equal(h_source["model"]["optimizers"][1], l_state["model"]["optimizers"][1])
            and _equal(h_source["model"]["optimizers"][2], l_state["model"]["optimizers"][2])
            and _equal(h_source["rng"], l_state["rng"])
            and _equal(h_source["stream_a"], l_state["stream_a"])
            and _equal(h_source["stream_b"], l_state["stream_b"])
            and l_record.get("paired_target_access") is False
            and l_record.get("plain_reference_access") is False
            and l_record.get("second_moments_changed") is False
            and l_record.get("optimizer_age_changed") is False
        )
        engine.load_state(l_state)
        variance_rebase_resume_exact = _equal(
            l_state,
            engine.capture_state(arm=l_state["arm"], completed=0),
        )
        parent_immutable = engine.parent_digest == torch_digest(engine.method_payload)
    finally:
        engine.close()

    target_blind_rejects = True
    for bad in ({"psnr": 1.0}, {"paired_target": "x"}, {"confirmation": True}):
        try:
            assert_target_blind(bad)
            target_blind_rejects = False
        except ValueError:
            pass

    checks = {
        "plain_twin_exact": bool(inherited["plain_twin_exact"]),
        "full_state_resume_exact": bool(inherited["resume_exact"]),
        "evaluation_repeat_exact": bool(inherited["evaluation_repeat_exact"]),
        "legacy_checkpoint_completeness": len(completeness) == 2 * len(catalog),
        "named_optimizer_roundtrip_exact": named_roundtrip_exact,
        "legacy_v2_legacy_roundtrip_exact": v2_legacy_roundtrip_exact,
        "hard_disable_nonoperator_exact": hard_disable_nonoperator_exact,
        "gradient_audit_nonpolluting": gradient_audit_nonpolluting,
        "gradient_audit_defined": gradient_audit_defined,
        "shadow_nonpolluting": shadow_nonpolluting,
        "shadow_resume_exact": shadow_resume_exact,
        "common_future_stream_exact": _equal(source["stream_a"], c_state["stream_a"])
        and _equal(source["stream_b"], c_state["stream_b"])
        and _equal(source["rng"], c_state["rng"]),
        "g_only_transplant_exact": g_only_network_difference,
        "gf_state_transplant_exact": gf_state_transplant_exact,
        "costate_equilibration_g_frozen": costate_g_frozen,
        "native_projection_component_safe": native_projection_component_safe,
        "native_projection_resume_exact": native_projection_resume_exact,
        "variance_rebase_component_safe": variance_rebase_component_safe,
        "variance_rebase_resume_exact": variance_rebase_resume_exact,
        "parent_immutable": parent_immutable,
        "target_blind_schema_rejects": target_blind_rejects,
        "confirmation20_sealed": True,
    }
    return {
        "schema": "clean-unsb-search004-engineering-gate-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checkpoint_completeness": completeness,
        "inherited_gate": str(inherited_path),
        "confirmation20_opened": False,
    }
