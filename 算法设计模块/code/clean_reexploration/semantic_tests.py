"""Effect-blind real-model semantic tests (section 10.2)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path("/home/yc/unsb_tired")
CODE_ROOT = REPO_ROOT / "算法设计模块/code"
RUNTIME_ROOT = Path(
    os.environ.get(
        "UNSB_REPAIR_RUNTIME",
        str(REPO_ROOT / "runtime_4090/clean_reexploration_repair_20260825"),
    )
)
RUNS_ROOT = RUNTIME_ROOT / "runs"

sys.path.insert(0, str(CODE_ROOT / "baseline"))
sys.path.insert(0, str(CODE_ROOT / "dt_covmatch"))
sys.path.insert(0, str(CODE_ROOT / "hj_patchnce"))
sys.path.insert(0, str(CODE_ROOT))


def _grad_hash(model) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in model.netG.parameters():
        if p.grad is not None:
            h.update(p.grad.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _make_fixed_batch(device="cuda:0") -> dict:
    g = torch.Generator().manual_seed(7)
    A = torch.randn(1, 3, 128, 128, generator=g)
    B = torch.randn(1, 3, 128, 128, generator=g)
    return {
        "A": A.to(device),
        "B": B.to(device),
        "A_paths": ["dummy/input/0001.png"],
        "B_paths": ["dummy/input/0002.png"],
    }


def _run_once(model, batch):
    model.set_input(batch, batch)
    model.optimize_parameters()
    return {
        "loss_G": float(model.loss_G.item()),
        "grad_G": _grad_hash(model),
    }


def _hnek_off_equals_plain() -> dict:
    from clean_reexploration.train_executor import _create_model, _data_dependent_init, _seed_all, _setup_backend
    from models.hnek.hnek_search import set_hnek_search_active

    _setup_backend("STRICT_CUDNN")
    batch = _make_fixed_batch()

    _seed_all(2026)
    plain, _ = _create_model("sb", "sem_plain")
    _data_dependent_init(plain, batch)
    p_out = _run_once(plain, batch)

    _seed_all(2026)
    hnek, _ = _create_model("hnek_search", "sem_hnek")
    _data_dependent_init(hnek, batch)
    set_hnek_search_active(hnek, False)  # OFF -> plain
    h_out = _run_once(hnek, batch)

    equal = (p_out["loss_G"] == h_out["loss_G"]) and (p_out["grad_G"] == h_out["grad_G"])
    return {"hnek_off_equals_plain": equal, "plain_loss": p_out["loss_G"], "hnek_off_loss": h_out["loss_G"]}


def _dt_lambda0_equals_plain() -> dict:
    from clean_reexploration.train_executor import _create_model, _data_dependent_init, _seed_all, _setup_backend

    _setup_backend("STRICT_CUDNN")
    batch = _make_fixed_batch()

    _seed_all(2026)
    plain, _ = _create_model("sb", "sem_plain2")
    _data_dependent_init(plain, batch)
    p_out = _run_once(plain, batch)

    _seed_all(2026)
    dt, opt = _create_model("dtcov", "sem_dt")
    opt.dtcov_lambda = 0.0  # force lambda 0
    dt.dtcov.config.lambda_value = 0.0
    _data_dependent_init(dt, batch)
    d_out = _run_once(dt, batch)

    equal = (p_out["loss_G"] == d_out["loss_G"]) and (p_out["grad_G"] == d_out["grad_G"])
    return {"dt_lambda0_equals_plain": equal, "plain_loss": p_out["loss_G"], "dt_loss": d_out["loss_G"]}


def _hj_strength0_equals_raw() -> dict:
    from clean_reexploration.train_executor import _create_model, _data_dependent_init, _seed_all, _setup_backend

    _setup_backend("STRICT_CUDNN")
    batch = _make_fixed_batch()

    _seed_all(2026)
    hj, opt = _create_model("hj", "sem_hj")
    opt.hj_enable = True
    opt.hj_strength = 0.0
    hj.hj_config.strength = 0.0
    hj.set_train_epoch(6)  # force active (>= start_epoch 5)
    _data_dependent_init(hj, batch)
    hj.set_train_epoch(6)
    h_out = _run_once(hj, batch)

    # Raw control: same HJ model but with HJ disabled.
    _seed_all(2026)
    raw, opt2 = _create_model("hj", "sem_hj_raw")
    opt2.hj_enable = False
    raw.set_train_epoch(6)
    _data_dependent_init(raw, batch)
    raw.set_train_epoch(6)
    r_out = _run_once(raw, batch)

    equal = (h_out["loss_G"] == r_out["loss_G"]) and (h_out["grad_G"] == r_out["grad_G"])
    return {"hj_strength0_equals_raw": equal, "hj_loss": h_out["loss_G"], "raw_loss": r_out["loss_G"]}


def _hnek_on_off_no_param_change() -> dict:
    from clean_reexploration.train_executor import _create_model, _data_dependent_init, _seed_all, _setup_backend
    from models.hnek.hnek_search import set_hnek_search_active

    _setup_backend("STRICT_CUDNN")
    batch = _make_fixed_batch()
    _seed_all(2026)
    hnek, _ = _create_model("hnek_search", "sem_hnek_keys")
    _data_dependent_init(hnek, batch)
    gen = hnek.netG.module if hasattr(hnek.netG, "module") else hnek.netG
    before_keys = list(gen.state_dict().keys())
    before_count = sum(p.numel() for p in gen.parameters())
    set_hnek_search_active(hnek, False)
    after_keys = list(gen.state_dict().keys())
    after_count = sum(p.numel() for p in gen.parameters())
    set_hnek_search_active(hnek, True)
    return {
        "hnek_on_off_no_param_change": (before_keys == after_keys and before_count == after_count),
        "param_count": before_count,
    }


def _physical_epoch_gates() -> dict:
    from clean_reexploration.train_executor import _create_model, _seed_all, _setup_backend

    _setup_backend("STRICT_CUDNN")
    _seed_all(2026)
    dt, _ = _create_model("dtcov", "sem_dt_gate")
    _seed_all(2026)
    hj, _ = _create_model("hj", "sem_hj_gate")

    # HJ: OFF at physical e4, ON at physical e5 (start_epoch=5).
    hj.set_train_epoch(4)
    hj_off = not hj._hj_active()
    hj.set_train_epoch(5)
    hj_on = hj._hj_active()

    # DT: OFF at physical e20 (active age 0), ON at physical e21 (active age 1).
    dt.set_train_epoch(20)
    dt_lambda_off = dt._scheduled_dtcov_lambda() <= 0.0
    dt.set_train_epoch(21)
    dt_lambda_on = dt._scheduled_dtcov_lambda() > 0.0

    return {
        "hj_off_at_e4": hj_off,
        "hj_on_at_e5": hj_on,
        "dt_off_at_e20": dt_lambda_off,
        "dt_on_at_e21": dt_lambda_on,
    }


def _hj_true_forward_equals_raw() -> dict:
    """HJ true projection must keep the forward loss identical (backward-only)."""
    from clean_reexploration.train_executor import _create_model, _data_dependent_init, _seed_all, _setup_backend

    _setup_backend("STRICT_CUDNN")
    batch = _make_fixed_batch()

    _seed_all(2026)
    hj, opt = _create_model("hj", "sem_hj_fwd")
    opt.hj_enable = True
    opt.hj_strength = 0.5
    hj.hj_config.strength = 0.5
    hj.set_train_epoch(6)
    _data_dependent_init(hj, batch)
    hj.set_train_epoch(6)
    hj.set_input(batch, batch)
    hj.optimize_parameters()
    hj_loss = float(hj.loss_G.item())

    _seed_all(2026)
    raw, opt2 = _create_model("hj", "sem_hj_fwd_raw")
    opt2.hj_enable = False
    raw.set_train_epoch(6)
    _data_dependent_init(raw, batch)
    raw.set_train_epoch(6)
    raw.set_input(batch, batch)
    raw.optimize_parameters()
    raw_loss = float(raw.loss_G.item())

    return {"hj_true_forward_equals_raw": (hj_loss == raw_loss), "hj_loss": hj_loss, "raw_loss": raw_loss}


def _dt_teacher_sha_records() -> dict:
    """DT teacher injection must record the injected netG state SHA."""
    from clean_reexploration.train_executor import _create_model, _seed_all, _setup_backend
    from clean_reexploration.full_state import hash_tensors

    _setup_backend("STRICT_CUDNN")
    _seed_all(2026)
    dt, _ = _create_model("dtcov", "sem_dt_teacher")
    gen = dt.netG.module if hasattr(dt.netG, "module") else dt.netG
    sd = gen.state_dict()
    expected = hash_tensors(sd)
    dt.dtcov.inject_teacher(sd)
    recorded = getattr(dt.dtcov, "_teacher_netG_sha256", None)
    return {"dt_teacher_sha_records": (recorded == expected), "sha": recorded}


def semantic_main() -> int:
    results = {}
    results.update(_hnek_off_equals_plain())
    results.update(_dt_lambda0_equals_plain())
    results.update(_hj_strength0_equals_raw())
    results.update(_hnek_on_off_no_param_change())
    results.update(_physical_epoch_gates())
    results.update(_hj_true_forward_equals_raw())
    results.update(_dt_teacher_sha_records())
    out = RUNTIME_ROOT / "state" / "SEMANTIC_TESTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    all_pass = all(v for v in results.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(semantic_main())
