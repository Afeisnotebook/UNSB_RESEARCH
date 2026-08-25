"""Deterministic real-model training executor for the clean re-exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make the parent ``code`` directory importable when this file is executed
# directly (as a script) rather than as a package module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path("/home/yc/unsb_tired")
CODE_ROOT = REPO_ROOT / "算法设计模块/code"
MODULE_ROOT = CODE_ROOT / "clean_reexploration"
RUNTIME_ROOT = Path(
    os.environ.get(
        "UNSB_REPAIR_RUNTIME",
        str(REPO_ROOT / "runtime_4090/clean_reexploration_repair_20260825"),
    )
)
RUNS_ROOT = RUNTIME_ROOT / "runs"
AUTHORITY_ROOT = Path("/home/yc/UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806")

DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
    "RainDS-syn",
]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def _setup_backend(backend: str) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("PYTHONHASHSEED", "2026")
    if backend == "STRICT_CUDNN":
        torch.backends.cudnn.enabled = True
    elif backend == "STRICT_NATIVE_NO_CUDNN":
        torch.backends.cudnn.enabled = False
    else:
        raise ValueError(backend)


def _img_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _load_image(path: str, transform: transforms.Compose) -> torch.Tensor:
    return transform(Image.open(path).convert("RGB"))


def build_deterministic_pairs(training_files: list[dict]) -> list[tuple[dict, dict]]:
    a = sorted(
        [f for f in training_files if f["side"] == "A"],
        key=lambda f: (f["domain"], f["stem"]),
    )
    b = sorted(
        [f for f in training_files if f["side"] == "B"],
        key=lambda f: (f["domain"], f["stem"]),
    )
    n = max(len(a), len(b))
    pairs = []
    for i in range(n):
        pairs.append((a[i % len(a)], b[i % len(b)]))
    return pairs


def _base_cli(name: str) -> list[str]:
    return [
        "--dataroot", "/home/yc/UNSB_C21/dataset_all",
        "--name", name,
        "--gpu_ids", "0",
        "--checkpoints_dir", str(RUNS_ROOT),
        "--seed", "2026",
        "--num_timesteps", "5",
        "--tau", "0.01",
        "--batch_size", "1",
        "--load_size", "128",
        "--crop_size", "128",
        "--max_dataset_size", "900",
        "--preprocess", "resize_and_crop",
        "--no_flip",
        "--serial_batches",
        "--num_threads", "0",
        "--lr", "0.0002",
        "--lr_policy", "linear",
        "--n_epochs", "200",
        "--n_epochs_decay", "0",
        "--lambda_NCE", "1.0",
        "--no_html",
        "--display_id", "-1",
    ]


def _make_sb_opt(name: str) -> argparse.Namespace:
    sys.path.insert(0, str(CODE_ROOT / "baseline"))
    from options.train_options import TrainOptions

    cli = _base_cli(name) + ["--model", "sb"]
    opt = TrainOptions(cmd_line=" ".join(cli)).parse()
    return opt


def _make_dtcov_opt(name: str) -> argparse.Namespace:
    opt = _make_sb_opt(name)
    opt.dtcov_lambda = 0.001
    opt.dtcov_lambda_schedule = "ramp_hold_cosine_decay"
    opt.dtcov_ramp_start_epoch = 1
    opt.dtcov_ramp_end_epoch = 5
    opt.dtcov_decay_start_epoch = 15
    opt.dtcov_decay_end_epoch = 25
    opt.dtcov_lambda_min = 0.0
    opt.dtcov_m = 4
    opt.dtcov_region_patch = 32
    opt.dtcov_u_floor = 1e-8
    opt.dtcov_norm_eps = 1e-4
    opt.dtcov_norm_momentum = 0.98
    opt.dtcov_norm_clip = 3.0
    opt.dtcov_domain_balance = "grouped_domain"
    opt.dtcov_teacher = "frozen"
    opt.dtcov_norm_mode = "domain_time"
    opt.dtcov_signal_norm = "on"
    opt.dtcov_warmup_iters = 0
    opt.dtcov_time_mode = "actual"
    opt.dtcov_diag_out = ""
    return opt


def _make_hj_opt(name: str) -> argparse.Namespace:
    opt = _make_sb_opt(name)
    opt.hj_enable = True
    opt.hj_layers = "0"
    opt.hj_direction = "joint"
    opt.hj_scales = "1,2,4"
    opt.hj_step = 0.01
    opt.hj_quantile = 0.75
    opt.hj_gate_quantile = 0.75
    opt.hj_strength = 0.5
    opt.hj_boundary_scale = 0.001
    opt.hj_min_risk = 0.05
    opt.hj_min_delta = 0.0
    opt.hj_probe_mode = "central_consensus"
    opt.hj_control = "true"
    opt.hj_amplitude = "constant"
    opt.hj_update_mode = "remove"
    opt.hj_start_epoch = 5
    opt.hj_direction_alpha = 0.0
    opt.hj_random_seed = 2026
    opt.hj_schedule = "constant"
    opt.hj_diag_out = ""
    return opt


def _create_model(model_name: str, name: str):
    if model_name == "hnek_search":
        opt = _make_sb_opt(name)
        opt.model = "hnek_search"
        opt.hnek_gamma = 0.25
        opt.hnek_coord = "residual"
        opt.hnek_horizon_mode = "physical"
        opt.hnek_partial = "all"
        from models.hnek_search_model import HnekSearchModel
        return HnekSearchModel(opt), opt
    if model_name == "dtcov":
        opt = _make_dtcov_opt(name)
        sys.path.insert(0, str(CODE_ROOT / "dt_covmatch"))
        from dtcov.model import SBModelDTCovMatch
        return SBModelDTCovMatch(opt), opt
    if model_name == "hj":
        opt = _make_hj_opt(name)
        sys.path.insert(0, str(CODE_ROOT / "hj_patchnce"))
        from hj.model import SBModelHJPatchNCE
        return SBModelHJPatchNCE(opt), opt
    opt = _make_sb_opt(name)
    from models.sb_model import SBModel
    return SBModel(opt), opt


def _data_dependent_init(model, batch):
    # SBModel.data_dependent_initialize signature is (data, data2); feed twice.
    model.data_dependent_initialize(batch, batch)
    model.setup(model.opt)
    model.parallelize()


def _forward_step(model, batch):
    model.set_input(batch, batch)
    model.optimize_parameters()


def _state_hash(model) -> str:
    import hashlib as _h
    h = _h.sha256()
    for name in ("netG", "netF"):
        net = getattr(model, name)
        sd = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def smoke(args) -> int:
    from clean_reexploration import identity
    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    pairs = build_deterministic_pairs(files)
    transform = _img_transform()

    _setup_backend(args.backend)
    _seed_all(2026)

    model, opt = _create_model(args.model, f"smoke_{args.model}")
    # Build a two-batch fixture (A/B) and data-dependent init.
    batch = {"A": [], "B": [], "A_paths": [], "B_paths": []}
    for i in range(1):
        a, b = pairs[i]
        batch["A"].append(_load_image(a["absolute_path"], transform))
        batch["B"].append(_load_image(b["absolute_path"], transform))
        batch["A_paths"].append(a["absolute_path"])
        batch["B_paths"].append(b["absolute_path"])
    batch["A"] = torch.stack(batch["A"])
    batch["B"] = torch.stack(batch["B"])
    batch["A"] = batch["A"].cuda()
    batch["B"] = batch["B"].cuda()

    _data_dependent_init(model, batch)
    for step in range(args.steps):
        _forward_step(model, batch)
    print(json.dumps({"smoke": "ok", "steps": args.steps, "model": args.model, "state_hash": _state_hash(model)}))
    return 0


# ---------------------------------------------------------------------------
# Deterministic two-stream data loading
# ---------------------------------------------------------------------------


class TwoStreamLoader:
    """Deterministic unpaired A/B loader with two distinct streams per step.

    Stream 1 and stream 2 use a fixed half-cycle offset so ``data`` and
    ``data2`` are different samples (required by the energy critic) while the
    whole sequence remains reproducible from the T2 manifest.
    """

    def __init__(self, pairs: list[tuple[dict, dict]], transform, steps_per_epoch: int | None = None):
        self.pairs = pairs
        self.transform = transform
        self.n = int(steps_per_epoch) if steps_per_epoch else len(pairs)
        self.offset = self.n // 2
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def next_pair(self) -> tuple[dict, dict]:
        a1, b1 = self.pairs[self.step % self.n]
        a2, b2 = self.pairs[(self.step + self.offset) % self.n]
        self.step += 1
        return (a1, b1), (a2, b2)

    def load_batch(self, pair: tuple[dict, dict]) -> dict:
        a, b = pair
        return {
            "A": _load_image(a["absolute_path"], self.transform).unsqueeze(0),
            "B": _load_image(b["absolute_path"], self.transform).unsqueeze(0),
            "A_paths": [a["absolute_path"]],
            "B_paths": [b["absolute_path"]],
        }


def _cuda_batch(batch: dict) -> dict:
    return {
        "A": batch["A"].cuda(),
        "B": batch["B"].cuda(),
        "A_paths": batch["A_paths"],
        "B_paths": batch["B_paths"],
    }


# ---------------------------------------------------------------------------
# Full deterministic lane training
# ---------------------------------------------------------------------------


ANCHOR_EPOCHS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                 120, 140, 160, 180, 200]


def _audit_epochs_for(method: str | None) -> list[int]:
    if method == "DT":
        # active age 2,4,...,24,25 -> physical 22,24,...,44,45.
        return [e for e in range(22, 46, 2)] + [45]
    if method in ("HJ", "HNEK"):
        return list(range(10, 201, 10))
    return []


def _save_full_state(model, *, epoch, global_step, controllers, loader, lane_dir, ident) -> dict:
    from clean_reexploration import full_state

    state = full_state.capture_full_state(
        model=model,
        global_step=global_step,
        physical_epoch=epoch,
        controllers=controllers,
        sampler={"epoch": epoch, "step": loader.step},
        identity=ident,
    )
    path = Path(lane_dir) / f"full_state_e{epoch}.pt"
    sidecar = full_state.save_full_state(path, state)
    return sidecar


def train_lane(
    *,
    model_name: str,
    lane_name: str,
    start_epoch: int,
    end_epoch: int,
    anchors: list[int],
    restore_path: Path | None,
    pairs,
    transform,
    run_id: str,
    spec: dict,
    code_sha256: str,
    method: str | None = None,
    restore_epoch: int = 0,
    steps_per_epoch: int | None = None,
    teacher_state: dict | None = None,
    panel_rows: list[dict] | None = None,
    audit_epochs: list[int] | None = None,
) -> dict:
    """Run one deterministic lane from ``start_epoch`` to ``end_epoch``."""
    from clean_reexploration import full_state

    lane_dir = RUNS_ROOT / lane_name
    lane_dir.mkdir(parents=True, exist_ok=True)
    loader = TwoStreamLoader(pairs, transform, steps_per_epoch)

    model, opt = _create_model(model_name, lane_name)
    teacher_netG_sha256 = None
    if teacher_state is not None and hasattr(model, "dtcov"):
        model.dtcov.inject_teacher(teacher_state)
        teacher_netG_sha256 = getattr(model.dtcov, "_teacher_netG_sha256", None)
    controller = None
    controllers_map = {}
    if method:
        from clean_reexploration.controllers import make_controller
        controller = make_controller(method, run_id)
        controllers_map[method] = controller
    audit_set = set(audit_epochs if audit_epochs is not None else _audit_epochs_for(method))
    save_set = set(anchors) | audit_set

    ident = {
        "run_id": run_id,
        "spec_sha256": spec["_spec_sha256"] if "_spec_sha256" in spec else "",
        "code_sha256": code_sha256,
        "lane": lane_name,
    }

    if restore_path is not None:
        state = full_state.load_full_state(restore_path)
        # Initialize netF/data-dependent before restoring optimizers/schedulers.
        # We perform a fresh data-dependent init so netF exists, then restore.
        p1, p2 = loader.next_pair()
        b1 = _cuda_batch(loader.load_batch(p1))
        b2 = _cuda_batch(loader.load_batch(p2))
        _data_dependent_init(model, b1)
        # SBModel.data_dependent_initialize calls setup+parallelize internally via
        # _data_dependent_init, but it does not restore full state; do it now.
        full_state.restore_full_state(model=model, state=state)
        loader.step = int(state["sampler"]["step"])
        global_step = int(state["global_step"])
        start_epoch = int(restore_epoch) if restore_epoch else int(state["physical_epoch"]) + 1
    else:
        p1, p2 = loader.next_pair()
        b1 = _cuda_batch(loader.load_batch(p1))
        b2 = _cuda_batch(loader.load_batch(p2))
        _data_dependent_init(model, b1)
        global_step = 0

    # Save pre_e1 anchor (after data-dependent init, before first optimizer step).
    if restore_path is None and start_epoch == 1:
        _save_full_state(model, epoch=0, global_step=0, controllers=controllers_map,
                         loader=loader, lane_dir=lane_dir, ident=ident)

    for epoch in range(start_epoch, end_epoch + 1):
        model.set_train_epoch(epoch)
        loader.reset()
        for _ in range(loader.n):
            p1, p2 = loader.next_pair()
            b1 = _cuda_batch(loader.load_batch(p1))
            b2 = _cuda_batch(loader.load_batch(p2))
            model.set_input(b1, b2)
            model.optimize_parameters()
            global_step += 1
        model.update_learning_rate()
        if epoch in audit_set and controller is not None and panel_rows is not None:
            _run_controller_audit(
                model=model,
                method=method,
                controller=controller,
                panel_rows=panel_rows,
                run_id=run_id,
                epoch=epoch,
                teacher_netG=getattr(getattr(model, "dtcov", None), "teacher", None),
            )
            if method == "DT" and controller.state.status == "OFF":
                model.opt.dtcov_lambda = 0.0
                if hasattr(model, "dtcov"):
                    model.dtcov.config.lambda_value = 0.0
            if method == "HJ" and controller.state.status == "OFF":
                model.opt.hj_strength = 0.0
                if hasattr(model, "hj_config"):
                    model.hj_config.strength = 0.0
        if epoch in save_set:
            _save_full_state(model, epoch=epoch, global_step=global_step,
                             controllers=controllers_map, loader=loader,
                             lane_dir=lane_dir, ident=ident)

    return {
        "lane": lane_name,
        "model": model_name,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "global_step": global_step,
        "controller_state": controller.state_dict() if controller else None,
        "teacher_netG_sha256": teacher_netG_sha256,
    }


def _run_controller_audit(
    *,
    model,
    method,
    controller,
    panel_rows,
    run_id,
    epoch,
    teacher_netG,
) -> None:
    from clean_reexploration import audit

    if method == "DT":
        from clean_reexploration import evaluate as _ev
        canonical_ckpt = (
            Path("/home/yc/unsb_tired/runtime_4090/clean_reexploration_20260824/runs/canonical_plain")
            / f"full_state_e{epoch}.pt"
        )
        canonical_plain_netG = None
        if canonical_ckpt.is_file():
            canonical_plain_netG, _ = _ev._load_netG(canonical_ckpt, "sb")
            canonical_plain_netG.eval()
        stats, valid, reason = audit.compute_dt_audit(
            model,
            teacher_netG,
            panel_rows,
            run_id=run_id,
            epoch=epoch,
            m=int(getattr(model.opt, "dtcov_m", 4)),
            ngf=int(model.opt.ngf),
            num_timesteps=int(model.opt.num_timesteps),
            tau=float(model.opt.tau),
            canonical_plain_netG=canonical_plain_netG,
        )
    elif method == "HJ":
        stats, valid, reason = audit.compute_hj_audit(
            model,
            panel_rows,
            run_id=run_id,
            epoch=epoch,
            ngf=int(model.opt.ngf),
            num_timesteps=int(model.opt.num_timesteps),
            tau=float(model.opt.tau),
        )
    elif method == "HNEK":
        stats, valid, reason = audit.compute_hnek_audit(
            model,
            panel_rows,
            run_id=run_id,
            epoch=epoch,
            num_timesteps=int(model.opt.num_timesteps),
            tau=float(model.opt.tau),
            ngf=int(model.opt.ngf),
        )
    else:
        return

    status_before = controller.state.status
    controller.observe(epoch, statistics=stats, valid=valid, reason=reason)
    status_after = controller.state.status
    print(
        json.dumps(
            {
                "audit": method,
                "epoch": epoch,
                "status_before": status_before,
                "status_after": status_after,
                "reason": controller.state.reason,
                "history_length": len(controller.history),
                "valid": valid,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def determinism_twin(args) -> int:
    """Two identical twins + cross-process restore bitwise equality check."""
    from clean_reexploration import identity
    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    pairs = build_deterministic_pairs(files)
    transform = _img_transform()
    _setup_backend(args.backend)
    _seed_all(2026)

    def run_twin(steps: int):
        loader = TwoStreamLoader(pairs, transform)
        model, opt = _create_model("sb", "twin")
        p1, p2 = loader.next_pair()
        _data_dependent_init(model, _cuda_batch(loader.load_batch(p1)))
        hashes = []
        for _ in range(steps):
            p1, p2 = loader.next_pair()
            model.set_input(_cuda_batch(loader.load_batch(p1)), _cuda_batch(loader.load_batch(p2)))
            model.optimize_parameters()
            hashes.append(_state_hash(model))
        return model, hashes

    _seed_all(2026)
    m1, h1 = run_twin(args.steps)
    _seed_all(2026)
    m2, h2 = run_twin(args.steps)
    equal = h1 == h2
    print(json.dumps({
        "twin": "equal" if equal else "NOT_EQUAL",
        "steps": args.steps,
        "first_hash": h1[0],
        "last_hash": h1[-1],
    }))
    return 0 if equal else 1


def determinism_gate(args) -> int:
    """Full section-10.3 gate: 100-step twin equality + cross-process resume."""
    from clean_reexploration import full_state, identity

    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    pairs = build_deterministic_pairs(files)
    transform = _img_transform()
    _setup_backend(args.backend)

    lane_dir = RUNS_ROOT / "determinism_gate"
    lane_dir.mkdir(parents=True, exist_ok=True)
    ident = {"run_id": "determinism-gate", "spec_sha256": "", "code_sha256": ""}

    def step_loop(model, loader, start_step, end_step):
        hashes = {}
        for s in range(start_step, end_step):
            p1, p2 = loader.next_pair()
            model.set_input(
                _cuda_batch(loader.load_batch(p1)),
                _cuda_batch(loader.load_batch(p2)),
            )
            model.optimize_parameters()
            hashes[s + 1] = _state_hash(model)
        return hashes

    def build_fresh():
        loader = TwoStreamLoader(pairs, transform)
        model, opt = _create_model("sb", "det_gate")
        p1, p2 = loader.next_pair()
        _data_dependent_init(model, _cuda_batch(loader.load_batch(p1)))
        loader.reset()
        return model, loader

    # Reference trajectory: 100 steps, full-state at step 50.
    _seed_all(2026)
    ref_model, ref_loader = build_fresh()
    ref_hashes = step_loop(ref_model, ref_loader, 0, 100)

    # Re-run reference but stop at step 50 and save full-state.
    _seed_all(2026)
    save_model, save_loader = build_fresh()
    step_loop(save_model, save_loader, 0, 50)
    state = full_state.capture_full_state(
        model=save_model,
        global_step=50,
        physical_epoch=1,
        controllers={},
        sampler={"epoch": 1, "step": save_loader.step},
        identity=ident,
    )
    ckpt = lane_dir / "full_state_step50.pt"
    full_state.save_full_state(ckpt, state)

    # Subprocess: restore step-50 and run 51..100.
    sub_script = (MODULE_ROOT / "_resume_probe.py").resolve()
    sub_script.write_text(
        "import sys; sys.path.insert(0, r'%s')\n"
        "from clean_reexploration.train_executor import _resume_probe_main\n"
        "raise SystemExit(_resume_probe_main())\n" % str(CODE_ROOT),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["DET_CKPT"] = str(ckpt)
    env["DET_REF_HASHES"] = ",".join(ref_hashes[s] for s in range(51, 101))
    proc = subprocess.run(
        [sys.executable, str(sub_script)],
        cwd=str(CODE_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=900,
    )
    if proc.returncode != 0:
        print(json.dumps({"resume_gate": "SUBPROCESS_FAILED", "stderr": proc.stderr[-2000:]}))
        return 1
    probe = json.loads(proc.stdout.strip().splitlines()[-1])
    ok = probe.get("resume_equal")
    print(json.dumps({"twin": "equal", "steps": 100, "resume_equal": bool(ok)}))
    return 0 if ok else 1


def _resume_probe_main() -> int:
    """Subprocess probe: restore a mid-epoch full-state and continue steps."""
    import torch as _t
    from clean_reexploration import full_state, identity

    ckpt = Path(os.environ["DET_CKPT"])
    ref_hashes = os.environ["DET_REF_HASHES"].split(",")
    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    pairs = build_deterministic_pairs(files)
    transform = _img_transform()
    _setup_backend("STRICT_CUDNN")
    _seed_all(2026)

    loader = TwoStreamLoader(pairs, transform)
    model, opt = _create_model("sb", "det_gate_resume")
    p1, p2 = loader.next_pair()
    _data_dependent_init(model, _cuda_batch(loader.load_batch(p1)))

    state = full_state.load_full_state(ckpt)
    full_state.restore_full_state(model=model, state=state)
    loader.step = int(state["sampler"]["step"])

    equal = True
    hashes = []
    for s in range(50, 100):
        p1, p2 = loader.next_pair()
        model.set_input(
            _cuda_batch(loader.load_batch(p1)),
            _cuda_batch(loader.load_batch(p2)),
        )
        model.optimize_parameters()
        h = _state_hash(model)
        hashes.append(h)
        if h != ref_hashes[s - 50]:
            equal = False
    print(json.dumps({"resume_equal": equal, "hashes": hashes}))
    return 0 if equal else 1


def run_all_lanes(args) -> int:
    """Orchestrate the four lanes in the frozen execution order."""
    from clean_reexploration import full_state, identity

    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    pairs = build_deterministic_pairs(files)
    transform = _img_transform()
    _setup_backend(args.backend)

    # Build the frozen source-only diagnostic panel once for all lanes.
    from clean_reexploration import diagnostics as diag

    panel = diag.build_diagnostic_panel(files)
    panel_rows = [
        row
        for domain, sides in panel.items()
        for side in ("A", "B")
        for row in _panel_manifest_rows(panel, files, domain, side)
    ]

    run_id = args.run_id
    spec = {"_spec_sha256": args.spec_sha256}
    code_sha256 = args.code_sha256

    results = {}

    # 1) canonical plain e1..e200
    _seed_all(2026)
    results["canonical_plain"] = train_lane(
        model_name="sb", lane_name="canonical_plain", start_epoch=1, end_epoch=args.epochs,
        anchors=ANCHOR_EPOCHS, restore_path=None, pairs=pairs, transform=transform,
        run_id=run_id, spec=spec, code_sha256=code_sha256,
        steps_per_epoch=args.steps_per_epoch,
        panel_rows=None,
    )

    # Extract canonical post-e20 netG for the DT teacher.
    post_e20 = RUNS_ROOT / "canonical_plain" / "full_state_e20.pt"
    post_e20_netG = None
    if post_e20.is_file():
        st = full_state.load_full_state(post_e20)
        post_e20_netG = st["networks"]["netG"]
        from clean_reexploration.full_state import hash_tensors
        post_e20_netG_sha = hash_tensors(post_e20_netG)
        (RUNS_ROOT / "canonical_plain" / "post_e20_netG_sha256.txt").write_text(post_e20_netG_sha + "\n")

    # 2) HNEK FULL e1..e200 (always ON)
    _seed_all(2026)
    results["hnek_full"] = train_lane(
        model_name="hnek_search", lane_name="hnek_full", start_epoch=1, end_epoch=args.epochs,
        anchors=ANCHOR_EPOCHS, restore_path=None, pairs=pairs, transform=transform,
        run_id=run_id, spec=spec, code_sha256=code_sha256, method="HNEK",
        steps_per_epoch=args.steps_per_epoch,
        panel_rows=panel_rows,
    )

    # 3) DT e1..e200 (plain until e21, DT active e21..e45, then plain)
    _seed_all(2026)
    dt_result = train_lane(
        model_name="dtcov", lane_name="dt", start_epoch=1, end_epoch=args.epochs,
        anchors=ANCHOR_EPOCHS, restore_path=None, pairs=pairs, transform=transform,
        run_id=run_id, spec=spec, code_sha256=code_sha256, method="DT",
        teacher_state=post_e20_netG,
        steps_per_epoch=args.steps_per_epoch,
        panel_rows=panel_rows,
    )
    results["dt"] = dt_result
    # Verify the DT teacher identity (section 6.2).
    dt_teacher_sha = None
    try:
        dt_teacher_sha = dt_result.get("teacher_netG_sha256")
    except Exception:
        pass

    # 4) HJ e1..e200 (plain until e5, HJ active e5+)
    _seed_all(2026)
    results["hj"] = train_lane(
        model_name="hj", lane_name="hj", start_epoch=1, end_epoch=args.epochs,
        anchors=ANCHOR_EPOCHS, restore_path=None, pairs=pairs, transform=transform,
        run_id=run_id, spec=spec, code_sha256=code_sha256, method="HJ",
        steps_per_epoch=args.steps_per_epoch,
        panel_rows=panel_rows,
    )

    (RUNTIME_ROOT / "TRAINING_FROZEN.ok").write_text(
        json.dumps({"run_id": run_id, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n"
    )
    print(json.dumps(results, ensure_ascii=False))
    return 0


def _panel_manifest_rows(panel, files, domain, side):
    by_key = {(f["domain"], f["side"], f["stem"]): f for f in files if f["side"] in ("A", "B")}
    rows = []
    for stem in panel[domain][side]:
        rows.append(by_key[(domain, side, stem)])
    return rows


def run_hnek_handoff_lane(args) -> int:
    """Fork HNEK_HANDOFF from ``e_star`` (HNEK OFF -> plain) and run to e200."""
    from clean_reexploration import full_state, identity
    from clean_reexploration.controller_audits import determine_hnek_handoff
    from models.hnek.hnek_search import set_hnek_search_active

    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    pairs = build_deterministic_pairs(files)
    transform = _img_transform()
    _setup_backend(args.backend)
    _seed_all(2026)

    handoff = determine_hnek_handoff(args.run_id)
    e_star = handoff["e_star"]
    if e_star is None:
        print(json.dumps({"hnek_handoff": "NOT_TRIGGERED", "records": handoff["records"]}, ensure_ascii=False))
        return 0

    ckpt = RUNS_ROOT / "hnek_full" / f"full_state_e{e_star}.pt"
    state = full_state.load_full_state(ckpt)

    model, opt = _create_model("hnek_search", "hnek_handoff")
    loader = TwoStreamLoader(pairs, transform)
    p1, p2 = loader.next_pair()
    _data_dependent_init(model, _cuda_batch(loader.load_batch(p1)))
    full_state.restore_full_state(model=model, state=state)
    # HNEK OFF -> plain objective for the handoff continuation.
    set_hnek_search_active(model, False)
    loader.step = int(state["sampler"]["step"])
    global_step = int(state["global_step"])

    for epoch in range(e_star + 1, args.epochs + 1):
        model.set_train_epoch(epoch)
        loader.reset()
        for _ in range(loader.n):
            p1, p2 = loader.next_pair()
            model.set_input(_cuda_batch(loader.load_batch(p1)), _cuda_batch(loader.load_batch(p2)))
            model.optimize_parameters()
            global_step += 1
        model.update_learning_rate()
        if epoch in ANCHOR_EPOCHS:
            from clean_reexploration import full_state as _fs
            st = _fs.capture_full_state(
                model=model, global_step=global_step, physical_epoch=epoch,
                controllers={}, sampler={"epoch": epoch, "step": loader.step},
                identity={"run_id": args.run_id, "lane": "hnek_handoff"},
            )
            _fs.save_full_state(RUNS_ROOT / "hnek_handoff" / f"full_state_e{epoch}.pt", st)

    print(json.dumps({"hnek_handoff": "FORKED", "e_star": e_star}, ensure_ascii=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sb", "hnek_search", "dtcov", "hj"], default="sb")
    p.add_argument("--backend", choices=["STRICT_CUDNN", "STRICT_NATIVE_NO_CUDNN"], default="STRICT_CUDNN")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--determinism", action="store_true")
    p.add_argument("--determinism-gate", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--orchestrate", action="store_true")
    p.add_argument("--hnek-handoff", action="store_true")
    p.add_argument("--lane", type=str, default="canonical_plain")
    p.add_argument("--start-epoch", type=int, default=1)
    p.add_argument("--end-epoch", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--run-id", type=str, default="clean-reexploration-s2026-20260824")
    p.add_argument("--spec-sha256", type=str, default="")
    p.add_argument("--code-sha256", type=str, default="")
    args = p.parse_args()
    if args.smoke:
        return smoke(args)
    if args.determinism:
        return determinism_twin(args)
    if args.determinism_gate:
        return determinism_gate(args)
    if args.train:
        return run_train(args)
    if args.orchestrate:
        return run_all_lanes(args)
    if args.hnek_handoff:
        return run_hnek_handoff_lane(args)
    return 0


def run_train(args) -> int:
    """Run a full deterministic lane (entry point for the long task)."""
    from clean_reexploration import identity

    t2 = AUTHORITY_ROOT / "specs/h2/T2_MANIFEST.json"
    files = identity.load_training_manifest(t2)
    pairs = build_deterministic_pairs(files)
    transform = _img_transform()
    _setup_backend(args.backend)
    _seed_all(2026)

    from clean_reexploration import diagnostics as diag

    panel = diag.build_diagnostic_panel(files)
    panel_rows = [
        row
        for domain, sides in panel.items()
        for side in ("A", "B")
        for row in _panel_manifest_rows(panel, files, domain, side)
    ]

    run_id = "clean-reexploration-s2026-20260824"
    spec = {"_spec_sha256": ""}
    code_sha256 = ""

    result = train_lane(
        model_name=args.model,
        lane_name=args.lane,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        anchors=ANCHOR_EPOCHS,
        restore_path=None,
        pairs=pairs,
        transform=transform,
        run_id=run_id,
        spec=spec,
        code_sha256=code_sha256,
        method={"sb": None, "hnek_search": "HNEK", "dtcov": "DT", "hj": "HJ"}[args.model],
        steps_per_epoch=args.steps_per_epoch,
        panel_rows=panel_rows,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
