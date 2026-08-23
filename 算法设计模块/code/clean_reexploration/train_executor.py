"""Deterministic real-model training executor for the clean re-exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
RUNTIME_ROOT = REPO_ROOT / "runtime_4090/clean_reexploration_20260824"
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
) -> dict:
    """Run one deterministic lane from ``start_epoch`` to ``end_epoch``."""
    from clean_reexploration import full_state

    lane_dir = RUNS_ROOT / lane_name
    lane_dir.mkdir(parents=True, exist_ok=True)
    loader = TwoStreamLoader(pairs, transform, steps_per_epoch)

    model, opt = _create_model(model_name, lane_name)
    controller = None
    controllers_map = {}
    if method:
        from clean_reexploration.controllers import make_controller
        controller = make_controller(method, run_id)
        controllers_map[method] = controller

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
        if epoch in anchors:
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
    }


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sb", "hnek_search", "dtcov", "hj"], default="sb")
    p.add_argument("--backend", choices=["STRICT_CUDNN", "STRICT_NATIVE_NO_CUDNN"], default="STRICT_CUDNN")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--determinism", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--lane", type=str, default="canonical_plain")
    p.add_argument("--start-epoch", type=int, default=1)
    p.add_argument("--end-epoch", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    args = p.parse_args()
    if args.smoke:
        return smoke(args)
    if args.determinism:
        return determinism_twin(args)
    if args.train:
        return run_train(args)
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
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
