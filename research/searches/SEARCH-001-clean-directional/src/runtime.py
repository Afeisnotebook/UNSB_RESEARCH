"""Deterministic data streams, model construction and full-state checkpoints."""

from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .protocol import LaneSpec


REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_SRC = REPO_ROOT / "foundation" / "canonical" / "src"
FOUNDATION_ROOT = REPO_ROOT / "foundation"


def install_import_paths() -> None:
    import sys

    for path in (str(CANONICAL_SRC), str(FOUNDATION_ROOT), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def capture_rng(*, include_cuda: bool = True) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if include_cuda and torch.cuda.is_available()
            else None
        ),
    }


def restore_rng(state: dict, *, include_cuda: bool = True) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if include_cuda and state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def isolated_seed_state(seed: int) -> dict:
    saved = capture_rng(include_cuda=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    result = capture_rng(include_cuda=False)
    restore_rng(saved, include_cuda=False)
    return result


class SerializableDataStream:
    """Independent shuffled A stream plus worker-like CPU data RNG state."""

    def __init__(self, dataset, *, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.order_rng = np.random.default_rng(self.seed)
        self.order: list[int] = []
        self.cursor = 0
        self.epoch = 0
        self.data_rng = isolated_seed_state(self.seed + 1_000_003)

    def _reshuffle(self) -> None:
        self.order = self.order_rng.permutation(len(self.dataset)).tolist()
        self.cursor = 0
        self.epoch += 1
        self.dataset.current_epoch = self.epoch

    @staticmethod
    def _batch(item: dict) -> dict:
        return {
            key: (value.unsqueeze(0) if torch.is_tensor(value) else [value])
            for key, value in item.items()
        }

    def next(self) -> dict:
        if self.cursor >= len(self.order):
            self._reshuffle()
        index = self.order[self.cursor]
        self.cursor += 1
        main = capture_rng(include_cuda=False)
        restore_rng(self.data_rng, include_cuda=False)
        try:
            item = self.dataset[index]
            self.data_rng = capture_rng(include_cuda=False)
        finally:
            restore_rng(main, include_cuda=False)
        return self._batch(item)

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "order_rng": copy.deepcopy(self.order_rng.bit_generator.state),
            "order": list(self.order),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "data_rng": self.data_rng,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["seed"]) != self.seed:
            raise RuntimeError("data-stream seed mismatch")
        self.order_rng.bit_generator.state = copy.deepcopy(state["order_rng"])
        self.order = list(state["order"])
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.data_rng = state["data_rng"]
        self.dataset.current_epoch = self.epoch


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def train_stems(rows: list[dict], per_domain: int) -> set[str]:
    selected: set[str] = set()
    domains = sorted({row["domain"] for row in rows})
    for domain in domains:
        domain_rows = sorted(
            (row for row in rows if row["domain"] == domain and row["split"] == "train"),
            key=lambda row: int(row["order"]),
        )[:per_domain]
        if len(domain_rows) != per_domain:
            raise RuntimeError(f"{domain}: expected {per_domain} train identities")
        selected.update(f'{domain}__{row["stem"]}' for row in domain_rows)
    return selected


def restrict_dataset(dataset, selected: set[str]) -> None:
    def keep(path: str) -> bool:
        return Path(path).stem in selected

    dataset.A_paths = [path for path in dataset.A_paths if keep(path)]
    dataset.B_paths = [path for path in dataset.B_paths if keep(path)]
    dataset.A_size = len(dataset.A_paths)
    dataset.B_size = len(dataset.B_paths)
    if dataset.A_size != len(selected) or dataset.B_size != len(selected):
        raise RuntimeError(
            f"materialized view mismatch: A={dataset.A_size}, B={dataset.B_size}, "
            f"expected={len(selected)}"
        )
    if getattr(dataset, "_dcum_enabled", False):
        dataset._B_by_domain = {}
        for path in dataset.B_paths:
            domain, _ = dataset._domain_and_stem(path)
            dataset._B_by_domain.setdefault(domain, []).append(path)


def option_args(
    spec: LaneSpec,
    *,
    dataroot: Path,
    checkpoint_dir: Path,
    steps_per_epoch: int,
    total_steps: int,
    seed: int,
    gpu: int,
) -> list[str]:
    args = [
        "--dataroot", str(dataroot), "--name", spec.name,
        "--checkpoints_dir", str(checkpoint_dir), "--model", spec.model,
        "--mode", "sb", "--dataset_mode", "unaligned", "--direction", "AtoB",
        "--gpu_ids", str(gpu), "--seed", str(seed), "--batch_size", "1",
        "--num_threads", "0", "--n_epochs", "200", "--n_epochs_decay", "0",
        "--lr", "0.0001", "--lambda_GAN", "1", "--lambda_SB", "1",
        "--lambda_NCE", "1", "--tau", "0.01", "--nce_T", "0.07",
        "--nce_idt", "true", "--load_size", "128", "--crop_size", "128",
        "--preprocess", "resize_and_crop", "--no_flip", "--display_id", "-1",
        "--no_html", "--print_freq", "100000000", "--save_latest_freq", "100000000",
        "--save_epoch_freq", "100000000", "--search_steps_per_epoch",
        str(steps_per_epoch), "--search_ptq_seed", str(seed),
    ]
    for mechanism in spec.mechanisms:
        if mechanism == "lbst":
            args += ["--search_lbst", "true"]
        elif mechanism == "ptq":
            args += ["--search_ptq", "true"]
        elif mechanism == "dcum":
            args += ["--dcum", "true"]
        elif mechanism == "aeb":
            args += ["--search_aeb", "true"]
        else:
            raise ValueError(f"unknown mechanism: {mechanism}")
    if spec.model == "dtcov":
        args += [
            "--dtcov_lambda", "0.001", "--dtcov_lambda_schedule", "ramp_hold_cosine_decay",
            "--dtcov_ramp_start_epoch", "1", "--dtcov_ramp_end_epoch", "5",
            "--dtcov_decay_start_epoch", "15", "--dtcov_decay_end_epoch", "25",
            "--dtcov_m", "4", "--dtcov_time_mode", "actual",
            "--dtcov_domain_balance", "grouped_domain", "--dtcov_teacher", "frozen",
            "--dtcov_norm_mode", "domain_time", "--dtcov_signal_norm", "on",
            "--dtcov_warmup_iters", "0", "--dtcov_search_start_step",
            str(int(total_steps * 0.20)), "--dtcov_search_duration_steps",
            str(max(1, int(total_steps * 0.50))),
        ]
    elif spec.model == "hj":
        args += [
            "--hj_enable", "true", "--hj_layers", "0", "--hj_direction", "joint",
            "--hj_probe_mode", "central_consensus", "--hj_strength", "0.5",
            "--hj_boundary_scale", "0.001", "--hj_min_risk", "0.05",
            "--hj_search_start_step", str(int(total_steps * 0.20)),
        ]
    elif spec.model == "hnek_search":
        args += [
            "--hnek_gamma", "0.25", "--hnek_coord", "residual",
            "--hnek_horizon_mode", "physical", "--hnek_partial", "all",
        ]
    return args


def build_options(spec: LaneSpec, **kwargs):
    install_import_paths()
    from options.train_options import TrainOptions

    args = option_args(spec, **kwargs)
    return TrainOptions(" ".join(args)).parse()


def build_datasets(opt, rows: list[dict], per_domain: int):
    install_import_paths()
    from data.unaligned_dataset import UnalignedDataset

    selected = train_stems(rows, per_domain)
    first = UnalignedDataset(opt)
    second = UnalignedDataset(opt)
    restrict_dataset(first, selected)
    restrict_dataset(second, selected)
    return first, second


def build_model(opt, ddi_first: dict, ddi_second: dict):
    install_import_paths()
    from models import create_model

    model = create_model(opt)
    model.data_dependent_initialize(ddi_first, ddi_second)
    model.setup(opt)
    model.parallelize()
    return model


def inner(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def model_state(model) -> dict:
    return {
        "networks": {
            name: _cpu_clone(inner(getattr(model, "net" + name)).state_dict())
            for name in model.model_names
        },
        "optimizers": [_cpu_clone(optimizer.state_dict()) for optimizer in model.optimizers],
        "schedulers": [copy.deepcopy(scheduler.state_dict()) for scheduler in model.schedulers],
        "extra": _cpu_clone(model.get_extra_training_state()),
    }


def _optimizer_to(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_model_state(model, state: dict, *, load_extra: bool = True) -> None:
    for name in model.model_names:
        inner(getattr(model, "net" + name)).load_state_dict(state["networks"][name], strict=True)
    if len(model.optimizers) != len(state["optimizers"]):
        raise RuntimeError("optimizer count mismatch")
    for optimizer, saved in zip(model.optimizers, state["optimizers"]):
        optimizer.load_state_dict(saved)
        _optimizer_to(optimizer, model.device)
    if len(model.schedulers) != len(state["schedulers"]):
        raise RuntimeError("scheduler count mismatch")
    for scheduler, saved in zip(model.schedulers, state["schedulers"]):
        scheduler.load_state_dict(saved)
    if load_extra:
        model.load_extra_training_state(state["extra"])


def save_checkpoint(
    path: Path,
    *,
    model,
    spec: LaneSpec,
    step: int,
    target_steps: int,
    stream_a: SerializableDataStream,
    stream_b: SerializableDataStream,
    metadata: dict,
) -> None:
    payload = {
        "schema": "clean-unsb-directional-v1",
        "spec": spec.to_dict(),
        "step": int(step),
        "target_steps": int(target_steps),
        "model": model_state(model),
        "rng": capture_rng(),
        "stream_a": stream_a.state_dict(),
        "stream_b": stream_b.state_dict(),
        "metadata": copy.deepcopy(metadata),
        "saved_unix": time.time(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model,
    spec: LaneSpec,
    stream_a: SerializableDataStream,
    stream_b: SerializableDataStream,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clean-unsb-directional-v1":
        raise RuntimeError("checkpoint schema mismatch")
    if payload["spec"] != spec.to_dict():
        raise RuntimeError("lane spec mismatch")
    load_model_state(model, payload["model"])
    stream_a.load_state_dict(payload["stream_a"])
    stream_b.load_state_dict(payload["stream_b"])
    restore_rng(payload["rng"])
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
