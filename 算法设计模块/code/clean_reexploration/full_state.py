"""Deterministic full training-state capture, save, load and restore."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def hash_tensors(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_tensor_bytes(state_dict[key]))
        digest.update(b"\x00")
    return digest.hexdigest()


def _rng_bundle() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().numpy().tolist(),
        "torch_cuda": [s.cpu().numpy().tolist() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_bundle(bundle: dict) -> None:
    random.setstate(bundle["python"])
    np.random.set_state(bundle["numpy"])
    torch.random.set_rng_state(torch.as_tensor(bundle["torch_cpu"], dtype=torch.uint8))
    if torch.cuda.is_available():
        states = [torch.as_tensor(s, dtype=torch.uint8).cuda() for s in bundle["torch_cuda"]]
        torch.cuda.set_rng_state_all(states)


def capture_full_state(
    *,
    model,
    global_step: int,
    physical_epoch: int,
    controllers: dict[str, Any],
    sampler: dict[str, Any],
    identity: dict[str, Any],
) -> dict:
    """Capture a complete, resumable training state."""
    state = {
        "schema_version": 1,
        "global_step": int(global_step),
        "physical_epoch": int(physical_epoch),
        "networks": {
            "netG": model.netG.module.state_dict() if hasattr(model.netG, "module") else model.netG.state_dict(),
            "netF": model.netF.module.state_dict() if hasattr(model.netF, "module") else model.netF.state_dict(),
        },
        "optimizers": {},
        "schedulers": {},
        "rng": _rng_bundle(),
        "controllers": {k: v.state_dict() for k, v in controllers.items()},
        "sampler": sampler,
        "identity": identity,
        "model_extra": {},
    }
    if model.isTrain:
        for name in ("netD", "netE"):
            net = getattr(model, name)
            state["networks"][name] = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
        for name in ("G", "D", "E", "F"):
            opt = getattr(model, f"optimizer_{name}", None)
            if opt is not None:
                state["optimizers"][name] = opt.state_dict()
        for i, sched in enumerate(getattr(model, "schedulers", [])):
            state["schedulers"][str(i)] = sched.state_dict()
        # Capture lane-specific mutable intervention state.
        for attr in (
            "_dtcov_epoch",
            "_dtcov_iter",
            "_dtcov_activated",
            "_dtcov_mismatch_ema",
            "_dtcov_plateau",
            "_dtcov_adaptive_lambda",
            "_dtcov_step_in_epoch",
            "hj_epoch",
            "_hj_step_in_epoch",
            "_hj_conflict_ema",
            "_hj_conflict_peak",
            "_hj_adaptive_weight",
            "hnek_active",
        ):
            if hasattr(model, attr):
                state["model_extra"][attr] = getattr(model, attr)
        if hasattr(model, "dtcov"):
            state["model_extra"]["dtcov_iter"] = model.dtcov.iter
            state["model_extra"]["dtcov_stats"] = {
                f"{d}\x1f{t}": dict(v)
                for (d, t), v in model.dtcov.stats.store.items()
            }
            if model.dtcov.teacher is not None:
                state["model_extra"]["teacher_hash"] = hash_tensors(model.dtcov.teacher.state_dict())
    state["networks_hash"] = hash_tensors(state["networks"]["netG"])
    return state


def restore_full_state(*, model, state: dict) -> None:
    """Restore a captured full state into ``model``."""
    _restore_rng_bundle(state["rng"])
    for name, sd in state["networks"].items():
        net = getattr(model, name)
        target = net.module if hasattr(net, "module") else net
        target.load_state_dict(sd)
    if model.isTrain:
        for name, sd in state["optimizers"].items():
            opt = getattr(model, f"optimizer_{name}", None)
            if opt is not None:
                opt.load_state_dict(sd)
        for i, sched in enumerate(getattr(model, "schedulers", [])):
            key = str(i)
            if key in state["schedulers"]:
                sched.load_state_dict(state["schedulers"][key])
        extra = state.get("model_extra", {})
        if hasattr(model, "dtcov"):
            if "dtcov_iter" in extra:
                model.dtcov.iter = extra["dtcov_iter"]
            if "dtcov_stats" in extra:
                model.dtcov.stats.store = {
                    (str(k).split("\x1f", 1)[0], int(str(k).split("\x1f", 1)[1])): dict(v)
                    for k, v in extra["dtcov_stats"].items()
                }
        for attr, value in extra.items():
            if attr in ("dtcov_iter", "dtcov_stats", "teacher_hash"):
                continue
            if hasattr(model, attr):
                setattr(model, attr, value)


def save_full_state(path: Path, state: dict) -> dict:
    """Atomically save a full state (temp file + fsync + rename) with a sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(state, buf)
    payload = buf.getvalue()
    payload_sha = hashlib.sha256(payload).hexdigest()

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".full_state_", suffix=".pt")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    sidecar = {
        "payload_sha256": payload_sha,
        "size": len(payload),
        "physical_epoch": state.get("physical_epoch"),
        "global_step": state.get("global_step"),
        "networks_hash": state.get("networks_hash"),
    }
    sidecar_path = path.with_suffix(path.suffix + ".sha256")
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar


def load_full_state(path: Path) -> dict:
    path = Path(path)
    payload = path.read_bytes()
    sidecar_path = path.with_suffix(path.suffix + ".sha256")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != sidecar["payload_sha256"]:
        raise RuntimeError(f"full-state payload hash mismatch: {path}")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    return state
