"""Generation-1 candidate construction on the accepted SEARCH-001 runtime.

This module owns candidate-specific options so SEARCH-001's frozen lanes stay
unchanged.  It currently exposes ELIPRC first; the gradient operators are added
only after their pure mathematical invariants pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .search001_compat import modules


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    lane_name: str
    model: str
    option_overrides: tuple[tuple[str, str], ...] = ()
    installer: str | None = None

    def lane_spec(self):
        protocol = modules()[0]
        return protocol.LaneSpec(
            name=self.lane_name,
            model=self.model,
            family="search005_generation1",
        )


PLAIN = CandidateSpec("PLAIN", "plain", "sb")
ELIPRC = CandidateSpec(
    "G1-HNEK-ELIPRC",
    "g1_hnek_eliprc",
    "hnek_search",
    (
        ("--hnek_gamma", "0.5"),
        ("--hnek_coord", "residual"),
        ("--hnek_horizon_mode", "physical"),
        ("--hnek_partial", "entropy_only"),
    ),
)
CNDRP = CandidateSpec(
    "G1-DT-CNDRP",
    "g1_dt_cndrp",
    "sb",
    installer="cndrp",
)
ACMP = CandidateSpec(
    "G1-HJ-ACMP",
    "g1_hj_acmp",
    "sb",
    installer="acmp",
)
FBCMP = CandidateSpec(
    "G2-HJ-FBCMP",
    "g2_hj_fbcmp",
    "sb",
    installer="fbcmp",
)
BCAVP = CandidateSpec(
    "G1-DT-HNEK-BCAVP",
    "g1_dt_hnek_bcavp",
    "sb",
    installer="bcavp",
)
PHCRP = CandidateSpec(
    "G1-HNEK-PHCRP",
    "g1_hnek_phcrp",
    "sb",
    installer="phcrp",
)
PHRSUP = CandidateSpec(
    "G2-HNEK-PHRSUP",
    "g2_hnek_phrsup",
    "sb",
    installer="phrsup",
)
BCNRP = CandidateSpec(
    "G2-DT-BCNRP",
    "g2_dt_bcnrp",
    "sb",
    installer="bcnrp",
)
PCOA = CandidateSpec(
    "G1-GAME-PCOA",
    "g1_game_pcoa",
    "sb",
    installer="pcoa",
)
NPOOA = CandidateSpec(
    "G2-GAME-NPOOA",
    "g2_game_npooa",
    "sb",
    installer="npooa",
)


def replace_option(arguments: list[str], name: str, value: str) -> None:
    """Replace one already-declared option without creating duplicates."""
    positions = [index for index, token in enumerate(arguments) if token == name]
    if len(positions) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(positions)}")
    index = positions[0]
    if index + 1 >= len(arguments):
        raise RuntimeError(f"option has no value: {name}")
    arguments[index + 1] = str(value)


def build_options(
    spec: CandidateSpec,
    *,
    dataroot: Path,
    checkpoint_dir: Path,
    steps_per_epoch: int,
    total_steps: int,
    seed: int,
    gpu: int,
):
    _, runtime, _ = modules()
    runtime.install_import_paths()
    from options.train_options import TrainOptions

    arguments = runtime.option_args(
        spec.lane_spec(),
        dataroot=dataroot,
        checkpoint_dir=checkpoint_dir,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    for name, value in spec.option_overrides:
        replace_option(arguments, name, value)
    return TrainOptions(" ".join(arguments)).parse()


def create_e0(
    path: Path,
    *,
    rows: list[dict],
    train_view: Path,
    option_dir: Path,
    per_domain: int,
    total_steps: int,
    seed: int,
    gpu: int,
) -> dict:
    """Create or load one plain initialization shared by every lane."""
    _, runtime, _ = modules()
    path = Path(path)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "clean-unsb-search005-e0-v1":
            raise RuntimeError("SEARCH-005 e0 schema mismatch")
        expected = (int(per_domain), int(total_steps), int(seed))
        actual = (
            int(payload["per_domain"]),
            int(payload["total_steps"]),
            int(payload["seed"]),
        )
        if actual != expected:
            raise RuntimeError(f"SEARCH-005 e0 contract mismatch: {actual} != {expected}")
        return payload

    steps_per_epoch = 6 * int(per_domain)
    runtime.seed_everything(seed)
    opt = build_options(
        PLAIN,
        dataroot=train_view,
        checkpoint_dir=option_dir,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    dataset_a, dataset_b = runtime.build_datasets(opt, rows, per_domain)
    stream_a = runtime.SerializableDataStream(dataset_a, seed=seed + 101)
    stream_b = runtime.SerializableDataStream(dataset_b, seed=seed + 202)
    model = runtime.build_model(opt, stream_a.next(), stream_b.next())
    payload = {
        "schema": "clean-unsb-search005-e0-v1",
        "model": runtime.model_state(model),
        "rng": runtime.capture_rng(),
        "stream_a": stream_a.state_dict(),
        "stream_b": stream_b.state_dict(),
        "per_domain": int(per_domain),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": int(total_steps),
        "seed": int(seed),
        "confirmation20_opened": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    del model
    torch.cuda.empty_cache()
    return payload


def prepare_lane(
    spec: CandidateSpec,
    *,
    e0: dict,
    rows: list[dict],
    train_view: Path,
    option_dir: Path,
    per_domain: int,
    total_steps: int,
    seed: int,
    gpu: int,
):
    _, runtime, _ = modules()
    steps_per_epoch = 6 * int(per_domain)
    runtime.seed_everything(seed)
    opt = build_options(
        spec,
        dataroot=train_view,
        checkpoint_dir=option_dir,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    dataset_a, dataset_b = runtime.build_datasets(opt, rows, per_domain)
    stream_a = runtime.SerializableDataStream(dataset_a, seed=seed + 101)
    stream_b = runtime.SerializableDataStream(dataset_b, seed=seed + 202)
    model = runtime.build_model(opt, stream_a.next(), stream_b.next())
    if spec.installer == "cndrp":
        from .model_operators import install_cndrp

        install_cndrp(model, seed=seed)
    elif spec.installer == "bcnrp":
        from .model_operators import install_cndrp

        install_cndrp(model, seed=seed, blockwise=True)
    elif spec.installer == "acmp":
        from .model_operators import install_acmp

        install_acmp(model, seed=seed)
    elif spec.installer == "fbcmp":
        from .model_operators import install_acmp

        install_acmp(model, seed=seed, future_consensus=True)
    elif spec.installer == "bcavp":
        from .model_operators import install_bcavp

        install_bcavp(model)
    elif spec.installer == "phcrp":
        from .model_operators import install_phcrp

        install_phcrp(model)
    elif spec.installer == "phrsup":
        from .model_operators import install_phrsup

        install_phrsup(model)
    elif spec.installer == "pcoa":
        from .model_operators import install_pcoa

        install_pcoa(model)
    elif spec.installer == "npooa":
        from .model_operators import install_npooa

        install_npooa(model)
    elif spec.installer is not None:
        raise ValueError(f"unknown candidate installer: {spec.installer}")
    runtime.load_model_state(model, e0["model"], load_extra=False)
    stream_a.load_state_dict(copy.deepcopy(e0["stream_a"]))
    stream_b.load_state_dict(copy.deepcopy(e0["stream_b"]))
    runtime.restore_rng(copy.deepcopy(e0["rng"]))
    model.set_search_step(0, total_steps)
    return model, stream_a, stream_b


def comparable_state(model) -> dict:
    """State that must match plain; excludes candidate bookkeeping only."""
    _, runtime, _ = modules()
    state = runtime.model_state(model)
    return {
        "networks": state["networks"],
        "optimizers": state["optimizers"],
        "schedulers": state["schedulers"],
    }


def nested_equal(first, second) -> bool:
    if torch.is_tensor(first):
        return torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            nested_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            nested_equal(left, right) for left, right in zip(first, second)
        )
    result = first == second
    return bool(result) if not isinstance(result, np.ndarray) else bool(result.all())
