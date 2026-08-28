"""Full-state V2, named optimizer states and component-safe transplantation."""

from __future__ import annotations

import copy
import hashlib
import io
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch

from .protocol import assert_target_blind


NETWORKS = ("G", "F", "D", "E")
OPTIMIZER_NAMES = ("G", "D", "E", "F")


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def torch_digest(value) -> str:
    buffer = io.BytesIO()
    torch.save(cpu_clone(value), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def exact_equal(left, right, path: str = "root") -> tuple[bool, str | None]:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False, path
        return (True, None) if torch.equal(left.cpu(), right.cpu()) else (False, path)
    if type(left) is not type(right):
        return False, path
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False, path
        for key in left:
            equal, mismatch = exact_equal(left[key], right[key], f"{path}.{key}")
            if not equal:
                return equal, mismatch
        return True, None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False, path
        for index, (a, b) in enumerate(zip(left, right)):
            equal, mismatch = exact_equal(a, b, f"{path}[{index}]")
            if not equal:
                return equal, mismatch
        return True, None
    try:
        equal = left == right
        if hasattr(equal, "all"):
            equal = bool(equal.all())
        return (True, None) if bool(equal) else (False, path)
    except Exception:
        return (True, None) if repr(left) == repr(right) else (False, path)


@dataclass(frozen=True)
class ComponentMask:
    networks: tuple[str, ...] = ()
    optimizers: tuple[str, ...] = ()
    schedulers: bool = False
    method_costate: bool = False
    streams_rng: bool = False

    def __post_init__(self):
        unknown_networks = set(self.networks) - set(NETWORKS)
        unknown_optimizers = set(self.optimizers) - set(OPTIMIZER_NAMES)
        if unknown_networks or unknown_optimizers:
            raise ValueError(f"unknown components: {unknown_networks}, {unknown_optimizers}")


@dataclass
class FullTrainingStateV2:
    step: int
    networks: dict[str, dict]
    named_optimizers: dict[str, dict]
    schedulers: list[dict]
    method_costate: dict
    rng: dict
    stream_a: dict
    stream_b: dict
    global_clock: dict
    shadow_state: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    schema: str = "clean-unsb-search004-full-state-v2"

    def __post_init__(self):
        assert_target_blind(self.diagnostics)

    def to_dict(self) -> dict:
        value = cpu_clone(self.__dict__)
        value["digest"] = self.digest()
        return value

    def digest(self) -> str:
        return torch_digest({key: value for key, value in self.__dict__.items() if key != "diagnostics"})


def optimizer_bindings(model) -> dict[str, tuple[torch.optim.Optimizer, list[tuple[str, torch.nn.Parameter]]]]:
    result = {}
    for name in OPTIMIZER_NAMES:
        optimizer = getattr(model, f"optimizer_{name}", None)
        network = getattr(model, f"net{name}", None)
        if optimizer is None or network is None:
            continue
        inner = network.module if isinstance(network, torch.nn.DataParallel) else network
        parameters = list(inner.named_parameters())
        optimizer_params = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if len(parameters) != len(optimizer_params) or any(a is not b for (_, a), b in zip(parameters, optimizer_params)):
            raise RuntimeError(f"optimizer/network parameter order mismatch for {name}")
        result[name] = (optimizer, parameters)
    return result


def export_named_optimizers(model) -> dict[str, dict]:
    exported = {}
    for name, (optimizer, parameters) in optimizer_bindings(model).items():
        parameter_name = {id(parameter): key for key, parameter in parameters}
        groups = []
        for group in optimizer.param_groups:
            metadata = {key: cpu_clone(value) for key, value in group.items() if key != "params"}
            metadata["parameters"] = [parameter_name[id(parameter)] for parameter in group["params"]]
            groups.append(metadata)
        states = {
            parameter_name[id(parameter)]: cpu_clone(state)
            for parameter, state in optimizer.state.items()
        }
        exported[name] = {"groups": groups, "states": states}
    return exported


def load_named_optimizers(model, saved: Mapping[str, dict], *, only: Iterable[str] | None = None) -> None:
    selected = set(only or saved.keys())
    bindings = optimizer_bindings(model)
    if not selected <= bindings.keys() or not selected <= saved.keys():
        raise RuntimeError(f"named optimizer set mismatch: {selected}, {bindings.keys()}, {saved.keys()}")
    for name in selected:
        optimizer, parameters = bindings[name]
        lookup = dict(parameters)
        spec = saved[name]
        expected_names = set(lookup)
        state_names = set(spec["states"])
        if not state_names <= expected_names:
            raise RuntimeError(f"unknown named optimizer states for {name}: {state_names - expected_names}")
        if len(spec["groups"]) != len(optimizer.param_groups):
            raise RuntimeError(f"optimizer group mismatch for {name}")
        optimizer.state.clear()
        for live_group, stored_group in zip(optimizer.param_groups, spec["groups"]):
            names = list(stored_group["parameters"])
            live_names = [key for key, parameter in parameters if any(parameter is p for p in live_group["params"])]
            if names != live_names:
                raise RuntimeError(f"optimizer parameter identity mismatch for {name}")
            for key, value in stored_group.items():
                if key != "parameters":
                    live_group[key] = cpu_clone(value)
        for parameter_name, state in spec["states"].items():
            parameter = lookup[parameter_name]
            loaded = {}
            for key, value in state.items():
                if torch.is_tensor(value):
                    if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and value.shape != parameter.shape:
                        raise RuntimeError(f"optimizer tensor shape mismatch: {name}.{parameter_name}.{key}")
                    if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and value.dtype != parameter.dtype:
                        raise RuntimeError(f"optimizer tensor dtype mismatch: {name}.{parameter_name}.{key}")
                    loaded[key] = value.to(parameter.device)
                else:
                    loaded[key] = copy.deepcopy(value)
            optimizer.state[parameter] = loaded


def capture_full_training_state_v2(
    model,
    runtime,
    *,
    step: int,
    rng: dict,
    stream_a: dict,
    stream_b: dict,
    global_clock: dict,
    method_costate: dict | None = None,
    shadow_state: dict | None = None,
    diagnostics: dict | None = None,
) -> FullTrainingStateV2:
    """Capture the canonical model through stable network/parameter names."""
    model_state = runtime.model_state(model)
    return FullTrainingStateV2(
        step=int(step),
        networks=cpu_clone(model_state["networks"]),
        named_optimizers=export_named_optimizers(model),
        schedulers=cpu_clone(model_state["schedulers"]),
        method_costate=cpu_clone(
            model_state.get("extra", {}) if method_costate is None else method_costate
        ),
        shadow_state=cpu_clone(shadow_state or {}),
        rng=cpu_clone(rng),
        stream_a=cpu_clone(stream_a),
        stream_b=cpu_clone(stream_b),
        global_clock=cpu_clone(global_clock),
        diagnostics=cpu_clone(diagnostics or {}),
    )


def load_full_training_state_v2(model, runtime, state: FullTrainingStateV2) -> None:
    """Restore V2 while rejecting every structural or numeric type mismatch."""
    live = runtime.model_state(model)
    if set(state.networks) != set(live["networks"]):
        raise RuntimeError("V2 network set mismatch")
    for network_name, stored in state.networks.items():
        current = live["networks"][network_name]
        if set(stored) != set(current):
            raise RuntimeError(f"V2 parameter-name mismatch for {network_name}")
        for parameter_name, value in stored.items():
            reference = current[parameter_name]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise RuntimeError(
                    f"V2 tensor mismatch: {network_name}.{parameter_name}"
                )
    if len(state.schedulers) != len(live["schedulers"]):
        raise RuntimeError("V2 scheduler count mismatch")
    runtime.load_model_state(
        model,
        {
            "networks": cpu_clone(state.networks),
            # Load the live positional containers first; named state is the
            # only authoritative optimizer payload and is applied below.
            "optimizers": cpu_clone(live["optimizers"]),
            "schedulers": cpu_clone(state.schedulers),
            "extra": cpu_clone(state.method_costate),
        },
        load_extra=False,
    )
    load_named_optimizers(model, state.named_optimizers)
    model.load_extra_training_state(cpu_clone(state.method_costate))


def zero_named_optimizer(model, names: Iterable[str]) -> None:
    bindings = optimizer_bindings(model)
    for name in names:
        if name not in bindings:
            raise RuntimeError(f"optimizer {name} is unavailable")
        bindings[name][0].state.clear()


def checkpoint_model_state(payload: dict) -> dict:
    if payload.get("schema") not in {
        "clean-unsb-directional-v1", "clean-unsb-search003-candidate-state-v1",
        "clean-unsb-search003-plain-state-v1", "clean-unsb-search003-e0-v1",
    }:
        raise RuntimeError(f"unsupported checkpoint schema: {payload.get('schema')}")
    return payload["model"]


def checkpoint_method_costate(payload: dict) -> dict:
    if payload.get("proposal_costate") is not None:
        return cpu_clone(payload["proposal_costate"])
    return cpu_clone(payload["model"].get("extra", {}))


def validate_checkpoint_payload(payload: dict) -> dict:
    model = checkpoint_model_state(payload)
    required_top = {"step", "model", "rng", "stream_a", "stream_b"}
    required_model = {"networks", "optimizers", "schedulers", "extra"}
    missing = required_top - payload.keys()
    model_missing = required_model - model.keys()
    network_missing = set(NETWORKS) - model["networks"].keys()
    if missing or model_missing or network_missing:
        raise RuntimeError(f"incomplete checkpoint: {missing}, {model_missing}, {network_missing}")
    if len(model["optimizers"]) != 4 or len(model["schedulers"]) != 4:
        raise RuntimeError("checkpoint does not contain four optimizers/schedulers")
    return {
        "schema": payload["schema"],
        "step": int(payload["step"]),
        "network_names": list(model["networks"]),
        "optimizer_count": len(model["optimizers"]),
        "scheduler_count": len(model["schedulers"]),
        "rng_keys": sorted(payload["rng"]),
        "stream_keys": sorted(payload["stream_a"]),
        "method_costate_keys": sorted(checkpoint_method_costate(payload)),
        "digest": torch_digest(payload),
    }
