"""Streaming geometry for full network-state separations."""

from __future__ import annotations

import math

import torch


def _network_sums(first: dict, second: dict, reference_delta: dict | None = None) -> dict:
    keys = tuple(first)
    if keys != tuple(second):
        raise ValueError("network state identities differ")
    if reference_delta is not None and keys != tuple(reference_delta):
        raise ValueError("reference delta identity differs")
    norm_sq = 0.0
    ref_sq = 0.0
    dot = 0.0
    for key in keys:
        if not torch.is_tensor(first[key]) or not torch.is_floating_point(first[key]):
            continue
        delta = second[key].detach().double() - first[key].detach().double()
        norm_sq += float(torch.sum(delta * delta).item())
        if reference_delta is not None:
            ref = reference_delta[key].detach().double()
            ref_sq += float(torch.sum(ref * ref).item())
            dot += float(torch.sum(delta * ref).item())
    result = {"gap_norm": norm_sq ** 0.5}
    if reference_delta is not None:
        denominator = (norm_sq * ref_sq) ** 0.5
        result.update({
            "initial_gap_norm": ref_sq ** 0.5,
            "retention_ratio": (norm_sq / ref_sq) ** 0.5 if ref_sq > 0.0 else 0.0,
            "initial_direction_cosine": max(-1.0, min(1.0, dot / denominator))
            if denominator > 0.0 else 0.0,
        })
    return result


def network_delta(first: dict, second: dict) -> dict:
    if tuple(first) != tuple(second):
        raise ValueError("network state identities differ")
    return {
        key: second[key].detach().cpu() - first[key].detach().cpu()
        for key in first
        if torch.is_tensor(first[key]) and torch.is_floating_point(first[key])
    }


def model_gap_geometry(
    baseline_model: dict,
    variant_model: dict,
    initial_delta: dict[str, dict] | None = None,
) -> dict:
    names = tuple(baseline_model["networks"])
    if names != tuple(variant_model["networks"]):
        raise ValueError("model network identities differ")
    by_network = {}
    total_norm_sq = 0.0
    total_ref_sq = 0.0
    total_dot = 0.0
    for name in names:
        ref = None if initial_delta is None else initial_delta.get(name)
        result = _network_sums(
            baseline_model["networks"][name],
            variant_model["networks"][name],
            ref,
        )
        by_network[name] = result
        total_norm_sq += result["gap_norm"] ** 2
        if ref is not None:
            total_ref_sq += result["initial_gap_norm"] ** 2
            total_dot += (
                result["initial_direction_cosine"]
                * result["gap_norm"]
                * result["initial_gap_norm"]
            )
    global_result = {"gap_norm": math.sqrt(total_norm_sq)}
    if initial_delta is not None:
        denominator = math.sqrt(total_norm_sq * total_ref_sq)
        global_result.update({
            "initial_gap_norm": math.sqrt(total_ref_sq),
            "retention_ratio": math.sqrt(total_norm_sq / total_ref_sq)
            if total_ref_sq > 0.0 else 0.0,
            "initial_direction_cosine": max(-1.0, min(1.0, total_dot / denominator))
            if denominator > 0.0 else 0.0,
        })
    return {"global": global_result, "networks": by_network}


def initial_network_delta(baseline_model: dict, variant_model: dict) -> dict[str, dict]:
    return {
        name: network_delta(
            baseline_model["networks"][name], variant_model["networks"][name]
        )
        for name in baseline_model["networks"]
    }
