"""GPU engineering gates for SEARCH-005 candidates."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from .candidate_runtime import (
    ACMP,
    BCAVP,
    BCNRP,
    CNDRP,
    ELIPRC,
    FBCMP,
    NPOOA,
    PLAIN,
    PHCRP,
    PHRSUP,
    PCOA,
    comparable_state,
    create_e0,
    nested_equal,
    prepare_lane,
)
from .search001_compat import modules


def _one_update(model, stream_a, stream_b, step: int, total_steps: int) -> None:
    model.set_train_epoch(1)
    model.set_search_step(step, total_steps)
    model.set_input(stream_a.next(), stream_b.next())
    model.optimize_parameters()


def _stream_state(stream) -> dict:
    return copy.deepcopy(stream.state_dict())


def _rng_state(runtime) -> dict:
    return copy.deepcopy(runtime.capture_rng())


def _release(model) -> None:
    del model
    torch.cuda.empty_cache()


def run_eliprc_invariants(
    *,
    rows: list[dict],
    train_view: Path,
    output: Path,
    seed: int,
    gpu: int,
) -> dict:
    """Verify endpoint identity, zero intervention and exact disk resume."""
    _, runtime, _ = modules()
    output = Path(output)
    gate_dir = output / "generation1_gate"
    total_steps = 8
    per_domain = 25
    e0 = create_e0(
        gate_dir / "e0.pt",
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )

    # Active ELIPRC must leave the endpoint forward map byte-identical.
    plain, plain_a, plain_b = prepare_lane(
        PLAIN,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    plain.set_input(plain_a.next(), plain_b.next())
    plain.forward()
    plain_endpoints = (plain.fake_B.detach().cpu(), plain.fake_B2.detach().cpu())
    plain_keys = {
        name: tuple(runtime.inner(getattr(plain, "net" + name)).state_dict())
        for name in plain.model_names
    }
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        ELIPRC,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    candidate.set_input(candidate_a.next(), candidate_b.next())
    candidate.forward()
    endpoint_identity = (
        torch.equal(plain_endpoints[0], candidate.fake_B.detach().cpu())
        and torch.equal(plain_endpoints[1], candidate.fake_B2.detach().cpu())
    )
    state_key_identity = plain_keys == {
        name: tuple(runtime.inner(getattr(candidate, "net" + name)).state_dict())
        for name in candidate.model_names
    }
    status = __import__(
        "models.hnek.hnek_search", fromlist=["hnek_search_installation_status"]
    ).hnek_search_installation_status(candidate)
    _release(candidate)

    # Disabling the coordinate must reproduce one complete plain update,
    # including D/E/G/F optimizer states and global RNG consumption.
    plain, plain_a, plain_b = prepare_lane(
        PLAIN,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    _one_update(plain, plain_a, plain_b, 0, total_steps)
    plain_state = comparable_state(plain)
    plain_rng = _rng_state(runtime)
    plain_streams = (_stream_state(plain_a), _stream_state(plain_b))
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        ELIPRC,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    from models.hnek.hnek_search import set_hnek_search_active

    set_hnek_search_active(candidate, False)
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    zero_state = comparable_state(candidate)
    zero_rng = _rng_state(runtime)
    zero_streams = (_stream_state(candidate_a), _stream_state(candidate_b))
    zero_components = {
        "model_and_optimizer": nested_equal(plain_state, zero_state),
        "rng": nested_equal(plain_rng, zero_rng),
        "data_stream_a": nested_equal(plain_streams[0], zero_streams[0]),
        "data_stream_b": nested_equal(plain_streams[1], zero_streams[1]),
    }
    zero_intervention_exact = all(zero_components.values())
    _release(candidate)

    # Continuous two-step reference.
    continuous, continuous_a, continuous_b = prepare_lane(
        ELIPRC,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    _one_update(continuous, continuous_a, continuous_b, 0, total_steps)
    _one_update(continuous, continuous_a, continuous_b, 1, total_steps)
    continuous_state = comparable_state(continuous)
    continuous_rng = _rng_state(runtime)
    continuous_streams = (_stream_state(continuous_a), _stream_state(continuous_b))
    _release(continuous)

    # Split at one step through the same full-state checkpoint API used by runs.
    split, split_a, split_b = prepare_lane(
        ELIPRC,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    _one_update(split, split_a, split_b, 0, total_steps)
    resume_path = gate_dir / "eliprc_resume_step1.pt"
    runtime.save_checkpoint(
        resume_path,
        model=split,
        spec=ELIPRC.lane_spec(),
        step=1,
        target_steps=total_steps,
        stream_a=split_a,
        stream_b=split_b,
        metadata={"gate": "full_state_resume"},
    )
    _release(split)

    resumed, resumed_a, resumed_b = prepare_lane(
        ELIPRC,
        e0=e0,
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )
    payload = runtime.load_checkpoint(
        resume_path,
        model=resumed,
        spec=ELIPRC.lane_spec(),
        stream_a=resumed_a,
        stream_b=resumed_b,
    )
    if int(payload["step"]) != 1:
        raise RuntimeError("resume checkpoint step mismatch")
    _one_update(resumed, resumed_a, resumed_b, 1, total_steps)
    resumed_streams = (_stream_state(resumed_a), _stream_state(resumed_b))
    resume_components = {
        "model_and_optimizer": nested_equal(continuous_state, comparable_state(resumed)),
        "rng": nested_equal(continuous_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(continuous_streams[0], resumed_streams[0]),
        "data_stream_b": nested_equal(continuous_streams[1], resumed_streams[1]),
    }
    resume_exact = all(resume_components.values())
    _release(resumed)

    passed = all(
        (endpoint_identity, state_key_identity, zero_intervention_exact, resume_exact)
    )
    result = {
        "schema": "clean-unsb-search005-generation1-gate-v1",
        "candidate_id": ELIPRC.candidate_id,
        "passed": passed,
        "endpoint_forward_byte_identical": endpoint_identity,
        "network_state_keys_identical": state_key_identity,
        "zero_intervention_full_update_exact": zero_intervention_exact,
        "zero_intervention_components": zero_components,
        "full_state_resume_exact": resume_exact,
        "resume_components": resume_components,
        "installation_status": status,
        "paired_target_readable_by_candidate": False,
        "confirmation20_opened": False,
    }
    if not passed:
        raise RuntimeError(f"ELIPRC engineering gate failed: {result}")
    return result


def run_cndrp_invariants(
    *,
    rows: list[dict],
    train_view: Path,
    output: Path,
    seed: int,
    gpu: int,
    candidate_spec=CNDRP,
) -> dict:
    """Verify that CNDRP is endpoint-neutral, disable-safe and resumable."""
    _, runtime, _ = modules()
    from .model_operators import set_cndrp_active

    output = Path(output)
    gate_dir = output / "generation1_gate"
    total_steps = 8
    per_domain = 25
    e0 = create_e0(
        gate_dir / "e0.pt",
        rows=rows,
        train_view=train_view,
        option_dir=gate_dir / "options",
        per_domain=per_domain,
        total_steps=total_steps,
        seed=seed,
        gpu=gpu,
    )

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    plain.set_input(plain_a.next(), plain_b.next())
    plain.forward()
    plain_endpoints = (plain.fake_B.detach().cpu(), plain.fake_B2.detach().cpu())
    plain_keys = {
        name: tuple(runtime.inner(getattr(plain, "net" + name)).state_dict())
        for name in plain.model_names
    }
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    candidate.set_input(candidate_a.next(), candidate_b.next())
    candidate.forward()
    endpoint_identity = (
        torch.equal(plain_endpoints[0], candidate.fake_B.detach().cpu())
        and torch.equal(plain_endpoints[1], candidate.fake_B2.detach().cpu())
    )
    state_key_identity = plain_keys == {
        name: tuple(runtime.inner(getattr(candidate, "net" + name)).state_dict())
        for name in candidate.model_names
    }
    _release(candidate)

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(plain, plain_a, plain_b, 0, total_steps)
    plain_state = comparable_state(plain)
    plain_rng = _rng_state(runtime)
    plain_streams = (_stream_state(plain_a), _stream_state(plain_b))
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    set_cndrp_active(candidate, False)
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    zero_components = {
        "model_and_optimizer": nested_equal(plain_state, comparable_state(candidate)),
        "rng": nested_equal(plain_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(plain_streams[0], _stream_state(candidate_a)),
        "data_stream_b": nested_equal(plain_streams[1], _stream_state(candidate_b)),
    }
    zero_intervention_exact = all(zero_components.values())
    _release(candidate)

    continuous, continuous_a, continuous_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(continuous, continuous_a, continuous_b, 0, total_steps)
    _one_update(continuous, continuous_a, continuous_b, 1, total_steps)
    continuous_state = comparable_state(continuous)
    continuous_rng = _rng_state(runtime)
    continuous_streams = (
        _stream_state(continuous_a), _stream_state(continuous_b)
    )
    last_diagnostics = copy.deepcopy(continuous._search005_cndrp_last)
    _release(continuous)

    split, split_a, split_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(split, split_a, split_b, 0, total_steps)
    resume_path = gate_dir / f"{candidate_spec.lane_name}_resume_step1.pt"
    runtime.save_checkpoint(
        resume_path, model=split, spec=candidate_spec.lane_spec(), step=1,
        target_steps=total_steps, stream_a=split_a, stream_b=split_b,
        metadata={"gate": "full_state_resume"},
    )
    _release(split)

    resumed, resumed_a, resumed_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    runtime.load_checkpoint(
        resume_path, model=resumed, spec=candidate_spec.lane_spec(),
        stream_a=resumed_a, stream_b=resumed_b,
    )
    _one_update(resumed, resumed_a, resumed_b, 1, total_steps)
    resume_components = {
        "model_and_optimizer": nested_equal(continuous_state, comparable_state(resumed)),
        "rng": nested_equal(continuous_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(continuous_streams[0], _stream_state(resumed_a)),
        "data_stream_b": nested_equal(continuous_streams[1], _stream_state(resumed_b)),
    }
    resume_exact = all(resume_components.values())
    _release(resumed)

    passed = all((endpoint_identity, state_key_identity, zero_intervention_exact, resume_exact))
    result = {
        "schema": "clean-unsb-search005-generation1-gate-v1",
        "candidate_id": candidate_spec.candidate_id,
        "passed": passed,
        "endpoint_forward_byte_identical": endpoint_identity,
        "network_state_keys_identical": state_key_identity,
        "zero_intervention_full_update_exact": zero_intervention_exact,
        "zero_intervention_components": zero_components,
        "full_state_resume_exact": resume_exact,
        "resume_components": resume_components,
        "sample_operator_diagnostics": last_diagnostics,
        "paired_target_readable_by_candidate": False,
        "confirmation20_opened": False,
    }
    if not passed:
        raise RuntimeError(f"{candidate_spec.candidate_id} engineering gate failed: {result}")
    return result


def run_bcnrp_invariants(**kwargs) -> dict:
    return run_cndrp_invariants(candidate_spec=BCNRP, **kwargs)


def run_acmp_invariants(
    *,
    rows: list[dict],
    train_view: Path,
    output: Path,
    seed: int,
    gpu: int,
    candidate_spec=ACMP,
) -> dict:
    """Verify that ACMP is endpoint-neutral, disable-safe and resumable."""
    _, runtime, _ = modules()
    from .model_operators import set_acmp_active

    gate_dir = Path(output) / "generation1_gate"
    total_steps, per_domain = 8, 25
    e0 = create_e0(
        gate_dir / "e0.pt", rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    plain.set_input(plain_a.next(), plain_b.next())
    plain.forward()
    plain_endpoints = (plain.fake_B.detach().cpu(), plain.fake_B2.detach().cpu())
    plain_keys = {
        name: tuple(runtime.inner(getattr(plain, "net" + name)).state_dict())
        for name in plain.model_names
    }
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    candidate.set_input(candidate_a.next(), candidate_b.next())
    candidate.forward()
    endpoint_identity = (
        torch.equal(plain_endpoints[0], candidate.fake_B.detach().cpu())
        and torch.equal(plain_endpoints[1], candidate.fake_B2.detach().cpu())
    )
    state_key_identity = plain_keys == {
        name: tuple(runtime.inner(getattr(candidate, "net" + name)).state_dict())
        for name in candidate.model_names
    }
    _release(candidate)

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(plain, plain_a, plain_b, 0, total_steps)
    plain_state, plain_rng = comparable_state(plain), _rng_state(runtime)
    plain_streams = (_stream_state(plain_a), _stream_state(plain_b))
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    set_acmp_active(candidate, False)
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    zero_components = {
        "model_and_optimizer": nested_equal(plain_state, comparable_state(candidate)),
        "rng": nested_equal(plain_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(plain_streams[0], _stream_state(candidate_a)),
        "data_stream_b": nested_equal(plain_streams[1], _stream_state(candidate_b)),
    }
    zero_intervention_exact = all(zero_components.values())
    _release(candidate)

    continuous, continuous_a, continuous_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(continuous, continuous_a, continuous_b, 0, total_steps)
    _one_update(continuous, continuous_a, continuous_b, 1, total_steps)
    continuous_state, continuous_rng = comparable_state(continuous), _rng_state(runtime)
    continuous_streams = (_stream_state(continuous_a), _stream_state(continuous_b))
    last_diagnostics = copy.deepcopy(continuous._search005_acmp_last)
    _release(continuous)

    split, split_a, split_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(split, split_a, split_b, 0, total_steps)
    resume_path = gate_dir / "acmp_resume_step1.pt"
    runtime.save_checkpoint(
        resume_path, model=split, spec=candidate_spec.lane_spec(), step=1,
        target_steps=total_steps, stream_a=split_a, stream_b=split_b,
        metadata={"gate": "full_state_resume"},
    )
    _release(split)

    resumed, resumed_a, resumed_b = prepare_lane(
        candidate_spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    runtime.load_checkpoint(
        resume_path, model=resumed, spec=candidate_spec.lane_spec(),
        stream_a=resumed_a, stream_b=resumed_b,
    )
    _one_update(resumed, resumed_a, resumed_b, 1, total_steps)
    resume_components = {
        "model_and_optimizer": nested_equal(continuous_state, comparable_state(resumed)),
        "rng": nested_equal(continuous_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(continuous_streams[0], _stream_state(resumed_a)),
        "data_stream_b": nested_equal(continuous_streams[1], _stream_state(resumed_b)),
    }
    resume_exact = all(resume_components.values())
    _release(resumed)

    constraints_safe = (
        float(last_diagnostics.get("native_metric_alignment", -1.0)) >= -1e-5
        and float(last_diagnostics.get("bridge_metric_alignment", -1.0)) >= -1e-5
    )
    passed = all((endpoint_identity, state_key_identity, zero_intervention_exact, resume_exact, constraints_safe))
    result = {
        "schema": "clean-unsb-search005-generation1-gate-v1",
        "candidate_id": candidate_spec.candidate_id,
        "passed": passed,
        "endpoint_forward_byte_identical": endpoint_identity,
        "network_state_keys_identical": state_key_identity,
        "zero_intervention_full_update_exact": zero_intervention_exact,
        "zero_intervention_components": zero_components,
        "full_state_resume_exact": resume_exact,
        "resume_components": resume_components,
        "projection_constraints_safe": constraints_safe,
        "sample_operator_diagnostics": last_diagnostics,
        "paired_target_readable_by_candidate": False,
        "confirmation20_opened": False,
    }
    if not passed:
        raise RuntimeError(f"ACMP engineering gate failed: {result}")
    return result


def run_fbcmp_invariants(**kwargs) -> dict:
    return run_acmp_invariants(candidate_spec=FBCMP, **kwargs)


def run_bcavp_invariants(
    *,
    rows: list[dict],
    train_view: Path,
    output: Path,
    seed: int,
    gpu: int,
) -> dict:
    """Gate BCAVP's physical projection, RNG isolation and exact resume."""
    _, runtime, _ = modules()
    from .model_operators import bcavp_installation_status, set_bcavp_active

    output = Path(output)
    gate_dir = output / "generation1_gate"
    total_steps = 8
    per_domain = 25
    e0 = create_e0(
        gate_dir / "e0.pt", rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    plain_keys = {
        name: tuple(runtime.inner(getattr(plain, "net" + name)).state_dict())
        for name in plain.model_names
    }
    plain.set_input(plain_a.next(), plain_b.next())
    plain.forward()
    plain_forward_rng = _rng_state(runtime)
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        BCAVP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    state_key_identity = plain_keys == {
        name: tuple(runtime.inner(getattr(candidate, "net" + name)).state_dict())
        for name in candidate.model_names
    }
    candidate.set_input(candidate_a.next(), candidate_b.next())
    candidate.forward()
    candidate_forward_rng = _rng_state(runtime)
    generator = runtime.inner(candidate.netG)
    forward_diag = copy.deepcopy(generator._search005_bcavp_generator_last)
    cap_enforced = (
        bool(forward_diag)
        and float(forward_diag["maximum_projected_ratio"]) <= 1.0 + 1e-5
    )
    forward_rng_preserved = nested_equal(plain_forward_rng, candidate_forward_rng)
    status = bcavp_installation_status(candidate)
    _release(candidate)

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(plain, plain_a, plain_b, 0, total_steps)
    plain_state = comparable_state(plain)
    plain_rng = _rng_state(runtime)
    plain_streams = (_stream_state(plain_a), _stream_state(plain_b))
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        BCAVP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    set_bcavp_active(candidate, False)
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    zero_components = {
        "model_and_optimizer": nested_equal(plain_state, comparable_state(candidate)),
        "rng": nested_equal(plain_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(plain_streams[0], _stream_state(candidate_a)),
        "data_stream_b": nested_equal(plain_streams[1], _stream_state(candidate_b)),
    }
    zero_intervention_exact = all(zero_components.values())
    _release(candidate)

    continuous, continuous_a, continuous_b = prepare_lane(
        BCAVP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(continuous, continuous_a, continuous_b, 0, total_steps)
    _one_update(continuous, continuous_a, continuous_b, 1, total_steps)
    continuous_state = runtime.model_state(continuous)
    continuous_rng = _rng_state(runtime)
    continuous_streams = (_stream_state(continuous_a), _stream_state(continuous_b))
    _release(continuous)

    split, split_a, split_b = prepare_lane(
        BCAVP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(split, split_a, split_b, 0, total_steps)
    resume_path = gate_dir / "bcavp_resume_step1.pt"
    runtime.save_checkpoint(
        resume_path, model=split, spec=BCAVP.lane_spec(), step=1,
        target_steps=total_steps, stream_a=split_a, stream_b=split_b,
        metadata={"gate": "full_state_resume"},
    )
    _release(split)

    resumed, resumed_a, resumed_b = prepare_lane(
        BCAVP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    payload = runtime.load_checkpoint(
        resume_path, model=resumed, spec=BCAVP.lane_spec(),
        stream_a=resumed_a, stream_b=resumed_b,
    )
    if int(payload["step"]) != 1:
        raise RuntimeError("BCAVP resume checkpoint step mismatch")
    _one_update(resumed, resumed_a, resumed_b, 1, total_steps)
    resume_components = {
        "full_model_optimizer_and_operator": nested_equal(
            continuous_state, runtime.model_state(resumed)
        ),
        "rng": nested_equal(continuous_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(continuous_streams[0], _stream_state(resumed_a)),
        "data_stream_b": nested_equal(continuous_streams[1], _stream_state(resumed_b)),
    }
    resume_exact = all(resume_components.values())
    _release(resumed)

    passed = all((
        state_key_identity, cap_enforced, forward_rng_preserved,
        zero_intervention_exact, resume_exact,
    ))
    result = {
        "schema": "clean-unsb-search005-generation1-gate-v1",
        "candidate_id": BCAVP.candidate_id,
        "passed": passed,
        "network_state_keys_identical": state_key_identity,
        "physical_variance_cap_enforced": cap_enforced,
        "forward_rng_bundle_preserved": forward_rng_preserved,
        "forward_diagnostics": forward_diag,
        "zero_intervention_full_update_exact": zero_intervention_exact,
        "zero_intervention_components": zero_components,
        "full_state_resume_exact": resume_exact,
        "resume_components": resume_components,
        "installation_status": status,
        "paired_target_readable_by_candidate": False,
        "confirmation20_opened": False,
    }
    if not passed:
        raise RuntimeError(f"BCAVP engineering gate failed: {result}")
    return result


def run_phcrp_invariants(
    *,
    rows: list[dict],
    train_view: Path,
    output: Path,
    seed: int,
    gpu: int,
) -> dict:
    """Gate PHCRP path contexts, exact self-null and disk resume."""
    _, runtime, _ = modules()
    from .model_operators import phcrp_installation_status, set_phcrp_active

    gate_dir = Path(output) / "generation1_gate"
    total_steps = 8
    per_domain = 25
    e0 = create_e0(
        gate_dir / "e0.pt", rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    plain_keys = {
        name: tuple(runtime.inner(getattr(plain, "net" + name)).state_dict())
        for name in plain.model_names
    }
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        PHCRP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    state_key_identity = plain_keys == {
        name: tuple(runtime.inner(getattr(candidate, "net" + name)).state_dict())
        for name in candidate.model_names
    }
    candidate.set_input(candidate_a.next(), candidate_b.next())
    candidate.forward()
    contexts = candidate._search005_phcrp_context
    histories_complete = all(history is not None for history in contexts["histories"])
    forward_diag = copy.deepcopy(candidate._search005_phcrp_last)
    cap_enforced = (
        bool(forward_diag)
        and float(forward_diag["maximum_cap_ratio"]) <= 1.0 + 1e-5
    )
    status = phcrp_installation_status(candidate)
    _release(candidate)

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(plain, plain_a, plain_b, 0, total_steps)
    plain_state = comparable_state(plain)
    plain_rng = _rng_state(runtime)
    plain_streams = (_stream_state(plain_a), _stream_state(plain_b))
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        PHCRP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    set_phcrp_active(candidate, False)
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    zero_components = {
        "model_and_optimizer": nested_equal(plain_state, comparable_state(candidate)),
        "rng": nested_equal(plain_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(plain_streams[0], _stream_state(candidate_a)),
        "data_stream_b": nested_equal(plain_streams[1], _stream_state(candidate_b)),
    }
    zero_intervention_exact = all(zero_components.values())
    _release(candidate)

    continuous, continuous_a, continuous_b = prepare_lane(
        PHCRP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(continuous, continuous_a, continuous_b, 0, total_steps)
    _one_update(continuous, continuous_a, continuous_b, 1, total_steps)
    continuous_state = runtime.model_state(continuous)
    continuous_rng = _rng_state(runtime)
    continuous_streams = (_stream_state(continuous_a), _stream_state(continuous_b))
    _release(continuous)

    split, split_a, split_b = prepare_lane(
        PHCRP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(split, split_a, split_b, 0, total_steps)
    resume_path = gate_dir / "phcrp_resume_step1.pt"
    runtime.save_checkpoint(
        resume_path, model=split, spec=PHCRP.lane_spec(), step=1,
        target_steps=total_steps, stream_a=split_a, stream_b=split_b,
        metadata={"gate": "full_state_resume"},
    )
    _release(split)

    resumed, resumed_a, resumed_b = prepare_lane(
        PHCRP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    payload = runtime.load_checkpoint(
        resume_path, model=resumed, spec=PHCRP.lane_spec(),
        stream_a=resumed_a, stream_b=resumed_b,
    )
    if int(payload["step"]) != 1:
        raise RuntimeError("PHCRP resume checkpoint step mismatch")
    _one_update(resumed, resumed_a, resumed_b, 1, total_steps)
    resume_components = {
        "full_model_optimizer_and_operator": nested_equal(
            continuous_state, runtime.model_state(resumed)
        ),
        "rng": nested_equal(continuous_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(continuous_streams[0], _stream_state(resumed_a)),
        "data_stream_b": nested_equal(continuous_streams[1], _stream_state(resumed_b)),
    }
    resume_exact = all(resume_components.values())
    _release(resumed)

    passed = all((
        state_key_identity, histories_complete, cap_enforced,
        zero_intervention_exact, resume_exact,
    ))
    result = {
        "schema": "clean-unsb-search005-generation1-gate-v1",
        "candidate_id": PHCRP.candidate_id,
        "passed": passed,
        "network_state_keys_identical": state_key_identity,
        "three_training_rollout_histories_complete": histories_complete,
        "pathwise_physical_cap_enforced": cap_enforced,
        "forward_diagnostics": forward_diag,
        "zero_intervention_full_update_exact": zero_intervention_exact,
        "zero_intervention_components": zero_components,
        "full_state_resume_exact": resume_exact,
        "resume_components": resume_components,
        "installation_status": status,
        "paired_target_readable_by_candidate": False,
        "confirmation20_opened": False,
    }
    if not passed:
        raise RuntimeError(f"PHCRP engineering gate failed: {result}")
    return result


def run_phrsup_invariants(
    *,
    rows: list[dict],
    train_view: Path,
    output: Path,
    seed: int,
    gpu: int,
) -> dict:
    """Gate PHRSUP endpoint identity, halfspace safety and exact resume."""
    _, runtime, _ = modules()
    from .model_operators import phrsup_installation_status, set_phrsup_active

    gate_dir = Path(output) / "generation1_gate"
    total_steps = 8
    per_domain = 25
    e0 = create_e0(
        gate_dir / "e0.pt", rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    plain_keys = {
        name: tuple(runtime.inner(getattr(plain, "net" + name)).state_dict())
        for name in plain.model_names
    }
    plain.set_input(plain_a.next(), plain_b.next())
    plain.forward()
    plain_endpoints = (plain.fake_B.detach().cpu(), plain.fake_B2.detach().cpu())
    plain_forward_rng = _rng_state(runtime)
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        PHRSUP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    state_key_identity = plain_keys == {
        name: tuple(runtime.inner(getattr(candidate, "net" + name)).state_dict())
        for name in candidate.model_names
    }
    candidate.set_input(candidate_a.next(), candidate_b.next())
    candidate.forward()
    endpoint_identity = (
        torch.equal(plain_endpoints[0], candidate.fake_B.detach().cpu())
        and torch.equal(plain_endpoints[1], candidate.fake_B2.detach().cpu())
    )
    forward_rng_identity = nested_equal(plain_forward_rng, _rng_state(runtime))
    histories_complete = all(
        history is not None
        for history in candidate._search005_phrsup_context["histories"]
    )
    status = phrsup_installation_status(candidate)
    _release(candidate)

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(plain, plain_a, plain_b, 0, total_steps)
    plain_state = comparable_state(plain)
    plain_rng = _rng_state(runtime)
    plain_streams = (_stream_state(plain_a), _stream_state(plain_b))
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        PHRSUP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    set_phrsup_active(candidate, False)
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    zero_components = {
        "model_and_optimizer": nested_equal(plain_state, comparable_state(candidate)),
        "rng": nested_equal(plain_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(plain_streams[0], _stream_state(candidate_a)),
        "data_stream_b": nested_equal(plain_streams[1], _stream_state(candidate_b)),
    }
    zero_intervention_exact = all(zero_components.values())
    _release(candidate)

    continuous, continuous_a, continuous_b = prepare_lane(
        PHRSUP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(continuous, continuous_a, continuous_b, 0, total_steps)
    _one_update(continuous, continuous_a, continuous_b, 1, total_steps)
    continuous_state = runtime.model_state(continuous)
    continuous_rng = _rng_state(runtime)
    continuous_streams = (_stream_state(continuous_a), _stream_state(continuous_b))
    sample_diag = copy.deepcopy(continuous._search005_phrsup_last)
    _release(continuous)

    split, split_a, split_b = prepare_lane(
        PHRSUP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(split, split_a, split_b, 0, total_steps)
    resume_path = gate_dir / "phrsup_resume_step1.pt"
    runtime.save_checkpoint(
        resume_path, model=split, spec=PHRSUP.lane_spec(), step=1,
        target_steps=total_steps, stream_a=split_a, stream_b=split_b,
        metadata={"gate": "full_state_resume"},
    )
    _release(split)

    resumed, resumed_a, resumed_b = prepare_lane(
        PHRSUP, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    payload = runtime.load_checkpoint(
        resume_path, model=resumed, spec=PHRSUP.lane_spec(),
        stream_a=resumed_a, stream_b=resumed_b,
    )
    if int(payload["step"]) != 1:
        raise RuntimeError("PHRSUP resume checkpoint step mismatch")
    _one_update(resumed, resumed_a, resumed_b, 1, total_steps)
    resume_components = {
        "full_model_optimizer_and_operator": nested_equal(
            continuous_state, runtime.model_state(resumed)
        ),
        "rng": nested_equal(continuous_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(continuous_streams[0], _stream_state(resumed_a)),
        "data_stream_b": nested_equal(continuous_streams[1], _stream_state(resumed_b)),
    }
    resume_exact = all(resume_components.values())
    _release(resumed)

    projection_safe = (
        float(sample_diag.get("projected_defect_alignment", -1.0)) >= -1e-5
        and float(sample_diag.get("native_descent_alignment", -1.0)) >= -1e-5
    )
    passed = all((
        state_key_identity, endpoint_identity, forward_rng_identity,
        histories_complete, zero_intervention_exact, resume_exact, projection_safe,
    ))
    result = {
        "schema": "clean-unsb-search005-generation2-gate-v1",
        "candidate_id": PHRSUP.candidate_id,
        "passed": passed,
        "network_state_keys_identical": state_key_identity,
        "endpoint_forward_byte_identical": endpoint_identity,
        "forward_rng_bundle_identical": forward_rng_identity,
        "three_training_rollout_histories_complete": histories_complete,
        "zero_intervention_full_update_exact": zero_intervention_exact,
        "zero_intervention_components": zero_components,
        "full_state_resume_exact": resume_exact,
        "resume_components": resume_components,
        "projection_constraints_safe": projection_safe,
        "sample_operator_diagnostics": sample_diag,
        "installation_status": status,
        "paired_target_readable_by_candidate": False,
        "confirmation20_opened": False,
    }
    if not passed:
        raise RuntimeError(f"PHRSUP engineering gate failed: {result}")
    return result


def _run_optimistic_invariants(
    *,
    rows: list[dict],
    train_view: Path,
    output: Path,
    seed: int,
    gpu: int,
    spec,
    tag: str,
    status_function,
    toggle_function,
    generation: str,
) -> dict:
    """Gate optimistic-game first-step identity, state and exact disk resume."""
    _, runtime, _ = modules()

    gate_dir = Path(output) / f"{generation}_gate"
    total_steps = 8
    per_domain = 25
    e0 = create_e0(
        gate_dir / "e0.pt", rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    plain.set_input(plain_a.next(), plain_b.next())
    plain.forward()
    plain_endpoints = (plain.fake_B.detach().cpu(), plain.fake_B2.detach().cpu())
    plain_keys = {
        name: tuple(runtime.inner(getattr(plain, "net" + name)).state_dict())
        for name in plain.model_names
    }
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    candidate.set_input(candidate_a.next(), candidate_b.next())
    candidate.forward()
    endpoint_identity = (
        torch.equal(plain_endpoints[0], candidate.fake_B.detach().cpu())
        and torch.equal(plain_endpoints[1], candidate.fake_B2.detach().cpu())
    )
    state_key_identity = plain_keys == {
        name: tuple(runtime.inner(getattr(candidate, "net" + name)).state_dict())
        for name in candidate.model_names
    }
    status = status_function(candidate)
    _release(candidate)

    plain, plain_a, plain_b = prepare_lane(
        PLAIN, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(plain, plain_a, plain_b, 0, total_steps)
    plain_state = comparable_state(plain)
    plain_rng = _rng_state(runtime)
    plain_streams = (_stream_state(plain_a), _stream_state(plain_b))
    _release(plain)

    candidate, candidate_a, candidate_b = prepare_lane(
        spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    first_step_components = {
        "model_and_optimizer": nested_equal(plain_state, comparable_state(candidate)),
        "rng": nested_equal(plain_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(plain_streams[0], _stream_state(candidate_a)),
        "data_stream_b": nested_equal(plain_streams[1], _stream_state(candidate_b)),
        "all_player_predictability_zero": all(
            float(row["predictability"]) == 0.0
            for row in getattr(candidate, f"_search005_{tag}_last").values()
        ),
    }
    first_step_exact = all(first_step_components.values())
    _release(candidate)

    candidate, candidate_a, candidate_b = prepare_lane(
        spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    toggle_function(candidate, False)
    _one_update(candidate, candidate_a, candidate_b, 0, total_steps)
    zero_components = {
        "model_and_optimizer": nested_equal(plain_state, comparable_state(candidate)),
        "rng": nested_equal(plain_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(plain_streams[0], _stream_state(candidate_a)),
        "data_stream_b": nested_equal(plain_streams[1], _stream_state(candidate_b)),
    }
    zero_intervention_exact = all(zero_components.values())
    _release(candidate)

    continuous, continuous_a, continuous_b = prepare_lane(
        spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(continuous, continuous_a, continuous_b, 0, total_steps)
    _one_update(continuous, continuous_a, continuous_b, 1, total_steps)
    continuous_state = runtime.model_state(continuous)
    continuous_rng = _rng_state(runtime)
    continuous_streams = (_stream_state(continuous_a), _stream_state(continuous_b))
    sample_diag = copy.deepcopy(getattr(continuous, f"_search005_{tag}_last"))
    _release(continuous)

    split, split_a, split_b = prepare_lane(
        spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    _one_update(split, split_a, split_b, 0, total_steps)
    resume_path = gate_dir / f"{tag}_resume_step1.pt"
    runtime.save_checkpoint(
        resume_path, model=split, spec=spec.lane_spec(), step=1,
        target_steps=total_steps, stream_a=split_a, stream_b=split_b,
        metadata={"gate": "full_state_resume"},
    )
    _release(split)

    resumed, resumed_a, resumed_b = prepare_lane(
        spec, e0=e0, rows=rows, train_view=train_view,
        option_dir=gate_dir / "options", per_domain=per_domain,
        total_steps=total_steps, seed=seed, gpu=gpu,
    )
    payload = runtime.load_checkpoint(
        resume_path, model=resumed, spec=spec.lane_spec(),
        stream_a=resumed_a, stream_b=resumed_b,
    )
    if int(payload["step"]) != 1:
        raise RuntimeError(f"{spec.candidate_id} resume checkpoint step mismatch")
    _one_update(resumed, resumed_a, resumed_b, 1, total_steps)
    resume_components = {
        "full_model_optimizer_and_operator": nested_equal(
            continuous_state, runtime.model_state(resumed)
        ),
        "rng": nested_equal(continuous_rng, _rng_state(runtime)),
        "data_stream_a": nested_equal(continuous_streams[0], _stream_state(resumed_a)),
        "data_stream_b": nested_equal(continuous_streams[1], _stream_state(resumed_b)),
    }
    resume_exact = all(resume_components.values())
    _release(resumed)

    predictability_valid = all(
        0.0 <= float(row["predictability"]) <= 1.0
        for row in sample_diag.values()
    )
    passed = all((
        endpoint_identity, state_key_identity, first_step_exact,
        zero_intervention_exact, resume_exact, predictability_valid,
    ))
    result = {
        "schema": f"clean-unsb-search005-{generation}-gate-v1",
        "candidate_id": spec.candidate_id,
        "passed": passed,
        "endpoint_forward_byte_identical": endpoint_identity,
        "network_state_keys_identical": state_key_identity,
        "first_active_update_exactly_plain": first_step_exact,
        "first_step_components": first_step_components,
        "zero_intervention_full_update_exact": zero_intervention_exact,
        "zero_intervention_components": zero_components,
        "full_state_resume_exact": resume_exact,
        "resume_components": resume_components,
        "predictability_coefficients_valid": predictability_valid,
        "sample_operator_diagnostics": sample_diag,
        "installation_status": status,
        "paired_target_readable_by_candidate": False,
        "confirmation20_opened": False,
    }
    if not passed:
        raise RuntimeError(f"{spec.candidate_id} engineering gate failed: {result}")
    return result


def run_pcoa_invariants(**kwargs) -> dict:
    from .model_operators import pcoa_installation_status, set_pcoa_active

    return _run_optimistic_invariants(
        spec=PCOA,
        tag="pcoa",
        status_function=pcoa_installation_status,
        toggle_function=set_pcoa_active,
        generation="generation1",
        **kwargs,
    )


def run_npooa_invariants(**kwargs) -> dict:
    from .model_operators import npooa_installation_status, set_npooa_active

    result = _run_optimistic_invariants(
        spec=NPOOA,
        tag="npooa",
        status_function=npooa_installation_status,
        toggle_function=set_npooa_active,
        generation="generation2",
        **kwargs,
    )
    norm_preserved = all(
        abs(float(row["norm_ratio"]) - 1.0) <= 2e-6
        for row in result["sample_operator_diagnostics"].values()
    )
    result["native_update_norm_preserved"] = norm_preserved
    result["passed"] = bool(result["passed"] and norm_preserved)
    if not result["passed"]:
        raise RuntimeError(f"{NPOOA.candidate_id} norm-preservation gate failed: {result}")
    return result
