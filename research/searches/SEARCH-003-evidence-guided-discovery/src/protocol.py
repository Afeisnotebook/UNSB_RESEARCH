"""Frozen identities and decision rules for SEARCH-003.

The algorithms are deliberately *not* frozen here.  SEARCH-003 freezes the
evidence contract that is allowed to create an algorithm: probe identities,
counterfactual horizons, target-blind signal gates and promotion rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


DOMAINS = (
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RSCityscapes",
    "RainCityscapes",
    "RainDS-syn",
    "SnowTrafficData",
)


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    model: str = "sb"
    mechanisms: tuple[str, ...] = ()
    family: str = "probe"
    failure_object: str = "unknown"
    trainable_candidate: bool = False

    def to_dict(self) -> dict:
        value = asdict(self)
        value["mechanisms"] = list(self.mechanisms)
        return value


def historical_probes() -> tuple[ProbeSpec, ...]:
    return (
        ProbeSpec("plain", family="baseline", failure_object="none"),
        ProbeSpec("dt", model="dtcov", failure_object="anchored_functional_bias"),
        ProbeSpec("hj", model="hj", failure_object="gradient_component_reversal"),
        ProbeSpec("hnek", model="hnek_search", failure_object="bridge_coordinate_conditioning"),
        ProbeSpec("lbst", mechanisms=("lbst",), failure_object="rollout_distribution_velocity"),
        ProbeSpec("ptq", mechanisms=("ptq",), failure_object="time_sampling_measure"),
        ProbeSpec("dcum", mechanisms=("dcum",), failure_object="domain_sampling_measure"),
        ProbeSpec("aeb", mechanisms=("aeb",), failure_object="latent_endpoint_variance"),
        ProbeSpec("lttr", model="lttr", failure_object="anchored_output_bias"),
        ProbeSpec("ta_kck", family="negative_control", failure_object="historical_path_constraint"),
    )


@dataclass(frozen=True)
class Search003Protocol:
    schema: str = "clean-unsb-search003-protocol-v1"
    source_commit: str = "649b7ec2bd520ccb174b73c5b5187f7ce08ebb22"
    manifest_sha256: str = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
    seed: int = 2026
    counterfactual_horizons: tuple[int, ...] = (1, 8, 32)
    decisive_horizon: int = 200
    signal_balanced_accuracy_min: float = 0.65
    signal_spearman_min: float = 0.30
    signal_domain_agreement_min: int = 4
    max_generation1_candidates: int = 6
    max_generations_per_mechanism: int = 2
    small_steps: int = 2400
    small_eval_steps: tuple[int, ...] = (400, 800, 1200, 1600, 2000, 2400)
    full_steps: int = 6000
    full_eval_steps: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000, 6000)
    extension_steps: int = 12000
    extension_eval_steps: tuple[int, ...] = (8000, 10000, 12000)
    confirmation20_opened: bool = False
    probes: tuple[ProbeSpec, ...] = field(default_factory=historical_probes)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["probes"] = [probe.to_dict() for probe in self.probes]
        return value


def promotion_decision(trajectory: list[dict]) -> dict:
    """Apply the preregistered 1600/2000/2400 small-view gate."""
    by_step = {int(row["step"]): row for row in trajectory}
    required = (1600, 2000, 2400)
    if any(step not in by_step for step in required):
        return {"promote": False, "reasons": ["incomplete_late_trajectory"]}
    late = [by_step[step] for step in required]
    deltas = [float(row["macro_psnr_delta"]) for row in late]
    candidate_psnr = [float(row["macro_psnr"]) for row in trajectory]
    rolling = [
        sum(candidate_psnr[index - 2 : index + 1]) / 3.0
        for index in range(2, len(candidate_psnr))
    ]
    late_rolling = sum(candidate_psnr[-3:]) / 3.0
    rollback = (max(rolling) - late_rolling) if rolling else 0.0
    conditions = {
        "late_mean_positive": sum(deltas) / len(deltas) > 0.0,
        "final_positive": deltas[-1] > 0.0,
        "coverage": sum(int(row["positive_domains"]) >= 4 for row in late) >= 2,
        "worst_domain": min(float(row["worst_domain_delta"]) for row in late) > -1.0,
        "absolute_retention": rollback <= 0.3,
    }
    return {
        "promote": all(conditions.values()),
        "conditions": conditions,
        "late_mean_delta": sum(deltas) / len(deltas),
        "candidate_absolute_rollback": rollback,
        "reasons": [name for name, passed in conditions.items() if not passed],
    }
