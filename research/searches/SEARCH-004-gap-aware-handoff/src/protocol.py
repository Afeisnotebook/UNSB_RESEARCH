"""Frozen protocol and preregistered decision rules for SEARCH-004."""

from __future__ import annotations

from dataclasses import asdict, dataclass


DOMAINS = (
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RSCityscapes",
    "RainCityscapes",
    "RainDS-syn",
    "SnowTrafficData",
)

FORBIDDEN_KEYS = {
    "target", "paired_target", "psnr", "ssim", "lpips", "confirmation",
}


def assert_target_blind(fields: dict | None) -> None:
    for key in (fields or {}):
        normalized = str(key).strip().lower()
        if normalized in FORBIDDEN_KEYS or normalized.startswith("paired_"):
            raise ValueError(f"target-aware field is forbidden: {key}")


@dataclass(frozen=True)
class Search004Protocol:
    schema: str = "clean-unsb-search004-protocol-v1"
    repository_anchor: str = "3674a390e9fde997ec1261660c2a96f2a7d49aa6"
    canonical_anchor: str = "649b7ec2bd520ccb174b73c5b5187f7ce08ebb22"
    manifest_sha256: str = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
    seed: int = 2026
    small_per_domain: int = 25
    local_horizons: tuple[int, ...] = (1, 8, 32, 200)
    extension_horizon: int = 800
    equilibration_steps: int = 32
    shadow_steps: int = 32
    moment_projection_batches: int = 4
    lift_min: float = 0.15
    auc_lift_min: float = 0.10
    domain_nonnegative_min: int = 4
    worst_domain_lift_min: float = -0.5
    defect_reduction_min: float = 0.25
    small_steps: int = 3200
    small_eval_steps: tuple[int, ...] = (400, 800, 1200, 1600, 2000, 2400, 2800, 3200)
    full_steps: int = 6000
    full_eval_steps: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000, 6000)
    extension_steps: int = 12000
    extension_eval_steps: tuple[int, ...] = (8000, 10000, 12000)
    max_generation1_candidates: int = 4
    max_e0_candidates: int = 3
    max_revisions_per_mechanism: int = 1
    confidence_alpha: float = 0.05
    confidence_min_observations: int = 8
    disk_cap_gib: float = 40.0
    confirmation20_opened: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


ARMS = (
    "P_common_plain",
    "U_uninterrupted",
    "A_hard_disable",
    "B_gf_zero_moment",
    "C_local_native_moment",
    "D0_hold_only",
    "D_costate_equilibration",
    "E_combined",
    "F_g_only_transplant",
    "H_native_moment_projection",
    "K_gf_state_transplant",
)
