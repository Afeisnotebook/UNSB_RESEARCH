"""HNEK-UNSB shim (read-only, bridge-native coordinate change)."""

from .hnek_adapter import (
    install_hnek_generator,
    install_hnek_model,
    uninstall_hnek_generator,
)
from .hnek_kernel import (
    bridge_schedule,
    endpoint_from_residual,
    horizon_from_condition,
    normalized_residual,
    physical_time_from_condition,
)

__all__ = [
    "bridge_schedule",
    "endpoint_from_residual",
    "horizon_from_condition",
    "normalized_residual",
    "physical_time_from_condition",
    "install_hnek_generator",
    "install_hnek_model",
    "uninstall_hnek_generator",
]
