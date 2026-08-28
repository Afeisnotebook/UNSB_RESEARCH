"""Pre-registered target-blind compatibility statistics for SEARCH-004."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceSequence:
    n: int
    mean: float
    lower: float
    upper: float
    radius: float
    alpha: float
    valid: bool


def empirical_bernstein_cs(
    samples: list[float] | tuple[float, ...],
    *,
    alpha: float = 0.05,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
    minimum: int = 8,
) -> ConfidenceSequence:
    """A union-bound anytime-valid empirical-Bernstein sequence.

    The controller inputs are cosine-like and therefore bounded.  At time n we
    allocate alpha/(n(n+1)); summing over n is at most alpha.  This is more
    conservative than a tuned mixture boundary but has an explicit validity
    argument and no learned constants.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not lower_bound < upper_bound:
        raise ValueError("invalid sample bounds")
    values = [float(value) for value in samples]
    if any(value < lower_bound or value > upper_bound for value in values):
        raise ValueError("sample outside declared confidence-sequence bounds")
    n = len(values)
    if n == 0:
        return ConfidenceSequence(0, 0.0, lower_bound, upper_bound, upper_bound - lower_bound, alpha, False)
    mean = sum(values) / n
    variance = (
        sum((value - mean) ** 2 for value in values) / (n - 1)
        if n > 1 else 0.0
    )
    width = upper_bound - lower_bound
    alpha_n = alpha / (n * (n + 1.0))
    log_term = math.log(3.0 / alpha_n)
    radius = math.sqrt(2.0 * variance * log_term / n) + 3.0 * width * log_term / n
    return ConfidenceSequence(
        n=n,
        mean=mean,
        lower=max(lower_bound, mean - radius),
        upper=min(upper_bound, mean + radius),
        radius=radius,
        alpha=alpha,
        valid=n >= int(minimum),
    )


def persistently_incompatible(samples: list[float] | tuple[float, ...]) -> bool:
    value = empirical_bernstein_cs(samples)
    return value.valid and value.upper < 0.0
