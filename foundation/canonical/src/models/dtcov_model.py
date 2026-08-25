"""Thin model registration so ``--model dtcov`` resolves to the clean DT implementation."""

import os
import sys

_FOUNDATION_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _FOUNDATION_ROOT not in sys.path:
    sys.path.insert(0, _FOUNDATION_ROOT)

_DTCOV_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "research",
        "candidates", "CAND-001-dt-covmatch",
    )
)
if _DTCOV_ROOT not in sys.path:
    sys.path.insert(0, _DTCOV_ROOT)

from dtcov.model import SBModelDTCovMatch  # noqa: E402


class DtcovModel(SBModelDTCovMatch):
    pass
