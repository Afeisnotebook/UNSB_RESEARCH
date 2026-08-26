"""Thin registration for the deterministic LTTR research candidate."""

from __future__ import annotations

import os
import sys

_LTTR_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "research",
        "candidates", "CAND-005-lttr",
    )
)
if _LTTR_ROOT not in sys.path:
    sys.path.insert(0, _LTTR_ROOT)

from lttr.model import SBModelLTTR  # noqa: E402


class LttrModel(SBModelLTTR):
    pass
