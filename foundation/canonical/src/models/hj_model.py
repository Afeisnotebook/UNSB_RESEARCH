"""Thin model registration so ``--model hj`` resolves to the clean HJ implementation."""

import os
import sys

_FOUNDATION_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _FOUNDATION_ROOT not in sys.path:
    sys.path.insert(0, _FOUNDATION_ROOT)

_HJ_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "research",
        "candidates", "CAND-002-hj-patchnce",
    )
)
if _HJ_ROOT not in sys.path:
    sys.path.insert(0, _HJ_ROOT)

from hj.model import SBModelHJPatchNCE  # noqa: E402


class HjModel(SBModelHJPatchNCE):
    pass
