"""Load SEARCH-005 operator installers without copying their implementation."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SEARCH005_ROOT = REPO_ROOT / "research" / "searches" / "SEARCH-005-long-horizon-operator-discovery"
PACKAGE = "_clean_unsb_search005_search004"


def load_model_operators():
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SEARCH005_ROOT / "src")]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    qualified = f"{PACKAGE}.model_operators"
    if qualified in sys.modules:
        return sys.modules[qualified]
    path = SEARCH005_ROOT / "src" / "model_operators.py"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module
