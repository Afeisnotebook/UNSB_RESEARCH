"""Load the accepted SEARCH-003 selector implementation without copying it."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SEARCH003_ROOT = REPO_ROOT / "research" / "searches" / "SEARCH-003-evidence-guided-discovery"
PACKAGE = "_clean_unsb_search003_search004"


def load_receding():
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SEARCH003_ROOT / "src")]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    qualified = f"{PACKAGE}.receding"
    if qualified in sys.modules:
        return sys.modules[qualified]
    path = SEARCH003_ROOT / "src" / "receding.py"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module
