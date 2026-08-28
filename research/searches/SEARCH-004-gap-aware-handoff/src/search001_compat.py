"""Load the accepted SEARCH-001 runtime without copying canonical code."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SEARCH001_ROOT = REPO_ROOT / "research" / "searches" / "SEARCH-001-clean-directional"
PACKAGE = "_clean_unsb_search001_search004"


def _load(name: str):
    qualified = f"{PACKAGE}.{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    path = SEARCH001_ROOT / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def modules():
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SEARCH001_ROOT / "src")]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    return _load("protocol"), _load("runtime"), _load("evaluate")

