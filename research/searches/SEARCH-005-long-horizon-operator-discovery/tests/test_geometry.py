from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search005_geometry", ROOT / "src" / "geometry.py"
)
geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


def model(value: float) -> dict:
    return {
        "networks": {
            "G": {"w": torch.tensor([value, 0.0])},
            "F": {"w": torch.tensor([0.0])},
        }
    }


def test_propagation_geometry_tracks_retention_and_rotation():
    base = model(0.0)
    pulse = model(1.0)
    initial = geometry.initial_network_delta(base, pulse)
    later = model(-2.0)
    result = geometry.model_gap_geometry(base, later, initial)
    assert result["global"]["retention_ratio"] == pytest.approx(2.0)
    assert result["global"]["initial_direction_cosine"] == pytest.approx(-1.0)
