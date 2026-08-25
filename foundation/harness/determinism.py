"""Determinism helpers: seeds, RNG accounting and auxiliary-RNG isolation."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed python, numpy and torch (if present) with a fixed integer."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def sub_seed(*parts) -> int:
    """Derive a reproducible 63-bit seed from a canonical tuple, without collisions."""
    text = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


@contextmanager
def rng_scope():
    """Save/restore numpy and torch RNG state around auxiliary random work."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = None
    cuda_states = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_state is not None:
            import torch

            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
