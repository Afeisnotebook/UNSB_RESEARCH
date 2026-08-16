"""Checkpoint identity audit: net tuple + full training state."""

from __future__ import annotations

import hashlib
from pathlib import Path


NET_PARTS = ("net_G", "net_F", "net_D", "net_E")
STATE_PART = "training_state"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epochs_with_prefix(ckpt_dir: Path, name: str) -> set[int]:
    root = ckpt_dir / name
    if not root.is_dir():
        return set()
    epochs = set()
    for path in root.iterdir():
        if not path.is_file():
            continue
        stem = path.name
        for part in NET_PARTS + (STATE_PART,):
            if stem.startswith("_") and stem.endswith(f"_{part}.pth"):
                continue
            if stem.endswith(f"_{part}.pth"):
                prefix = stem[: -len(f"_{part}.pth")]
                if prefix.isdigit():
                    epochs.add(int(prefix))
    return epochs


def audit_checkpoint(ckpt_dir: Path, name: str, *, hash_files: bool = False) -> dict:
    """Audit which epochs have a complete net tuple and/or full training state."""
    ckpt_dir = Path(ckpt_dir)
    root = ckpt_dir / name
    if not root.is_dir():
        return {"name": name, "exists": False}

    epochs = _epochs_with_prefix(ckpt_dir, name)
    net_complete = []
    state_complete = []
    artifacts = []
    for epoch in sorted(epochs):
        net_files = [root / f"{epoch}_{part}.pth" for part in NET_PARTS]
        if all(p.is_file() for p in net_files):
            net_complete.append(epoch)
            entry = {"epoch": epoch, "kind": "net_tuple", "files": {}}
            for part, p in zip(NET_PARTS, net_files):
                entry["files"][part] = {
                    "size": p.stat().st_size,
                    "sha256": _sha256(p) if hash_files else None,
                }
            artifacts.append(entry)
        state_file = root / f"{epoch}_{STATE_PART}.pth"
        if state_file.is_file():
            state_complete.append(epoch)
            artifacts.append(
                {
                    "epoch": epoch,
                    "kind": "training_state",
                    "files": {
                        STATE_PART: {
                            "size": state_file.stat().st_size,
                            "sha256": _sha256(state_file) if hash_files else None,
                        }
                    },
                }
            )

    return {
        "name": name,
        "exists": True,
        "epochs_net_complete": net_complete,
        "epochs_state_complete": state_complete,
        "artifacts": artifacts,
    }
