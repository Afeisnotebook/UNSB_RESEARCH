#!/usr/bin/env python3
"""Build five-domain dataroots (hardlinks) from DATA_MANIFEST.json.

Creates:
  datasets/single/<domain>/trainA  (100 inputs, original stems)
  datasets/single/<domain>/trainB  (100 targets, original stems)
  datasets/aio/trainA              (5x100 inputs, <domain>__<stem>.<ext>)
  datasets/aio/trainB              (5x100 targets, <domain>__<stem>.<ext>)

Uses hardlinks; falls back to symlinks if hardlink fails.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "DATA_MANIFEST.json"
DATASETS = PROJECT_ROOT / "datasets"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = data["files"]
    aio_link_map = []

    by_role = {
        (r["domain"], r["role"]): r
        for r in files
        if r["role"] in {"trainA", "trainB"}
    }
    train_a = [r for r in files if r["role"] == "trainA"]
    train_b = [r for r in files if r["role"] == "trainB"]

    # Single-task dataroots.
    for r in train_a:
        dst = DATASETS / "single" / r["domain"] / "trainA" / f"{r['stem']}.{r['ext']}"
        link(Path(r["source_path"]), dst)
    for r in train_b:
        dst = DATASETS / "single" / r["domain"] / "trainB" / f"{r['stem']}.{r['ext']}"
        link(Path(r["source_path"]), dst)

    # All-in-one union dataroot with domain-prefixed names.
    for r in train_a:
        name = f"{r['domain']}__{r['stem']}.{r['ext']}"
        dst = DATASETS / "aio" / "trainA" / name
        link(Path(r["source_path"]), dst)
        aio_link_map.append({"split": "trainA", "filename": name, "source": r["source_path"]})
    for r in train_b:
        name = f"{r['domain']}__{r['stem']}.{r['ext']}"
        dst = DATASETS / "aio" / "trainB" / name
        link(Path(r["source_path"]), dst)
        aio_link_map.append({"split": "trainB", "filename": name, "source": r["source_path"]})

    (DATASETS / "aio" / "link_map.csv").write_text(
        "split,filename,source\n"
        + "\n".join(f"{m['split']},{m['filename']},{m['source']}" for m in aio_link_map),
        encoding="utf-8",
    )

    counts = {
        "single_trainA": len(list((DATASETS / "single").rglob("trainA/*"))),
        "single_trainB": len(list((DATASETS / "single").rglob("trainB/*"))),
        "aio_trainA": len(list((DATASETS / "aio" / "trainA").iterdir())),
        "aio_trainB": len(list((DATASETS / "aio" / "trainB").iterdir())),
    }
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
