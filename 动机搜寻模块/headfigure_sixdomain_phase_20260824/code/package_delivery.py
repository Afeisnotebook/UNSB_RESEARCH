from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-module", required=True)
    parser.add_argument("--existing-validation", required=True)
    parser.add_argument("--zip-output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    module_root = Path(args.repo_module).resolve()
    existing = Path(args.existing_validation).resolve()
    delivery = module_root / "final_delivery"
    if delivery.exists():
        shutil.rmtree(delivery)
    delivery.mkdir(parents=True)

    copy_file(module_root / "README_CN.md", delivery / "README_CN.md")
    copy_file(
        module_root / "SIXDOMAIN_PHASE_FROZEN_PROTOCOL.json",
        delivery / "SIXDOMAIN_PHASE_FROZEN_PROTOCOL.json",
    )
    for path in sorted((module_root / "code").glob("*.py")):
        copy_file(path, delivery / "code" / path.name)

    state_names = [
        "PREPARE_AUDIT.json",
        "DATA_MANIFEST.csv",
        "HELDOUT_MANIFEST.json",
        "PATH_MAP.json",
        "TRAINING_STATE.json",
        "MEASUREMENT_STATE.json",
    ]
    for name in state_names:
        copy_file(run_root / "state" / name, delivery / "state" / name)
    for path in sorted((run_root / "reports").glob("*")):
        if path.is_file():
            copy_file(path, delivery / "reports" / path.name)
    for path in sorted((run_root / "figures").glob("UNSB_SIXDOMAIN_PHASE_HEADFIGURE.*")):
        copy_file(path, delivery / "figures" / path.name)
    for name in (
        "RECIPROCAL_KERNEL_BY_AGE.csv",
        "RECIPROCAL_KERNEL_PRIMARY.csv",
        "RECIPROCAL_REPRESENTATIVE_PCA.csv",
    ):
        copy_file(run_root / "raw" / name, delivery / "raw" / name)

    for name in (
        "PHASE_STATISTICS.json",
        "PHASE_CELL_SUMMARY.csv",
        "CROSSFIT_DIRECTION_DETAILS.csv",
    ):
        copy_file(existing / name, delivery / "existing_five_domain_validation" / name)

    files = sorted(path for path in delivery.rglob("*") if path.is_file())
    manifest_lines = [
        f"{sha256(path)}  {path.relative_to(delivery).as_posix()}" for path in files
    ]
    manifest = delivery / "MANIFEST.sha256"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    zip_output = Path(args.zip_output).resolve()
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    if zip_output.exists():
        zip_output.unlink()
    archive_root = "UNSB_SIXDOMAIN_PHASE_HANDOFF_20260825"
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(delivery.rglob("*")):
            if path.is_file():
                archive.write(path, f"{archive_root}/{path.relative_to(delivery).as_posix()}")
    sidecar = zip_output.with_suffix(zip_output.suffix + ".sha256")
    sidecar.write_text(f"{sha256(zip_output)}  {zip_output.name}\n", encoding="utf-8")

    with zipfile.ZipFile(zip_output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate ZIP entries")
        forbidden = ("/.git/", "/__pycache__/", "/checkpoints/", "/data_views/")
        if any(any(token in f"/{name}" for token in forbidden) for name in names):
            raise RuntimeError("forbidden content in ZIP")
        archive.testzip()
    print(f"delivery={delivery}")
    print(f"zip={zip_output}")
    print(f"sha256={sha256(zip_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
