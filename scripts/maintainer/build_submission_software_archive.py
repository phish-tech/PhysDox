"""Build the reviewer-friendly EMNLP camera-ready software archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "3919_PhysDox_Benchmarking_LLMs_Software_camera_ready.zip"

ROOT_FILES = [
    ".python-version",
    ".env.example",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE-DATA",
    "pyproject.toml",
    "uv.lock",
]

ARCHIVE_DOCS = {
    "docs/software_archive/README.md": "README.md",
    "docs/software_archive/QUICKSTART.md": "QUICKSTART.md",
    "docs/software_archive/REPRODUCIBILITY.md": "REPRODUCIBILITY.md",
}

COPY_DIRS = ["src", "scripts"]

EXCLUDED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignore_names(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDED_NAMES or name.endswith(".pyc")}


def copy_required_files(staging: Path) -> None:
    for relative in ROOT_FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for source_relative, target_relative in ARCHIVE_DOCS.items():
        source = ROOT / source_relative
        if not source.exists():
            raise FileNotFoundError(source)
        target = staging / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for relative in COPY_DIRS:
        source = ROOT / relative
        target = staging / relative
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copytree(source, target, ignore=ignore_names)


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def manifest(staging: Path, archive_name: str) -> dict:
    files = [
        {
            "path": path.relative_to(staging).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in iter_files(staging)
        if path.name != "software_manifest.json"
    ]
    return {
        "archive": archive_name,
        "purpose": "EMNLP Findings camera-ready software upload",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "PhysDox",
        "scope": "evaluation software, audit scripts, quickstart, and reproducibility docs",
        "expected_paired_data_archive": "3919_PhysDox_Benchmarking_LLMs_Data.zip",
        "included_roots": ["src/hallucination_benchmark", "scripts"],
        "included_docs": ["README.md", "QUICKSTART.md", "REPRODUCIBILITY.md"],
        "cli_entrypoints": [
            "benchmark-two-stage-eval",
            "benchmark-evaluate",
            "benchmark-experiment",
            "benchmark-generate",
            "benchmark-rebalance",
        ],
        "audit_commands": [
            "uv run python scripts/audit_experiments.py --root <data-archive>/experiment_outputs/experiments/main_20260510_main",
            "uv run python scripts/audit_response_format.py --root <data-archive>/experiment_outputs/experiments/main_20260510_main",
            "uv run python scripts/export_main_results_table.py --root <data-archive>/experiment_outputs/experiments/main_20260510_main --output ./reproduced_tables",
        ],
        "excluded": [
            "data/",
            "experiments/",
            "dist/",
            "tmp/",
            ".git/",
            ".env",
            "runtime outputs",
            "checkpoints",
            "logs",
            "source workbooks",
            "review spreadsheets",
        ],
        "file_count": len(files) + 1,
        "files": files,
    }


def write_zip(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in iter_files(staging):
            zf.write(path, path.relative_to(staging).as_posix())


def build_archive(output: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="physdox_software_archive_") as tempdir:
        staging = Path(tempdir) / "software"
        staging.mkdir()
        copy_required_files(staging)
        manifest_path = staging / "software_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest(staging, output.name), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_zip(staging, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_archive(args.output)
    print(output)
    print(f"size_bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
