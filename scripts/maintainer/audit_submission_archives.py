"""Audit submitted EMNLP camera-ready software and data archives.

The audit is intentionally lightweight: it verifies archive contents, counts
released datasets and experiment outputs, and optionally checks that the
software archive exposes the expected CLI entrypoints after extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


SOFTWARE_REQUIRED = [
    ".env.example",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-DATA",
    "QUICKSTART.md",
    "README.md",
    "REPRODUCIBILITY.md",
    "pyproject.toml",
    "software_manifest.json",
    "uv.lock",
    "src/hallucination_benchmark/two_stage_eval.py",
    "src/hallucination_benchmark/rule_checkers.py",
    "src/hallucination_benchmark/llm_client.py",
    "scripts/audit_experiments.py",
    "scripts/audit_response_format.py",
    "scripts/export_main_results_table.py",
]

DATA_REQUIRED = [
    "README.md",
    "data_manifest.json",
    "benchmark_data/data/current/benchmark_balanced.json",
    "benchmark_data/data/gold/gold_0510_final_full.json",
    "benchmark_data/data/seeds/constraints.json",
    "benchmark_data/data/seeds/domains.json",
    "experiment_outputs/experiment_summary.json",
]

FORBIDDEN_MEMBER_PATTERNS = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)"),
    re.compile(r"(^|/)(checkpoint|run_log|repair_manifest)\.(json|jsonl)$"),
    re.compile(r"\.(key|pem|p12|xlsx)$", re.IGNORECASE),
]

TEXT_EXTENSIONS = {".cff", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
TEXT_SCAN_EXCLUDED = {
    "scripts/audit_release.py",
    "scripts/audit_submission_archives.py",
}
ABSOLUTE_PATH_PATTERN = re.compile(r"/(?:Users|home|mnt)/[^\s'\"`]+")
SECRET_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{20,}|gh[opusr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_summary(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        corrupt_member = zf.testzip()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "file_count": len(infos),
        "uncompressed_bytes": sum(info.file_size for info in infos),
        "corrupt_member": corrupt_member,
    }


def read_json_from_zip(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name) as handle:
        return json.load(handle)


def count_jsonl_rows(zf: zipfile.ZipFile, names: list[str]) -> int:
    total = 0
    for name in names:
        with zf.open(name) as handle:
            total += sum(1 for _ in handle)
    return total


def audit_member_names(names: set[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if any(pattern.search(name) for pattern in FORBIDDEN_MEMBER_PATTERNS)
    )


def audit_software_text(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    findings: list[str] = []
    for name in sorted(names - TEXT_SCAN_EXCLUDED):
        if Path(name).suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = zf.read(name).decode("utf-8")
        except UnicodeDecodeError:
            continue
        for match in ABSOLUTE_PATH_PATTERN.finditer(text):
            findings.append(f"absolute path in {name}: {match.group(0)}")
        for match in SECRET_PATTERN.finditer(text):
            findings.append(f"possible credential in {name}: {match.group(0)[:12]}...")
    return findings


def audit_software_manifest(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    errors: list[str] = []
    try:
        manifest = read_json_from_zip(zf, "software_manifest.json")
    except Exception as exc:
        return [f"cannot read software_manifest.json: {exc}"]

    entries = manifest.get("files", [])
    by_name = {entry.get("path"): entry for entry in entries}
    archive_files = names - {"software_manifest.json"}
    if set(by_name) != archive_files:
        missing = sorted(archive_files - set(by_name))
        extra = sorted(set(by_name) - archive_files)
        if missing:
            errors.append(f"manifest missing archive files: {missing}")
        if extra:
            errors.append(f"manifest lists absent files: {extra}")

    if manifest.get("file_count") != len(names):
        errors.append(
            f"manifest file_count={manifest.get('file_count')} but archive has {len(names)}"
        )

    for name in sorted(archive_files & set(by_name)):
        content = zf.read(name)
        entry = by_name[name]
        if entry.get("size_bytes") != len(content):
            errors.append(f"size mismatch: {name}")
        if entry.get("sha256") != hashlib.sha256(content).hexdigest():
            errors.append(f"SHA-256 mismatch: {name}")
    return errors


def audit_software(path: Path, verify_cli: bool) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = {name for name in zf.namelist() if not name.endswith("/")}
        missing = [name for name in SOFTWARE_REQUIRED if name not in names]
        scripts = sorted(name for name in names if name.startswith("scripts/"))
        package_files = sorted(
            name for name in names if name.startswith("src/hallucination_benchmark/")
        )
        unsafe_paths = sorted(
            name
            for name in names
            if name.startswith("/") or ".." in Path(name).parts
        )
        manifest_errors = audit_software_manifest(zf, names)
        forbidden_members = audit_member_names(names)
        sensitive_text_findings = audit_software_text(zf, names)

    result: dict[str, Any] = {
        **archive_summary(path),
        "required_files_missing": missing,
        "script_file_count": len(scripts),
        "package_file_count": len(package_files),
        "manifest_errors": manifest_errors,
        "unsafe_paths": unsafe_paths,
        "forbidden_members": forbidden_members,
        "sensitive_text_findings": sensitive_text_findings,
    }
    if verify_cli:
        result["cli_checks"] = verify_software_cli(path)
    return result


def verify_software_cli(path: Path) -> list[dict[str, Any]]:
    uv_path = shutil.which("uv")
    if not uv_path:
        return [{"command": "uv", "status": "missing", "returncode": None}]

    checks = [
        ["uv", "run", "benchmark-two-stage-eval", "--help"],
        ["uv", "run", "python", "scripts/audit_experiments.py", "--help"],
        ["uv", "run", "python", "scripts/audit_response_format.py", "--help"],
    ]
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="physdox_software_audit_") as tempdir:
        root = Path(tempdir)
        env = os.environ.copy()
        env.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(root / ".uv-python"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(root)
        for command in checks:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
            output = completed.stdout.strip().splitlines()
            results.append(
                {
                    "command": " ".join(command),
                    "returncode": completed.returncode,
                    "status": "passed" if completed.returncode == 0 else "failed",
                    "output_head": output[:8],
                }
            )
    return results


def verify_two_archive_workflow(
    software_path: Path, data_path: Path
) -> list[dict[str, Any]]:
    checks = [
        [
            "uv",
            "run",
            "python",
            "scripts/audit_experiments.py",
            "--root",
            "../data_archive/experiment_outputs/experiments/main_20260510_main",
        ],
        [
            "uv",
            "run",
            "python",
            "scripts/audit_response_format.py",
            "--root",
            "../data_archive/experiment_outputs/experiments/main_20260510_main",
        ],
        [
            "uv",
            "run",
            "python",
            "scripts/export_main_results_table.py",
            "--root",
            "../data_archive/experiment_outputs/experiments/main_20260510_main",
            "--output",
            "./reproduced_tables",
        ],
    ]
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="physdox_workflow_audit_") as tempdir:
        root = Path(tempdir)
        software_root = root / "software"
        data_root = root / "data_archive"
        software_root.mkdir()
        data_root.mkdir()
        with zipfile.ZipFile(software_path) as zf:
            zf.extractall(software_root)
        with zipfile.ZipFile(data_path) as zf:
            zf.extractall(data_root)

        env = os.environ.copy()
        env.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(root / ".uv-python"))
        for command in checks:
            completed = subprocess.run(
                command,
                cwd=software_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
            output = completed.stdout.strip().splitlines()
            response_format_findings = (
                "scripts/audit_response_format.py" in command
                and completed.returncode == 1
            )
            results.append(
                {
                    "command": " ".join(command),
                    "returncode": completed.returncode,
                    "status": (
                        "passed"
                        if completed.returncode == 0
                        else "completed_with_findings"
                        if response_format_findings
                        else "failed"
                    ),
                    "output_tail": output[-8:],
                }
            )
    return results


def audit_data(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        missing = [name for name in DATA_REQUIRED if name not in names]
        per_item = sorted(
            name
            for name in names
            if name.endswith("/per_item_results.jsonl")
            and "/experiments/main_20260510_main/" in name
        )
        summaries = sorted(
            name
            for name in names
            if name.endswith("/summary.json")
            and "/experiments/main_20260510_main/" in name
        )
        confusion = sorted(
            name
            for name in names
            if name.endswith("/confusion_matrices.json")
            and "/experiments/main_20260510_main/" in name
        )
        three_class = sorted(
            name
            for name in names
            if name.endswith("/three_class_analysis.json")
            and "/experiments/main_20260510_main/" in name
        )
        tables = sorted(
            name
            for name in names
            if "/experiments/main_20260510_main/tables/" in name
        )

        silver = read_json_from_zip(
            zf, "benchmark_data/data/current/benchmark_balanced.json"
        )
        gold = read_json_from_zip(zf, "benchmark_data/data/gold/gold_0510_final_full.json")
        per_item_rows = count_jsonl_rows(zf, per_item)
        forbidden_members = audit_member_names(names)

    return {
        **archive_summary(path),
        "required_files_missing": missing,
        "silver_items": len(silver),
        "silver_labels": dict(Counter(item.get("label") for item in silver)),
        "silver_unique_ids": len({item.get("id") for item in silver}),
        "silver_unique_protocols": len({item.get("protocol") for item in silver}),
        "gold_items": len(gold),
        "gold_labels": dict(Counter(item.get("label") for item in gold)),
        "gold_unique_ids": len({item.get("id") for item in gold}),
        "gold_unique_protocols": len({item.get("protocol") for item in gold}),
        "main_per_item_files": len(per_item),
        "main_per_item_rows": per_item_rows,
        "main_summary_files": len(summaries),
        "main_confusion_files": len(confusion),
        "main_three_class_files": len(three_class),
        "main_table_files": len(tables),
        "forbidden_members": forbidden_members,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    software = report["software"]
    data = report["data"]
    cli_checks = software.get("cli_checks", [])
    cli_rows = "\n".join(
        f"| `{check['command']}` | {check['status']} | {check['returncode']} |"
        for check in cli_checks
    )
    if not cli_rows:
        cli_rows = "| Not run | n/a | n/a |"
    workflow_checks = report.get("workflow_checks", [])
    workflow_rows = "\n".join(
        f"| `{check['command']}` | {check['status']} | {check['returncode']} |"
        for check in workflow_checks
    )
    if not workflow_rows:
        workflow_rows = "| Not run | n/a | n/a |"

    text = f"""# Submission Archive Audit

Generated by `scripts/audit_submission_archives.py`.

## Software Archive

| Field | Value |
|---|---:|
| File | `{software['name']}` |
| Size | {software['size_bytes']:,} bytes |
| SHA-256 | `{software['sha256']}` |
| Files | {software['file_count']} |
| Python package files | {software['package_file_count']} |
| Script files | {software['script_file_count']} |
| Missing required files | {len(software['required_files_missing'])} |
| Manifest errors | {len(software['manifest_errors'])} |
| Unsafe archive paths | {len(software['unsafe_paths'])} |
| Forbidden members | {len(software['forbidden_members'])} |
| Sensitive text findings | {len(software['sensitive_text_findings'])} |
| Corrupt member | {software['corrupt_member'] or 'none'} |

## Data Archive

| Field | Value |
|---|---:|
| File | `{data['name']}` |
| Size | {data['size_bytes']:,} bytes |
| SHA-256 | `{data['sha256']}` |
| Files | {data['file_count']} |
| Missing required files | {len(data['required_files_missing'])} |
| Silver items | {data['silver_items']:,} |
| Silver unique protocols | {data['silver_unique_protocols']:,} |
| Gold items | {data['gold_items']:,} |
| Gold unique protocols | {data['gold_unique_protocols']:,} |
| Main per-item result files | {data['main_per_item_files']} |
| Main per-item rows | {data['main_per_item_rows']:,} |
| Main summary files | {data['main_summary_files']} |
| Main confusion-matrix files | {data['main_confusion_files']} |
| Main three-class analysis files | {data['main_three_class_files']} |
| Main table files | {data['main_table_files']} |
| Forbidden members | {len(data['forbidden_members'])} |
| Corrupt member | {data['corrupt_member'] or 'none'} |

## CLI Checks

| Command | Status | Return code |
|---|---|---:|
{cli_rows}

## Two-Archive Workflow Checks

| Command | Status | Return code |
|---|---|---:|
{workflow_rows}

## Interpretation

The software archive contains the evaluation package, CLI metadata, audit
scripts, result-table exporter, lock file, and license/citation files. The data
archive contains the Gold and Silver benchmark copies plus the main paper-facing
experiment outputs, including per-item JSONL outputs for all 48 runs. The
response-format audit reports the documented individual self-consistency sample
findings while all aggregate run outputs remain complete. Together, the
camera-ready archives provide a machine-checkable offline audit workflow.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software-zip", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/camera_ready_audit"),
    )
    parser.add_argument("--verify-cli", action="store_true")
    parser.add_argument("--verify-workflow", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "software": audit_software(args.software_zip, args.verify_cli),
        "data": audit_data(args.data_zip),
    }
    if args.verify_workflow:
        report["workflow_checks"] = verify_two_archive_workflow(
            args.software_zip, args.data_zip
        )
    json_path = args.output_dir / "archive_audit.json"
    md_path = args.output_dir / "archive_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, md_path)
    checksums_path = args.output_dir / "SHA256SUMS.txt"
    checksums_path.write_text(
        f"{report['software']['sha256']}  {args.software_zip.name}\n"
        f"{report['data']['sha256']}  {args.data_zip.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {checksums_path}")

    errors = (
        report["software"]["required_files_missing"]
        + report["software"]["manifest_errors"]
        + report["software"]["unsafe_paths"]
        + report["software"]["forbidden_members"]
        + report["software"]["sensitive_text_findings"]
        + report["data"]["required_files_missing"]
        + report["data"]["forbidden_members"]
    )
    failed_checks = [
        check
        for check in (
            report["software"].get("cli_checks", [])
            + report.get("workflow_checks", [])
        )
        if check["status"] == "failed"
    ]
    if report["software"]["corrupt_member"] or report["data"]["corrupt_member"]:
        errors.append("corrupt archive member")
    if errors or failed_checks:
        print("Archive audit failed.")
        return 1
    print("Archive audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
