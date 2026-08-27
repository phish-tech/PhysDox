#!/usr/bin/env python3
"""Audit the repository surface before a public release."""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "LICENSE-DATA",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "data/README.md",
    "experiments/README.md",
    "docs/release_manifest.md",
    "data/current/benchmark_balanced.json",
    "data/current/benchmark_balanced_manifest.json",
    "data/gold/gold_0510_final_full.json",
    "data/gold/gold_0510_final_full_manifest.json",
    "data/seeds/constraints.json",
    "data/seeds/domains.json",
]

FORBIDDEN_TRACKED_PATTERNS = [
    ".env",
    "*.key",
    "data/current/benchmark_balanced_unified.json",
    "data/current/review_*.xlsx",
    "data/gold/source/*",
    "data/gold/*.xlsx",
    "data/gold/gold_0507_*",
    "data/gold/gold_0510_final_core_*",
    "data/gold/gold_0510_protocol_rewrite_*",
    "data/gold/gold_0510_super_confusing_*",
    "data/releases/*",
    "data/legacy/*",
    "data/generated/*",
    "data/runtime/*",
    "experiments/**/per_item_results.jsonl",
    "experiments/**/run_log.jsonl",
    "experiments/**/checkpoint.json",
    "experiments/**/repair_manifest.json",
    "experiments/**/tables/*.xlsx",
]

SUSPICIOUS_TEXT_PATTERNS = [
    "/home/julian/",
    "/mnt/c/Users/",
]

TEXT_EXTENSIONS = {
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}


def release_surface_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line]


def matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.rstrip("/") + "/**")


def audit_required_files(errors: list[str]) -> None:
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            errors.append(f"missing required release file: {rel}")


def audit_forbidden_tracked(tracked: list[str], errors: list[str]) -> None:
    for path in tracked:
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if matches(path, pattern):
                errors.append(f"forbidden tracked file: {path} (matches {pattern})")
                break


def audit_json(tracked: list[str], errors: list[str]) -> None:
    for path in tracked:
        if not path.endswith(".json"):
            continue
        try:
            with (ROOT / path).open(encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            errors.append(f"invalid JSON: {path}: {exc}")


def audit_suspicious_text(tracked: list[str], errors: list[str]) -> None:
    for path in tracked:
        if path == "scripts/audit_release.py":
            continue
        full_path = ROOT / path
        if full_path.suffix not in TEXT_EXTENSIONS:
            continue
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SUSPICIOUS_TEXT_PATTERNS:
            if pattern in text:
                errors.append(f"suspicious text in {path}: {pattern}")


def main() -> int:
    errors: list[str] = []
    tracked = release_surface_files()
    audit_required_files(errors)
    audit_forbidden_tracked(tracked, errors)
    audit_json(tracked, errors)
    audit_suspicious_text(tracked, errors)

    if errors:
        print("Release audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Release audit passed.")
    print(f"Release-surface files checked: {len(tracked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
