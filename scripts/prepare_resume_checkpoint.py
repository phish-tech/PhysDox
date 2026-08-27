#!/usr/bin/env python3
"""Create a clean resume checkpoint from a completed two-stage run.

Use this when a completed run contains a small number of HTTP-contaminated or
missing predictions. The script keeps clean stage outputs and removes only the
stages that must be rerun.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

LABELS = {"valid", "minor_issue", "fatal_violation"}
THREE_CLASS_CATEGORIES = {
    "Signal & Sampling",
    "Perception & Anatomy",
    "Physical / Hardware",
}
HTTP_MARKERS = (
    "API call failed",
    "HTTP Error",
    "HTTP Error 402",
    "error code:",
    "Bad Gateway",
    "Forbidden",
    "Too Many Requests",
    "Insufficient Balance",
    "余额不足",
    "预扣费额度失败",
    "timed out",
    "timeout",
    "Remote end closed connection",
    "Connection reset",
)


def _has_http(stage: dict | None) -> bool:
    if not isinstance(stage, dict):
        return False
    text = "\n".join([stage.get("raw_text") or "", stage.get("error") or ""])
    return any(marker in text for marker in HTTP_MARKERS)


def _has_response(stage: dict | None) -> bool:
    if not isinstance(stage, dict):
        return False
    raw_response = stage.get("raw_response")
    raw_text = stage.get("raw_text")
    return raw_response is not None or bool(raw_text)


def _clean_stage1(stage: dict | None, strategy: str) -> bool:
    if not isinstance(stage, dict):
        return False
    return (
        stage.get("strategy", "zero-shot") == strategy
        and stage.get("prediction") in LABELS
        and _has_response(stage)
        and not _has_http(stage)
    )


def _clean_stage2(stage: dict | None, strategy: str) -> bool:
    if not isinstance(stage, dict):
        return False
    return (
        stage.get("strategy", "zero-shot") == strategy
        and stage.get("three_class_prediction") in THREE_CLASS_CATEGORIES
        and _has_response(stage)
        and not _has_http(stage)
    )


def prepare_checkpoint(exp_dir: Path, force: bool = False) -> dict:
    summary_path = exp_dir / "summary.json"
    per_item_path = exp_dir / "per_item_results.jsonl"
    checkpoint_path = exp_dir / "checkpoint.json"
    manifest_path = exp_dir / "repair_manifest.json"

    if checkpoint_path.exists() and not force:
        raise SystemExit(f"{checkpoint_path} already exists; pass --force to overwrite it.")
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")
    if not per_item_path.exists():
        raise SystemExit(f"Missing per-item output: {per_item_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    strategy = summary["strategy"]
    checkpoint = {}
    stats = {
        "experiment": exp_dir.name,
        "strategy": strategy,
        "created_at": datetime.now().isoformat(),
        "rows": 0,
        "kept_stage1": 0,
        "dropped_stage1": 0,
        "kept_stage2": 0,
        "dropped_stage2": 0,
    }

    with per_item_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row["id"]
            cached = {}
            stats["rows"] += 1

            stage1 = row.get("stage1")
            if _clean_stage1(stage1, strategy):
                cached["stage1"] = stage1
                stats["kept_stage1"] += 1
            else:
                stats["dropped_stage1"] += 1

            stage2 = row.get("stage2")
            if row.get("label") == "fatal_violation":
                if _clean_stage2(stage2, strategy):
                    cached["stage2"] = stage2
                    stats["kept_stage2"] += 1
                else:
                    stats["dropped_stage2"] += 1

            checkpoint[item_id] = cached

    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing checkpoint.json.")
    args = parser.parse_args()
    stats = prepare_checkpoint(Path(args.experiment_dir), force=args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
