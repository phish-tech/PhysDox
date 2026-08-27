#!/usr/bin/env python3
"""Audit experiment outputs for incomplete or HTTP-contaminated results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__load_error__": str(exc)}


def _scan_stage(stage: dict | None, stage_name: str) -> tuple[bool, bool, str]:
    if not isinstance(stage, dict):
        return True, False, ""
    raw = stage.get("raw_text") or ""
    error = stage.get("error") or ""
    scan_text = f"{raw}\n{error}"
    has_http = any(marker in scan_text for marker in HTTP_MARKERS)
    if stage_name == "stage1":
        missing = not bool(stage.get("prediction"))
    else:
        missing = not bool(stage.get("three_class_prediction"))
    sample = (error or raw)[:160].replace("\n", " ")
    return missing, has_http, sample


def _expects_stage2(row: dict) -> bool:
    """Stage 2 is only defined for fatal rows."""
    if row.get("label") == "fatal_violation":
        return True
    stage1 = row.get("stage1") or {}
    if stage1.get("reference") == "fatal_violation":
        return True
    return False


def audit_experiment(exp_dir: Path) -> dict:
    checkpoint_path = exp_dir / "checkpoint.json"
    per_item_path = exp_dir / "per_item_results.jsonl"
    summary_path = exp_dir / "summary.json"
    source_path = checkpoint_path if checkpoint_path.exists() else per_item_path
    status = "completed" if summary_path.exists() else "checkpoint"
    if summary_path.exists() and checkpoint_path.exists():
        status = "repair_checkpoint"
    result = {
        "experiment": exp_dir.name,
        "status": status,
        "source": source_path.name if source_path.exists() else "none",
        "items": 0,
        "stage1_missing": 0,
        "stage1_http": 0,
        "stage2_missing": 0,
        "stage2_http": 0,
        "examples": [],
    }
    if not source_path.exists():
        return result

    if source_path.name == "checkpoint.json":
        data = _load_json(source_path)
        rows = []
        if isinstance(data, dict):
            for item_id, cached in data.items():
                if isinstance(cached, dict):
                    rows.append({"id": item_id, **cached})
                else:
                    rows.append({"id": item_id, "__invalid_checkpoint_entry__": True})
    else:
        rows = []
        with source_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    result["items"] = len(rows)
    for row in rows:
        item_id = row.get("id", "")
        for stage_name in ("stage1", "stage2"):
            if stage_name == "stage2" and not _expects_stage2(row):
                continue
            missing, has_http, raw_sample = _scan_stage(row.get(stage_name), stage_name)
            if missing:
                result[f"{stage_name}_missing"] += 1
            if has_http:
                result[f"{stage_name}_http"] += 1
            if (missing or has_http) and len(result["examples"]) < 8:
                result["examples"].append({
                    "id": item_id,
                    "stage": stage_name,
                    "missing": missing,
                    "http": has_http,
                    "raw": raw_sample,
                })
    return result


def _has_experiment_outputs(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return (path / "summary.json").exists() or (path / "checkpoint.json").exists()
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments/main_20260510_main")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    args = parser.parse_args()

    root = Path(args.root)
    results = [
        audit_experiment(path)
        for path in sorted(root.iterdir())
        if _has_experiment_outputs(path)
    ]
    problematic = [
        row for row in results
        if row["stage1_missing"] or row["stage1_http"] or row["stage2_missing"] or row["stage2_http"]
    ]
    if args.json:
        print(json.dumps({"experiments": results, "problematic": problematic}, ensure_ascii=False, indent=2))
        return 1 if problematic else 0

    print("experiment\tstatus\titems\ts1_missing\ts1_http\ts2_missing\ts2_http")
    for row in results:
        print(
            f"{row['experiment']}\t{row['status']}\t{row['items']}\t"
            f"{row['stage1_missing']}\t{row['stage1_http']}\t"
            f"{row['stage2_missing']}\t{row['stage2_http']}"
        )
    if problematic:
        print("\nProblematic examples:")
        for row in problematic:
            print(f"- {row['experiment']}")
            for example in row["examples"][:3]:
                print(
                    f"  {example['id']} {example['stage']} "
                    f"missing={example['missing']} http={example['http']} raw={example['raw']}"
                )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
