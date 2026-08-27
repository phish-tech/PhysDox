#!/usr/bin/env python3
"""Audit whether model responses match the requested two-stage JSON formats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hallucination_benchmark.two_stage_eval import (
    FINE_CONSTRAINTS,
    LABELS,
    THREE_CLASS_CATEGORIES,
    parse_stage1,
    parse_stage2,
)

TRANSPORT_MARKERS = (
    "API call failed",
    "HTTP Error",
    "Bad Gateway",
    "Forbidden",
    "Too Many Requests",
    "Insufficient Balance",
    "余额不足",
    "预扣费额度失败",
    "empty response",
    "timed out",
    "timeout",
    "Connection reset",
)


def _load_rows(exp_dir: Path) -> tuple[str, list[dict]]:
    per_item = exp_dir / "per_item_results.jsonl"
    checkpoint = exp_dir / "checkpoint.json"
    if per_item.exists():
        rows = []
        with per_item.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return "per_item_results.jsonl", rows
    if checkpoint.exists():
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        return "checkpoint.json", [
            {"id": item_id, **cached}
            for item_id, cached in data.items()
            if isinstance(cached, dict)
        ]
    return "none", []


def _has_transport_error(stage: dict | None) -> bool:
    if not isinstance(stage, dict):
        return False
    text = "\n".join(str(stage.get(key) or "") for key in ("raw_text", "error"))
    return any(marker in text for marker in TRANSPORT_MARKERS)


def _strict_stage1_raw(raw_response) -> bool:
    return isinstance(raw_response, dict) and raw_response.get("label") in LABELS


def _strict_stage2_raw(raw_response) -> bool:
    if not isinstance(raw_response, dict):
        return False
    categories = raw_response.get("three_class_categories")
    constraints = raw_response.get("fine_constraints")
    return (
        isinstance(categories, list)
        and bool(categories)
        and categories[0] in THREE_CLASS_CATEGORIES
        and isinstance(constraints, list)
        and bool(constraints)
        and constraints[0] in FINE_CONSTRAINTS
    )


def _audit_stage1(stage: dict | None) -> tuple[str, str]:
    if not isinstance(stage, dict):
        return "missing", "stage is not a dict"
    if _has_transport_error(stage):
        return "transport_error", str(stage.get("error") or stage.get("raw_text") or "")[:160]
    raw_response = stage.get("raw_response")
    if isinstance(raw_response, dict) and isinstance(raw_response.get("self_consistency_samples"), list):
        return "self_cons_wrapper", ""
    if _strict_stage1_raw(raw_response):
        return "strict_json", ""
    recovered = parse_stage1(raw_response, stage.get("raw_text") or "")
    if recovered in LABELS:
        return "parser_recovered", str(stage.get("raw_text") or raw_response or "")[:160]
    return "unparseable", str(stage.get("raw_text") or raw_response or "")[:160]


def _audit_stage2(stage: dict | None) -> tuple[str, str]:
    if not isinstance(stage, dict):
        return "missing", "stage is not a dict"
    if _has_transport_error(stage):
        return "transport_error", str(stage.get("error") or stage.get("raw_text") or "")[:160]
    raw_response = stage.get("raw_response")
    if isinstance(raw_response, dict) and isinstance(raw_response.get("self_consistency_samples"), list):
        return "self_cons_wrapper", ""
    if _strict_stage2_raw(raw_response):
        return "strict_json", ""
    fine, three = parse_stage2(raw_response, stage.get("raw_text") or "")
    if three:
        return "parser_recovered", str(stage.get("raw_text") or raw_response or "")[:160]
    return "unparseable", str(stage.get("raw_text") or raw_response or "")[:160]


def _audit_self_cons(stage: dict | None, stage_name: str) -> tuple[Counter, list[str]]:
    counts = Counter()
    examples = []
    raw_response = stage.get("raw_response") if isinstance(stage, dict) else None
    samples = raw_response.get("self_consistency_samples") if isinstance(raw_response, dict) else None
    if not isinstance(samples, list):
        return counts, examples
    for sample in samples:
        if not isinstance(sample, dict):
            counts["sample_missing"] += 1
            continue
        raw = {
            "raw_response": sample.get("raw_response"),
            "raw_text": sample.get("raw_text") or "",
            "error": sample.get("error"),
        }
        status, detail = _audit_stage1(raw) if stage_name == "stage1" else _audit_stage2(raw)
        counts[f"sample_{status}"] += 1
        if status != "strict_json" and len(examples) < 3:
            examples.append(detail)
    return counts, examples


def _expects_stage2(row: dict) -> bool:
    return row.get("label") == "fatal_violation" or (row.get("stage1") or {}).get("reference") == "fatal_violation"


def audit_experiment(exp_dir: Path) -> dict:
    source, rows = _load_rows(exp_dir)
    counts = Counter()
    examples = []
    sample_counts = Counter()
    for row in rows:
        item_id = row.get("id", "")
        status, detail = _audit_stage1(row.get("stage1"))
        counts[f"stage1_{status}"] += 1
        sc_counts, sc_examples = _audit_self_cons(row.get("stage1"), "stage1")
        sample_counts.update(sc_counts)
        if status != "strict_json" and len(examples) < 10:
            examples.append({"id": item_id, "stage": "stage1", "status": status, "detail": detail})
        for detail in sc_examples:
            if len(examples) < 10:
                examples.append({"id": item_id, "stage": "stage1_sample", "status": "sample_non_strict", "detail": detail})

        if not _expects_stage2(row):
            counts["stage2_not_applicable"] += 1
            continue
        status, detail = _audit_stage2(row.get("stage2"))
        counts[f"stage2_{status}"] += 1
        sc_counts, sc_examples = _audit_self_cons(row.get("stage2"), "stage2")
        sample_counts.update(sc_counts)
        if status != "strict_json" and len(examples) < 10:
            examples.append({"id": item_id, "stage": "stage2", "status": status, "detail": detail})
        for detail in sc_examples:
            if len(examples) < 10:
                examples.append({"id": item_id, "stage": "stage2_sample", "status": "sample_non_strict", "detail": detail})

    strict_failures = sum(
        value
        for key, value in counts.items()
        if key.startswith(("stage1_", "stage2_"))
        and key not in {
            "stage1_strict_json",
            "stage2_strict_json",
            "stage1_self_cons_wrapper",
            "stage2_self_cons_wrapper",
            "stage2_not_applicable",
        }
    )
    parser_recovered = counts["stage1_parser_recovered"] + counts["stage2_parser_recovered"]
    hard_failures = (
        counts["stage1_missing"]
        + counts["stage1_transport_error"]
        + counts["stage1_unparseable"]
        + counts["stage2_missing"]
        + counts["stage2_transport_error"]
        + counts["stage2_unparseable"]
    )
    sample_hard_failures = (
        sample_counts["sample_missing"]
        + sample_counts["sample_transport_error"]
        + sample_counts["sample_unparseable"]
    )
    return {
        "experiment": exp_dir.name,
        "source": source,
        "rows": len(rows),
        "strict_failures": strict_failures,
        "parser_recovered": parser_recovered,
        "hard_failures": hard_failures,
        "sample_hard_failures": sample_hard_failures,
        "counts": dict(counts),
        "self_cons_sample_counts": dict(sample_counts),
        "examples": examples,
    }


def _has_outputs(path: Path) -> bool:
    return path.is_dir() and ((path / "per_item_results.jsonl").exists() or (path / "checkpoint.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments/main_20260510_main")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    results = []
    for path in sorted(root.iterdir()):
        if not _has_outputs(path):
            continue
        if args.completed_only and not (path / "summary.json").exists():
            continue
        results.append(audit_experiment(path))

    totals = Counter()
    sample_totals = Counter()
    for result in results:
        totals.update(result["counts"])
        sample_totals.update(result["self_cons_sample_counts"])

    payload = {
        "experiments": results,
        "totals": dict(totals),
        "self_cons_sample_totals": dict(sample_totals),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if any(row["hard_failures"] or row["sample_hard_failures"] for row in results) else 0

    print("experiment\tsource\trows\tstrict_failures\tparser_recovered\thard_failures")
    for row in results:
        print(
            f"{row['experiment']}\t{row['source']}\t{row['rows']}\t"
            f"{row['strict_failures']}\t{row['parser_recovered']}\t"
            f"{row['hard_failures']}+{row['sample_hard_failures']}"
        )
    print("\nTotals:")
    for key, value in sorted(totals.items()):
        print(f"- {key}: {value}")
    if sample_totals:
        print("\nSelf-consistency sample totals:")
        for key, value in sorted(sample_totals.items()):
            print(f"- {key}: {value}")
    problematic = [row for row in results if row["hard_failures"] or row["sample_hard_failures"]]
    if problematic:
        print("\nHard failure examples:")
        for row in problematic[:10]:
            print(f"- {row['experiment']}")
            for example in row["examples"][:3]:
                print(f"  {example['id']} {example['stage']} {example['status']}: {example['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
