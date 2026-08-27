#!/usr/bin/env python3
"""Import partner gold annotations and build confusing gold subsets."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

from hallucination_benchmark.config import GOLD_DATA_DIR

GOLD_SOURCE_DIR = GOLD_DATA_DIR / "source"
GOLD_SOURCE_FILE = GOLD_SOURCE_DIR / "gold_dataset_final_edit_0507.xlsx"
GOLD_FULL_FILE = GOLD_DATA_DIR / "gold_0507_full.json"
GOLD_MANIFEST_FILE = GOLD_DATA_DIR / "gold_0507_full_manifest.json"
CONFUSING_JSON = GOLD_DATA_DIR / "gold_0507_confusing_500.json"
CONFUSING_XLSX = GOLD_DATA_DIR / "gold_0507_confusing_500.xlsx"
FINAL_FULL_JSON = GOLD_DATA_DIR / "gold_0510_final_full.json"
FINAL_FULL_MANIFEST = GOLD_DATA_DIR / "gold_0510_final_full_manifest.json"
FINAL_FULL_XLSX = GOLD_DATA_DIR / "gold_0510_final_full.xlsx"
FINAL_CORE_JSON = GOLD_DATA_DIR / "gold_0510_final_core_500.json"
FINAL_CORE_XLSX = GOLD_DATA_DIR / "gold_0510_final_core_500.xlsx"
PREVIOUS_CORE_JSON = GOLD_DATA_DIR / "gold_0507_confusing_500.json"

THREE_CLASS_CATEGORIES = [
    "Signal & Sampling",
    "Perception & Anatomy",
    "Physical / Hardware",
]
FINE_TO_THREE_CLASS = {
    "sampling_theorem": "Signal & Sampling",
    "sensor_bandwidth": "Signal & Sampling",
    "snr_floor": "Signal & Sampling",
    "data_bandwidth": "Signal & Sampling",
    "adc_resolution": "Signal & Sampling",
    "latency_constraint": "Signal & Sampling",
    "anatomical_misplacement": "Perception & Anatomy",
    "optical_obstruction": "Perception & Anatomy",
    "power_budget": "Physical / Hardware",
    "temperature_range": "Physical / Hardware",
    "accuracy_impossibility": "Physical / Hardware",
    "ematerial_block": "Physical / Hardware",
}
FINAL_CORE_FATAL_THREE_CLASS_QUOTAS = {
    "Signal & Sampling": 103,
    "Perception & Anatomy": 50,
    "Physical / Hardware": 60,
}
FINAL_CORE_LABEL_QUOTAS = {
    "minor_issue": 74,
    "fatal_violation": sum(FINAL_CORE_FATAL_THREE_CLASS_QUOTAS.values()),
    "valid": 213,
}

DIRTY_PATTERNS = {
    "wrapper_source_protocol": re.compile(r"source protocol text", re.I),
    "bracket_placeholder": re.compile(r"\[(?:text|protocol title)\]", re.I),
    "llm_trace": re.compile(r"First,\s+the user wants|as an ai|language model", re.I),
    "as_written_above": re.compile(r"as written above", re.I),
    "full_protocol_placeholder": re.compile(r"full protocol text|modified protocol text", re.I),
    "nan_residue": re.compile(r"\bnan\b", re.I),
    "cjk_protocol_text": re.compile(r"[\u4e00-\u9fff]"),
}


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "none":
        return ""
    return text


def _choose_protocol(row: dict) -> tuple[str, str]:
    for col in ("final_protocol", "manual_rewrite", "Generated_Prompt"):
        text = _clean_text(row.get(col))
        if text:
            return _sanitize_protocol(text), col
    return "", "missing"


def _sanitize_protocol(text: str) -> str:
    """Remove review notes that were accidentally appended to protocol text."""
    cleaned = text.strip()
    leak_patterns = [
        r"\s*\[Error Summary:[\s\S]*?\]\s*$",
        r"\s*\[[^\[\]]*(?:数据泄露|错误提示|错误原因|非最优|删除|不要出现|不要)[\s\S]*?\]\s*$",
        r"\s*（[^（）]*(?:数据泄露|错误提示|错误原因|非最优|删除|不要出现|不要)[^（）]*）\s*\]?\s*$",
        r"\s*\([^()]*(?:data leakage|error summary|do not include)[^()]*\)\s*$",
    ]
    changed = True
    while changed:
        before = cleaned
        for pattern in leak_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.I).strip()
        changed = cleaned != before
    return cleaned


def _normalize_label(value: str) -> str:
    label = _clean_text(value)
    aliases = {
        "fatal": "fatal_violation",
        "fatal violation": "fatal_violation",
        "minor": "minor_issue",
        "minor issue": "minor_issue",
    }
    return aliases.get(label.lower().replace("_", " "), label)


def _annotation_value(row: dict, clean_col: str, final_col: str, patched_col: str = "") -> str:
    for col in (clean_col, final_col, patched_col):
        if not col:
            continue
        text = _clean_text(row.get(col))
        if text:
            return text
    return ""


def _resolved_three_class_category(row: dict, label: str) -> str:
    if label in {"valid", "minor_issue"}:
        return "N/A"
    constraint = _clean_text(row.get("Violated_Constraint"))
    return FINE_TO_THREE_CLASS.get(constraint, "")


def _build_gold_item(row: dict, index: int) -> dict:
    protocol, protocol_source = _choose_protocol(row)
    label = _normalize_label(row.get("Label"))
    is_valid = label == "valid"
    return {
        "id": f"gold_0507_{index:06d}",
        "source_id": _clean_text(row.get("ID")),
        "source_row": index + 1,
        "domain": _clean_text(row.get("Domain")),
        "scenario": _clean_text(row.get("Domain")),
        "label": label,
        "protocol": protocol,
        "protocol_source": protocol_source,
        "three_class_category": _resolved_three_class_category(row, label),
        "violated_constraint": "none" if is_valid else _clean_text(row.get("Violated_Constraint")),
        "violated_principle": "" if is_valid else _annotation_value(
            row, "clean_violated_principle", "final_violated_principle_resolved", "patched_violated_principle"
        ),
        "why_invalid": "" if is_valid else _annotation_value(
            row, "clean_why_invalid", "final_why_invalid_resolved", "patched_why_invalid"
        ),
        "minimal_fix": "" if is_valid else _annotation_value(
            row, "clean_minimal_fix", "final_minimal_fix_resolved", "patched_minimal_fix"
        ),
        "difficulty": _clean_text(row.get("Difficulty")) or "medium",
        "surface_plausibility": "high",
        "severity": "none" if is_valid else ("minor" if label == "minor_issue" else "fatal"),
        "source_correct_protocol": "",
        "what_changed": _clean_text(row.get("Error_Summary")),
        "review_decision": _clean_text(row.get("Decision")),
        "gold_review_decision": _clean_text(row.get("gold_review_decision")),
        "annotation_patch_status": _clean_text(row.get("annotation_patch_status")),
        "annotation_patch_source": _clean_text(row.get("annotation_patch_source")),
        "protocol_rewrite_status": _clean_text(row.get("protocol_rewrite_status")),
        "protocol_rewrite_source_mimo_id": _clean_text(row.get("protocol_rewrite_source_mimo_id")),
        "protocol_rewrite_match_key": _clean_text(row.get("protocol_rewrite_match_key")),
        "protocol_rewrite_updated_at": _clean_text(row.get("protocol_rewrite_updated_at")),
    }


def load_xlsx_rows(path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    return [dict(zip(header, values)) for values in ws.iter_rows(min_row=2, values_only=True)]


def quality_report(items: list[dict], source_rows: list[dict] | None = None) -> dict:
    protocols = [item["protocol"] for item in items]
    report = {
        "total": len(items),
        "unique_ids": len({item["id"] for item in items}),
        "unique_source_ids": len({item.get("source_id", "") for item in items if item.get("source_id")}),
        "duplicate_source_id_rows": len(items) - len({item.get("source_id", "") for item in items if item.get("source_id")}),
        "unique_protocols": len(set(protocols)),
        "duplicate_protocol_rows": len(items) - len(set(protocols)),
        "label_distribution": dict(Counter(item["label"] for item in items)),
        "domain_distribution": dict(Counter(item["domain"] for item in items)),
        "constraint_distribution": dict(Counter(item["violated_constraint"] for item in items)),
        "three_class_distribution": dict(Counter(
            item.get("three_class_category", FINE_TO_THREE_CLASS.get(item.get("violated_constraint", ""), "N/A"))
            for item in items
            if item["label"] == "fatal_violation"
        )),
        "protocol_source_distribution": dict(Counter(item["protocol_source"] for item in items)),
        "protocol_rewrite_status_distribution": dict(Counter(item.get("protocol_rewrite_status", "") for item in items)),
        "short_under_80_words": sum(1 for protocol in protocols if len(protocol.split()) < 80),
        "dirty_hits": {
            name: sum(1 for protocol in protocols if pattern.search(protocol))
            for name, pattern in DIRTY_PATTERNS.items()
        },
        "missing_nonvalid_annotations": {
            field: sum(1 for item in items if item["label"] != "valid" and not item.get(field))
            for field in ("three_class_category", "violated_principle", "why_invalid", "minimal_fix")
        },
        "valid_with_error_metadata": sum(
            1 for item in items
            if item["label"] == "valid"
            and (item.get("violated_constraint") not in ("", "none") or item.get("why_invalid") or item.get("minimal_fix"))
        ),
    }
    if source_rows is not None:
        report["source_rows"] = len(source_rows)
    return report


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def import_gold(
    source: Path,
    copy_source: bool = True,
    output_file: Path = GOLD_FULL_FILE,
    manifest_file: Path = GOLD_MANIFEST_FILE,
    source_copy_file: Path | None = None,
) -> tuple[list[dict], dict]:
    if not source.exists():
        raise FileNotFoundError(f"Gold source not found: {source}")
    GOLD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source_copy_file = source_copy_file or GOLD_SOURCE_DIR / source.name
    if copy_source:
        shutil.copy2(source, source_copy_file)

    rows = load_xlsx_rows(source)
    items = [_build_gold_item(row, idx + 1) for idx, row in enumerate(rows)]
    manifest = {
        "imported_at": datetime.now().isoformat(),
        "source_file": str(source),
        "copied_source_file": str(source_copy_file) if copy_source else "",
        "output_file": str(output_file),
        "field_mapping": {
            "id": "new stable gold_0507_###### id",
            "source_id": "ID",
            "protocol": "final_protocol > manual_rewrite > Generated_Prompt",
            "label": "Label",
            "fatal_minor_annotations": "clean_* > final_*_resolved > patched_*",
            "three_class_category": "derived from Violated_Constraint",
        },
        "quality_report": quality_report(items, rows),
    }
    save_json(output_file, items)
    save_json(manifest_file, manifest)
    return items, manifest


def _load_eval_correctness(eval_file: Path | None) -> dict[str, bool | None]:
    if not eval_file or not eval_file.exists():
        return {}
    with eval_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    correctness = {}
    for row in rows:
        item_id = row.get("id")
        result = row.get("eval_result") or row.get("stage1") or row.get("_exp_result") or {}
        correctness[item_id] = result.get("correct")
    return correctness


def build_confusing_subset(items: list[dict], eval_correctness: dict[str, bool | None], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    enriched = []
    for item in items:
        copied = dict(item)
        copied["mimo_correct"] = eval_correctness.get(item["id"])
        copied["mimo_fooled"] = eval_correctness.get(item["id"]) is False
        enriched.append(copied)

    minor = [item for item in enriched if item["label"] == "minor_issue"]
    fatal = [item for item in enriched if item["label"] == "fatal_violation"]
    valid = [item for item in enriched if item["label"] == "valid"]

    selected = list(minor)
    remaining_slots = max(0, n - len(selected))
    fatal_quota = remaining_slots // 2 + remaining_slots % 2
    valid_quota = remaining_slots // 2

    def ranked(pool: list[dict]) -> list[dict]:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return sorted(
            shuffled,
            key=lambda item: (
                item.get("mimo_correct") is not False,
                item.get("mimo_correct") is not None,
                item["domain"],
                item["id"],
            ),
        )

    selected.extend(ranked(fatal)[:fatal_quota])
    selected.extend(ranked(valid)[:valid_quota])
    if len(selected) < n:
        used = {item["id"] for item in selected}
        leftovers = [item for item in ranked(enriched) if item["id"] not in used]
        selected.extend(leftovers[: n - len(selected)])
    return selected[:n]


def _item_three_class(item: dict) -> str:
    return item.get("three_class_category") or FINE_TO_THREE_CLASS.get(item.get("violated_constraint", ""), "")


def _load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _enrich_with_previous_core(items: list[dict], previous_core: list[dict]) -> list[dict]:
    previous_by_id = {item["id"]: item for item in previous_core}
    enriched = []
    for item in items:
        copied = dict(item)
        copied["three_class_category"] = _item_three_class(copied) or "N/A"
        previous = previous_by_id.get(item["id"], {})
        for key in ("mimo_correct", "mimo_fooled"):
            if key in previous:
                copied[key] = previous[key]
        copied["previous_core_member"] = item["id"] in previous_by_id
        enriched.append(copied)
    return enriched


def build_final_core_subset(
    items: list[dict],
    previous_core: list[dict],
    label_quotas: dict[str, int] | None = None,
    fatal_three_class_quotas: dict[str, int] | None = None,
) -> list[dict]:
    """Build the paper-facing 500-row gold core with 3-class Stage 2 labels."""
    label_quotas = label_quotas or FINAL_CORE_LABEL_QUOTAS
    fatal_three_class_quotas = fatal_three_class_quotas or FINAL_CORE_FATAL_THREE_CLASS_QUOTAS
    enriched = _enrich_with_previous_core(items, previous_core)

    def status_priority(item: dict) -> int:
        return {"DONE": 0, "TODO": 1, "SKIPPED": 2, "": 3}.get(item.get("protocol_rewrite_status", ""), 4)

    def stable_rank(item: dict) -> tuple:
        return (
            item.get("mimo_fooled") is not True,
            item.get("previous_core_member") is not True,
            status_priority(item),
            item.get("domain", ""),
            item["id"],
        )

    def fatal_rank(item: dict) -> tuple:
        return (
            item.get("protocol_rewrite_status") != "DONE",
            item.get("mimo_fooled") is not True,
            item.get("previous_core_member") is not True,
            status_priority(item),
            item.get("domain", ""),
            item["id"],
        )

    selected: list[dict] = []
    selected_ids: set[str] = set()

    minor = sorted([item for item in enriched if item["label"] == "minor_issue"], key=stable_rank)
    selected.extend(minor[: label_quotas["minor_issue"]])
    selected_ids.update(item["id"] for item in selected)

    fatal = [item for item in enriched if item["label"] == "fatal_violation"]
    for category in THREE_CLASS_CATEGORIES:
        pool = sorted(
            [item for item in fatal if _item_three_class(item) == category and item["id"] not in selected_ids],
            key=fatal_rank,
        )
        quota = fatal_three_class_quotas[category]
        if len(pool) < quota:
            raise ValueError(f"Not enough fatal rows for {category}: need {quota}, found {len(pool)}")
        chosen = pool[:quota]
        selected.extend(chosen)
        selected_ids.update(item["id"] for item in chosen)

    valid = sorted(
        [item for item in enriched if item["label"] == "valid" and item["id"] not in selected_ids],
        key=stable_rank,
    )
    selected.extend(valid[: label_quotas["valid"]])
    selected_ids.update(item["id"] for item in selected)

    target_n = sum(label_quotas.values())
    if len(selected) != target_n:
        raise ValueError(f"Final core size mismatch: expected {target_n}, got {len(selected)}")
    if len(selected_ids) != len(selected):
        raise ValueError("Final core contains duplicate IDs")

    order = {"minor_issue": 0, "fatal_violation": 1, "valid": 2}
    return sorted(selected, key=lambda item: (order.get(item["label"], 9), _item_three_class(item), item["id"]))


def export_confusing_xlsx(items: list[dict], output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Gold Confusing"
    headers = [
        "id", "source_id", "domain", "label", "three_class_category", "violated_constraint",
        "difficulty", "protocol_rewrite_status", "mimo_correct", "mimo_fooled", "protocol",
        "why_invalid", "minimal_fix", "expert_action", "expert_rewrite", "review_notes",
    ]
    widths = [18, 16, 18, 16, 24, 24, 12, 18, 14, 14, 80, 60, 50, 18, 70, 50]
    header_fill = PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid")
    for idx, (header, width) in enumerate(zip(headers, widths), start=1):
        cell = ws.cell(row=1, column=idx, value=header)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        ws.column_dimensions[openpyxl.utils.get_column_letter(idx)].width = width
    wrap = Alignment(wrap_text=True, vertical="top")
    for row_idx, item in enumerate(items, start=2):
        values = [item.get(header, "") for header in headers]
        for col_idx, value in enumerate(values, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value).alignment = wrap
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{openpyxl.utils.get_column_letter(len(headers))}{len(items) + 1}"
    wb.save(output)


def import_gold_main():
    parser = argparse.ArgumentParser(description="Import partner gold xlsx into normalized JSON")
    parser.add_argument("--source", required=True, help="Partner gold xlsx source path")
    parser.add_argument("--no-copy-source", action="store_true", help="Do not copy the xlsx into data/gold/source")
    parser.add_argument("--output-json", default=str(GOLD_FULL_FILE), help="Normalized gold JSON output")
    parser.add_argument("--output-manifest", default=str(GOLD_MANIFEST_FILE), help="Import manifest output")
    args = parser.parse_args()
    output_json = Path(args.output_json)
    output_manifest = Path(args.output_manifest)
    items, manifest = import_gold(
        Path(args.source),
        copy_source=not args.no_copy_source,
        output_file=output_json,
        manifest_file=output_manifest,
    )
    print(f"Imported {len(items)} gold items -> {output_json}")
    print(json.dumps(manifest["quality_report"], ensure_ascii=False, indent=2))


def build_confusing_main():
    parser = argparse.ArgumentParser(description="Build 500-item confusing gold subset")
    parser.add_argument("--gold", default=str(GOLD_FULL_FILE), help="Normalized full gold JSON")
    parser.add_argument("--eval", default="", help="Optional eval JSON used to prioritize fooled items")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=str(CONFUSING_JSON))
    parser.add_argument("--output-xlsx", default=str(CONFUSING_XLSX))
    args = parser.parse_args()
    with Path(args.gold).open("r", encoding="utf-8") as f:
        items = json.load(f)
    eval_correctness = _load_eval_correctness(Path(args.eval) if args.eval else None)
    subset = build_confusing_subset(items, eval_correctness, args.n, args.seed)
    output_json = Path(args.output_json)
    output_xlsx = Path(args.output_xlsx)
    manifest = {
        "built_at": datetime.now().isoformat(),
        "source_gold": args.gold,
        "eval_file": args.eval,
        "target_n": args.n,
        "seed": args.seed,
        "quality_report": quality_report(subset),
        "mimo_fooled": sum(1 for item in subset if item.get("mimo_fooled")),
        "mimo_correct_known": sum(1 for item in subset if item.get("mimo_correct") is not None),
    }
    save_json(output_json, subset)
    save_json(output_json.with_name(output_json.stem + "_manifest.json"), manifest)
    export_confusing_xlsx(subset, output_xlsx)
    print(f"Built {len(subset)} confusing gold items -> {output_json}")
    print(json.dumps(manifest["quality_report"], ensure_ascii=False, indent=2))


def build_final_gold_main():
    parser = argparse.ArgumentParser(description="Build final 683-item gold set with 3-class Stage 2 labels")
    parser.add_argument("--source", required=True, help="Latest partner gold xlsx source")
    parser.add_argument("--output-json", default=str(FINAL_FULL_JSON), help="Final full gold JSON output")
    parser.add_argument("--output-manifest", default=str(FINAL_FULL_MANIFEST), help="Final full manifest output")
    parser.add_argument("--output-xlsx", default=str(FINAL_FULL_XLSX), help="Final full review XLSX output")
    parser.add_argument("--no-copy-source", action="store_true", help="Do not copy the xlsx into data/gold/source")
    args = parser.parse_args()

    items, manifest = import_gold(
        Path(args.source),
        copy_source=not args.no_copy_source,
        output_file=Path(args.output_json),
        manifest_file=Path(args.output_manifest),
    )
    manifest.update({
        "dataset_policy": "Use all 683 partner-curated gold rows. No 500-row downsampling for primary gold experiments.",
        "stage2_policy": "Use only three_class_category for Stage 2; legacy 4-class error_category is not emitted.",
        "output_xlsx": args.output_xlsx,
    })
    save_json(Path(args.output_manifest), manifest)
    export_confusing_xlsx(items, Path(args.output_xlsx))
    print(f"Built {len(items)} final gold items -> {args.output_json}")
    print(json.dumps(manifest["quality_report"], ensure_ascii=False, indent=2))


def build_final_core_main():
    parser = argparse.ArgumentParser(description="Build legacy 500-item gold core with 3-class Stage 2 labels")
    parser.add_argument("--source", required=True, help="Latest partner gold xlsx source")
    parser.add_argument("--full-json", default=str(FINAL_FULL_JSON), help="Normalized full gold JSON output")
    parser.add_argument("--full-manifest", default=str(FINAL_FULL_MANIFEST), help="Full import manifest output")
    parser.add_argument("--previous-core", default=str(PREVIOUS_CORE_JSON), help="Previous core JSON for MIMO hardness metadata")
    parser.add_argument("--output-json", default=str(FINAL_CORE_JSON), help="Final 500 core JSON output")
    parser.add_argument("--output-xlsx", default=str(FINAL_CORE_XLSX), help="Final 500 review XLSX output")
    parser.add_argument("--no-copy-source", action="store_true", help="Do not copy the xlsx into data/gold/source")
    args = parser.parse_args()

    full_items, full_manifest = import_gold(
        Path(args.source),
        copy_source=not args.no_copy_source,
        output_file=Path(args.full_json),
        manifest_file=Path(args.full_manifest),
    )
    previous_core = _load_json_list(Path(args.previous_core))
    subset = build_final_core_subset(full_items, previous_core)
    output_json = Path(args.output_json)
    output_xlsx = Path(args.output_xlsx)
    manifest = {
        "built_at": datetime.now().isoformat(),
        "source_file": args.source,
        "source_full_gold": args.full_json,
        "previous_core": args.previous_core,
        "output_file": args.output_json,
        "selection_policy": (
            "Final 500 keeps 74 minor, 213 fatal, and 213 valid rows. "
            "Fatal rows are selected by 3-class quotas, with protocol_rewrite_status=DONE prioritized before older rows."
        ),
        "label_quotas": FINAL_CORE_LABEL_QUOTAS,
        "fatal_three_class_quotas": FINAL_CORE_FATAL_THREE_CLASS_QUOTAS,
        "quality_report_full": full_manifest["quality_report"],
        "quality_report": quality_report(subset),
        "previous_core_retained": sum(1 for item in subset if item.get("previous_core_member")),
        "protocol_rewrite_done_rows": sum(1 for item in subset if item.get("protocol_rewrite_status") == "DONE"),
        "fatal_protocol_rewrite_done_by_three_class": dict(Counter(
            item.get("three_class_category")
            for item in subset
            if item["label"] == "fatal_violation" and item.get("protocol_rewrite_status") == "DONE"
        )),
    }
    save_json(output_json, subset)
    save_json(output_json.with_name(output_json.stem + "_manifest.json"), manifest)
    export_confusing_xlsx(subset, output_xlsx)
    print(f"Built {len(subset)} final gold core items -> {output_json}")
    print(json.dumps(manifest["quality_report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import_gold_main()
