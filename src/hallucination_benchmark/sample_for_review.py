#!/usr/bin/env python3
"""
Human Review Sampler (v2)
=========================
Stratified sampling with pro-fooling priority.
Prioritizes items that mimo-v2.5-pro failed to detect.

Usage:
  python3 sample_for_review.py
  python3 sample_for_review.py --n 800 --fooling-ratio 0.7
"""

import argparse
import json
import os
import random
from collections import defaultdict

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

from hallucination_benchmark.config import CURRENT_DATA_DIR

DATA_FILE = os.path.join(CURRENT_DATA_DIR, "benchmark_balanced.json")
EVAL_FILE = os.path.join(CURRENT_DATA_DIR, "benchmark_balanced_eval_L2_mimo-v2.5-pro.json")
DEFAULT_OUTPUT = os.path.join(CURRENT_DATA_DIR, "review_800.xlsx")


def load_eval_results(eval_file):
    """Load eval results and build {id: correct} map."""
    if not os.path.exists(eval_file):
        print(f"  WARNING: {eval_file} not found, no pro-fooling priority")
        return {}
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_items = json.load(f)
    result = {}
    for item in eval_items:
        item_id = item.get("id", "")
        eval_result = item.get("eval_result", {})
        result[item_id] = eval_result.get("correct", None)
    return result


def stratified_sample(items, eval_map, n=800, fooling_ratio=0.7, seed=42):
    """Stratified sampling with pro-fooling priority for fatal items.
    
    For fatal_violation:
      - fooling_ratio% from items where pro was FOOLED (high quality)
      - (1-fooling_ratio)% from items where pro DETECTED (for diversity)
    For valid/minor: standard proportional sampling.
    """
    random.seed(seed)

    # Tag items with pro-fooling status
    for item in items:
        item["_pro_correct"] = eval_map.get(item["id"], None)

    # Separate fatal vs non-fatal
    fatal_items = [i for i in items if i["label"] == "fatal_violation"]
    non_fatal_items = [i for i in items if i["label"] != "fatal_violation"]

    # Target counts (maintain distribution)
    label_counts = defaultdict(int)
    for item in items:
        label_counts[item["label"]] += 1
    total = len(items)

    target_fatal = round(n * label_counts["fatal_violation"] / total)
    target_valid = round(n * label_counts["valid"] / total)
    target_minor = n - target_fatal - target_valid

    print(f"  Targets: fatal={target_fatal}, valid={target_valid}, minor={target_minor}")

    # --- Sample fatal: stratified by (category, constraint), with pro-fooling priority ---
    fatal_groups = defaultdict(lambda: {"fooled": [], "detected": []})
    for item in fatal_items:
        cat = item.get("error_category", "unknown")
        con = item.get("violated_constraint", "unknown")
        pro_correct = item.get("_pro_correct")
        if pro_correct is False:  # pro was fooled
            fatal_groups[(cat, con)]["fooled"].append(item)
        else:  # pro detected or unknown
            fatal_groups[(cat, con)]["detected"].append(item)

    sampled_fatal = []
    print(f"\n  Fatal sampling (pro-fooling priority, ratio={fooling_ratio}):")
    print(f"  {'category':35s} {'constraint':20s} {'fooled':>6s} {'detect':>6s} {'quota':>6s} {'q_fool':>6s}")
    print(f"  {'-'*35} {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for (cat, con), group in sorted(fatal_groups.items()):
        fooled = group["fooled"]
        detected = group["detected"]
        pool_size = len(fooled) + len(detected)
        proportion = pool_size / len(fatal_items)
        quota = max(1, round(proportion * target_fatal))
        if quota > pool_size:
            quota = pool_size

        # Split quota: fooling_ratio from fooled, rest from detected
        q_fooled = min(round(quota * fooling_ratio), len(fooled))
        q_detected = min(quota - q_fooled, len(detected))

        # If not enough fooled, fill from detected
        if q_fooled + q_detected < quota:
            q_detected = min(quota - q_fooled, len(detected))

        # If not enough detected, fill from fooled
        if q_fooled + q_detected < quota:
            q_fooled = min(quota - q_detected, len(fooled))

        selected = []
        if q_fooled > 0:
            selected.extend(random.sample(fooled, q_fooled))
        if q_detected > 0:
            selected.extend(random.sample(detected, q_detected))

        sampled_fatal.extend(selected)
        print(f"  {cat:35s} {con:20s} {len(fooled):6d} {len(detected):6d} {quota:6d} {q_fooled:6d}")

    # --- Sample non-fatal: proportional by label ---
    valid_items = [i for i in non_fatal_items if i["label"] == "valid"]
    minor_items = [i for i in non_fatal_items if i["label"] == "minor_issue"]

    sampled_valid = random.sample(valid_items, min(target_valid, len(valid_items)))
    sampled_minor = random.sample(minor_items, min(target_minor, len(minor_items)))

    # Combine
    sampled = sampled_fatal + sampled_valid + sampled_minor

    # Adjust to exact n
    if len(sampled) > n:
        sampled = random.sample(sampled, n)
    elif len(sampled) < n:
        used_ids = {i["id"] for i in sampled}
        leftover = [i for i in items if i["id"] not in used_ids]
        if leftover:
            extra = random.sample(leftover, min(n - len(sampled), len(leftover)))
            sampled.extend(extra)

    # Stats
    fooled_count = sum(1 for i in sampled if i.get("_pro_correct") is False and i["label"] == "fatal_violation")
    detected_count = sum(1 for i in sampled if i.get("_pro_correct") is True and i["label"] == "fatal_violation")
    fatal_total = sum(1 for i in sampled if i["label"] == "fatal_violation")

    print(f"\n  Final: {len(sampled)} items")
    print(f"  Fatal: {fatal_total} (pro-fooled: {fooled_count}, pro-detected: {detected_count})")
    print(f"  Valid: {sum(1 for i in sampled if i['label'] == 'valid')}")
    print(f"  Minor: {sum(1 for i in sampled if i['label'] == 'minor_issue')}")

    return sampled


def export_xlsx(items, output_path):
    """Export to xlsx with review columns."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Review"

    headers = [
        ("A", "ID", 15),
        ("B", "Domain", 18),
        ("C", "Label", 15),
        ("D", "Error_Category", 30),
        ("E", "Violated_Constraint", 25),
        ("F", "Difficulty", 10),
        ("G", "Pro_Fooled", 12),
        ("H", "Generated_Prompt", 80),
        ("I", "Target_Error", 60),
        ("J", "Expert_Action", 18),
        ("K", "Expert_Rewrite", 60),
        ("L", "Refined_Category", 25),
    ]

    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    wrap_align = Alignment(wrap_text=True, vertical="top")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    for col_letter, name, width in headers:
        col_idx = openpyxl.utils.column_index_from_string(col_letter)
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[col_letter].width = width

    for row_idx, item in enumerate(items, start=2):
        target_parts = []
        if item.get("violated_principle"):
            target_parts.append(item["violated_principle"])
        if item.get("why_invalid"):
            target_parts.append(item["why_invalid"])
        if item.get("minimal_fix"):
            target_parts.append(f"Fix: {item['minimal_fix']}")
        target_error = "\n".join(target_parts) if target_parts else ""

        pro_correct = item.get("_pro_correct")
        if pro_correct is False:
            pro_fooled = "YES"
        elif pro_correct is True:
            pro_fooled = "no"
        else:
            pro_fooled = "N/A"

        row_data = [
            item.get("id", ""),
            item.get("domain", ""),
            item.get("label", ""),
            item.get("error_category", ""),
            item.get("violated_constraint", ""),
            item.get("difficulty", ""),
            pro_fooled,
            item.get("protocol", ""),
            target_error,
            "",  # Expert_Action
            "",  # Expert_Rewrite
            "",  # Refined_Category
        ]

        for col_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = wrap_align
            cell.border = thin_border

    from openpyxl.worksheet.datavalidation import DataValidation
    dv = DataValidation(
        type="list",
        formula1='"A-直接录用,B-修改文本,C-丢弃"',
        allow_blank=True,
    )
    dv.error = "Please select A, B, or C"
    dv.errorTitle = "Invalid action"
    ws.add_data_validation(dv)
    for row_idx in range(2, len(items) + 2):
        dv.add(ws.cell(row=row_idx, column=10))

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:L{len(items) + 1}"

    wb.save(output_path)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample for human review")
    parser.add_argument("--data", default=DATA_FILE, help="Input dataset")
    parser.add_argument("--eval", default=EVAL_FILE, help="Eval results file")
    parser.add_argument("--n", type=int, default=800, help="Number of items to sample")
    parser.add_argument("--fooling-ratio", type=float, default=0.7,
                        help="Ratio of pro-fooled items in fatal quota (default: 0.7)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output xlsx path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {args.data}")

    eval_map = load_eval_results(args.eval)
    print(f"Loaded {len(eval_map)} eval results from {args.eval}")

    sampled = stratified_sample(items, eval_map, n=args.n, fooling_ratio=args.fooling_ratio, seed=args.seed)
    export_xlsx(sampled, args.output)


if __name__ == "__main__":
    main()
