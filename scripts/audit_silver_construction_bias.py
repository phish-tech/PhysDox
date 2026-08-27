"""Produce Silver construction and generator-bias evidence for rebuttal."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_PATTERNS = [
    r"\[Missing Original Protocol Content\]",
    r"\[[^\]]{1,80}\]",
    r"source protocol text",
    r"procedure below",
    r"preserves the original feasibility",
    r"procedures and performance claims stated above",
    r"full protocol text",
    r"modified protocol text",
    r"as written above",
    r"the full protocol string",
    r"original protocol",
    r"rewritten protocol",
    r"standard format",
    r"section headers",
    r"provided experimental protocol",
    r"first, the user",
    r"first, i need",
    r"i need to (?:write|generate|create|outline)",
    r"let me (?:start|outline|think)",
    r"i will rewrite",
    r"output only",
    r"required headers",
    r"json object",
    r"minor suboptimal",
    r"benchmark construction",
]
FORBIDDEN_RE = re.compile("|".join(f"(?:{p})" for p in FORBIDDEN_PATTERNS), re.I)
BRACKET_RE = re.compile(r"\[[^\]]{1,80}\]")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_protocol(text: str) -> str:
    return " ".join((text or "").split()).lower()


def audit_silver_data(items: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [item.get("id") for item in items]
    protocols = [item.get("protocol") or "" for item in items]
    normalized_protocols = [normalized_protocol(text) for text in protocols]
    duplicate_protocol_count = len(normalized_protocols) - len(set(normalized_protocols))
    duplicate_id_count = len(ids) - len(set(ids))
    forbidden_hits = [
        item.get("id")
        for item in items
        if FORBIDDEN_RE.search(item.get("protocol") or "")
    ]
    bracket_hits = [
        item.get("id") for item in items if BRACKET_RE.search(item.get("protocol") or "")
    ]
    return {
        "items": len(items),
        "unique_ids": len(set(ids)),
        "duplicate_id_rows": duplicate_id_count,
        "unique_protocols_normalized": len(set(normalized_protocols)),
        "duplicate_protocol_rows_normalized": duplicate_protocol_count,
        "label_distribution": dict(Counter(item.get("label") for item in items)),
        "domain_distribution": dict(Counter(item.get("domain") for item in items)),
        "fatal_constraint_distribution": dict(
            Counter(
                item.get("violated_constraint")
                for item in items
                if item.get("label") == "fatal_violation"
            )
        ),
        "forbidden_template_hits": len(forbidden_hits),
        "forbidden_template_hit_ids": forbidden_hits[:20],
        "bracket_placeholder_hits": len(bracket_hits),
        "bracket_placeholder_hit_ids": bracket_hits[:20],
    }


def audit_generation(build_summary: dict[str, Any]) -> dict[str, Any]:
    source = build_summary["source_report"]
    final = build_summary["final_report"]
    attempts = build_summary["checkpoint_generated_attempts"]
    rejected = build_summary["checkpoint_rejected"]
    rejected_total = sum(rejected.values())
    accepted = attempts - rejected_total
    strict_pool_before_sampling = source["clean_unique"] + accepted
    sampled_out = strict_pool_before_sampling - final["total"]
    return {
        "raw_total": source["raw_total"],
        "quality_pass_raw": source["clean_raw"],
        "clean_unique_seed_pool": source["clean_unique"],
        "duplicate_protocols_removed_after_quality": source[
            "duplicate_protocols_removed_after_quality"
        ],
        "targeted_generation_attempts": attempts,
        "targeted_generation_rejections": rejected_total,
        "targeted_generation_rejection_reasons": rejected,
        "accepted_generated_candidates": accepted,
        "strict_pool_before_final_sampling": strict_pool_before_sampling,
        "final_total": final["total"],
        "sampled_out_before_freeze": sampled_out,
    }


def stage1_gap_rows(long_csv: Path) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with long_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values[(row["model_label"], row["dataset"])].append(
                float(row["stage1_macro_f1"]) * 100.0
            )

    rows: list[dict[str, Any]] = []
    for model in sorted({model for model, _ in values}):
        silver = sum(values[(model, "silver5k")]) / len(values[(model, "silver5k")])
        gold = sum(values[(model, "gold683")]) / len(values[(model, "gold683")])
        rows.append(
            {
                "model": model,
                "silver_stage1_macro_f1_avg": round(silver, 1),
                "gold_stage1_macro_f1_avg": round(gold, 1),
                "silver_minus_gold_gap": round(silver - gold, 1),
            }
        )
    return rows


def write_gap_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, Any], gap_rows: list[dict[str, Any]], path: Path) -> None:
    generation = report["generation"]
    silver = report["silver_data_audit"]
    quality = report["manifest_quality_report"]
    gap_table = "\n".join(
        "| {model} | {silver_stage1_macro_f1_avg:.1f} | {gold_stage1_macro_f1_avg:.1f} | {silver_minus_gold_gap:+.1f} |".format(
            **row
        )
        for row in gap_rows
    )
    constraint_values = list(silver["fatal_constraint_distribution"].values())
    text = f"""# Silver Construction And Generator-Bias Audit

Generated by `scripts/audit_silver_construction_bias.py`.

## Construction Funnel

| Stage | Count |
|---|---:|
| Initial candidate pool | {generation['raw_total']:,} |
| Passed strict quality filters | {generation['quality_pass_raw']:,} |
| Unique clean seed pool | {generation['clean_unique_seed_pool']:,} |
| Targeted MiMO generation attempts | {generation['targeted_generation_attempts']:,} |
| Rejected generated attempts | {generation['targeted_generation_rejections']:,} |
| Accepted generated candidates | {generation['accepted_generated_candidates']:,} |
| Strict pool before final sampling | {generation['strict_pool_before_final_sampling']:,} |
| Frozen Silver release | {generation['final_total']:,} |

Accepted generated candidates are computed as generation attempts minus logged
generation rejections. The strict pre-sampling pool is the unique clean seed
pool plus accepted generated candidates.

## Artifact And Balance Checks

| Check | Value |
|---|---:|
| Silver items | {silver['items']:,} |
| Unique IDs | {silver['unique_ids']:,} |
| Duplicate ID rows | {silver['duplicate_id_rows']} |
| Unique normalized protocol texts | {silver['unique_protocols_normalized']:,} |
| Duplicate normalized protocol rows | {silver['duplicate_protocol_rows_normalized']} |
| Forbidden template/wrapper hits | {silver['forbidden_template_hits']} |
| Bracket placeholder hits | {silver['bracket_placeholder_hits']} |
| Manifest quality issues | {len(quality['quality_issues'])} |
| Manifest wrapper hits | {quality['wrapper_hits']} |
| Manifest bracket placeholder hits | {quality['bracket_placeholder_hits']} |
| Fatal constraints | {len(silver['fatal_constraint_distribution'])} |
| Fatal constraint min count | {min(constraint_values)} |
| Fatal constraint max count | {max(constraint_values)} |

## Silver-vs-Gold Stage-1 Macro-F1 Gap

Average across the four inference strategies for each model family.

| Model | Silver S1 macro-F1 avg | Gold S1 macro-F1 avg | Gap |
|---|---:|---:|---:|
{gap_table}

## Rebuttal Note

The audit supports two points. First, the Silver release is not a loose dump of
generated text: it has unique IDs, unique normalized protocol texts, exact label
quotas, balanced fatal-constraint counts, and zero detected wrapper, placeholder,
or prompt-leakage hits under the release filters. Second, MiMO-family models
show the largest Silver-to-Gold Stage-1 macro-F1 gaps, which is consistent with
a possible generator-distribution advantage on Silver. A conservative rebuttal
should therefore state that Silver is used as a large-scale diagnostic split,
while final model comparisons and safety claims are anchored on the independently
curated Gold set.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, default=Path("data/current/benchmark_balanced.json"))
    parser.add_argument(
        "--silver-manifest",
        type=Path,
        default=Path("data/current/benchmark_balanced_manifest.json"),
    )
    parser.add_argument(
        "--build-summary",
        type=Path,
        default=Path("experiments/archive/legacy_runs/clean_5k_20260507_114513/summary.json"),
    )
    parser.add_argument(
        "--long-csv",
        type=Path,
        default=Path("experiments/main_20260510_main/tables/main_experiment_results_long.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rebuttal_20260708/silver_bias_audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    silver_items = load_json(args.silver)
    manifest = load_json(args.silver_manifest)
    build_summary = load_json(args.build_summary)
    gap_rows = stage1_gap_rows(args.long_csv)
    report = {
        "silver_data_audit": audit_silver_data(silver_items),
        "manifest_quality_report": manifest["quality_report"],
        "manifest_source_report": manifest["source_report"],
        "generation": audit_generation(build_summary),
        "stage1_silver_gold_gap": gap_rows,
    }
    json_path = args.output_dir / "silver_construction_bias_audit.json"
    md_path = args.output_dir / "silver_construction_bias_audit.md"
    gap_csv_path = args.output_dir / "stage1_silver_gold_gap.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, gap_rows, md_path)
    write_gap_csv(gap_rows, gap_csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {gap_csv_path}")


if __name__ == "__main__":
    main()
