#!/usr/bin/env python3
"""Export the main experiment matrix into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


DEFAULT_ROOT = Path("experiments/main_20260510_main")
DEFAULT_OUTPUT = DEFAULT_ROOT / "tables"

DATASET_ORDER = ["silver5k", "gold683"]
DATASET_LABELS = {
    "silver5k": "Silver 5k",
    "gold683": "Gold 683",
}
STRATEGY_ORDER = ["zero-shot", "cot", "self-cons", "tool-aug"]
STRATEGY_LABELS = {
    "zero-shot": "Zero-Shot",
    "cot": "CoT",
    "self-cons": "Self-Cons",
    "tool-aug": "Tool-Aug",
}
MODEL_ORDER = [
    "mimo-v2.5-pro",
    "mimo-v2.5",
    "claude-opus-4-7",
    "gpt-5.5",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]
MODEL_LABELS = {
    "mimo-v2.5-pro": "MiMO v2.5 Pro",
    "mimo-v2.5": "MiMO v2.5",
    "claude-opus-4-7": "Claude Opus 4.7",
    "gpt-5.5": "GPT-5.5",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}
MODEL_FAMILIES = {
    "mimo-v2.5-pro": "MiMO",
    "mimo-v2.5": "MiMO",
    "claude-opus-4-7": "Claude",
    "gpt-5.5": "GPT",
    "deepseek-v4-flash": "DeepSeek",
    "deepseek-v4-pro": "DeepSeek",
}

STAGE1_CLASSES = ["valid", "minor_issue", "fatal_violation"]
THREE_CLASS_CLASSES = [
    "Signal & Sampling",
    "Perception & Anatomy",
    "Physical / Hardware",
]
FINE12_CLASSES = [
    "sampling_theorem",
    "sensor_bandwidth",
    "adc_resolution",
    "snr_floor",
    "latency_constraint",
    "data_bandwidth",
    "anatomical_misplacement",
    "optical_obstruction",
    "power_budget",
    "temperature_range",
    "accuracy_impossibility",
    "ematerial_block",
]

STAGE2_SCOPES = ["oracle", "gated", "end_to_end"]


def confusion_matrix(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    matrix = {label: {pred: 0 for pred in labels + ["__missing__"]} for label in labels}
    for true, pred in zip(y_true, y_pred):
        if true not in matrix:
            continue
        key = pred if pred in labels else "__missing__"
        matrix[true][key] += 1
    return matrix


def classification_metrics(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    per_class = {}
    recalls = []
    f1s = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        support = sum(1 for true in y_true if true == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / len(labels) if labels else 0.0,
        "macro_f1": sum(f1s) / len(labels) if labels else 0.0,
        "missing_predictions": sum(1 for pred in y_pred if pred not in labels),
        "prediction_distribution": dict(Counter(pred if pred in labels else "__missing__" for pred in y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
    }


def top_k_accuracy(y_true: list[str], y_pred_ranked: list[list[str]], labels: Iterable[str], k: int) -> dict:
    allowed = set(labels)
    hits = 0
    total = len(y_true)
    for true, ranked in zip(y_true, y_pred_ranked):
        cleaned = [pred for pred in ranked if pred in allowed]
        if true in cleaned[:k]:
            hits += 1
    return {"k": k, "total": total, "hits": hits, "accuracy": hits / total if total else 0.0}


def dataset_key(data_file: str) -> str:
    if "gold_0510_final_full" in data_file:
        return "gold683"
    if "benchmark_balanced" in data_file:
        return "silver5k"
    raise ValueError(f"Unknown dataset file: {data_file}")


def order_index(value: str, order: list[str]) -> int:
    try:
        return order.index(value)
    except ValueError:
        return len(order)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def metric(stage: dict[str, Any] | None, key: str) -> Any:
    if not stage:
        return None
    return stage.get(key)


def topk(stage: dict[str, Any] | None, key: str) -> Any:
    if not stage:
        return None
    value = stage.get(key)
    if isinstance(value, dict):
        return value.get("accuracy")
    return None


def class_metric(stage: dict[str, Any] | None, class_name: str, key: str) -> Any:
    if not stage:
        return None
    stats = (stage.get("per_class") or {}).get(class_name) or {}
    return stats.get(key)


def _load_per_item(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "per_item_results.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stage2_metrics(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    taxonomy: str,
) -> dict[str, Any]:
    if taxonomy == "three":
        labels = THREE_CLASS_CLASSES
        ref_key = "three_class_reference"
        pred_key = "three_class_prediction"
        ranked_key = "three_class_ranked"
        topk_key = "top2_accuracy"
        topk = 2
    elif taxonomy == "fine":
        labels = FINE12_CLASSES
        ref_key = "fine_reference"
        pred_key = "fine_prediction"
        ranked_key = "fine_ranked"
        topk_key = "top3_accuracy"
        topk = 3
    else:
        raise ValueError(f"Unknown taxonomy: {taxonomy}")

    y_true: list[str] = []
    y_pred: list[str | None] = []
    ranked: list[list[str]] = []
    for row in rows:
        if row.get("label") != "fatal_violation":
            continue
        stage1 = row.get("stage1") or {}
        stage2 = row.get("stage2") or {}
        detected_fatal = stage1.get("prediction") == "fatal_violation"
        if scope == "gated" and not detected_fatal:
            continue
        ref = stage2.get(ref_key)
        if not ref:
            ref = row.get("three_class_category") if taxonomy == "three" else row.get("violated_constraint")
        if not ref or ref not in labels:
            continue
        y_true.append(ref)
        if scope == "end_to_end" and not detected_fatal:
            y_pred.append(None)
            ranked.append([])
        else:
            y_pred.append(stage2.get(pred_key))
            ranked.append(stage2.get(ranked_key) or [])

    metrics = classification_metrics(y_true, y_pred, labels)
    metrics[topk_key] = top_k_accuracy(y_true, ranked, labels, topk)
    return metrics


def add_gated_metrics(summary: dict[str, Any]) -> None:
    run_dir = Path(summary["_run_dir"])
    rows = _load_per_item(run_dir)
    true_fatal = [row for row in rows if row.get("label") == "fatal_violation"]
    detected_fatal = [
        row
        for row in true_fatal
        if (row.get("stage1") or {}).get("prediction") == "fatal_violation"
    ]
    summary["stage2_gate"] = {
        "oracle_fatal_total": len(true_fatal),
        "stage1_detected_fatal_total": len(detected_fatal),
        "fatal_detection_recall": len(detected_fatal) / len(true_fatal) if true_fatal else 0.0,
    }
    summary["stage2_three_class_oracle"] = summary.get("stage2_three_class_categories")
    summary["stage2_fine_oracle"] = summary.get("stage2_fine_constraints")
    summary["stage2_three_class_gated"] = _stage2_metrics(rows, scope="gated", taxonomy="three")
    summary["stage2_fine_gated"] = _stage2_metrics(rows, scope="gated", taxonomy="fine")
    summary["stage2_three_class_end_to_end"] = _stage2_metrics(rows, scope="end_to_end", taxonomy="three")
    summary["stage2_fine_end_to_end"] = _stage2_metrics(rows, scope="end_to_end", taxonomy="fine")


def load_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        data["_summary_path"] = str(path)
        data["_run_dir"] = str(path.parent)
        data["_dataset_key"] = dataset_key(data["data_file"])
        add_gated_metrics(data)
        rows.append(data)
    rows.sort(
        key=lambda row: (
            order_index(row["_dataset_key"], DATASET_ORDER),
            order_index(row["model"], MODEL_ORDER),
            order_index(row["strategy"], STRATEGY_ORDER),
            row["experiment_id"],
        )
    )
    return rows


def build_long_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        s1 = summary.get("stage1_problem_detection")
        gate = summary.get("stage2_gate") or {}
        row = {
            "dataset": summary["_dataset_key"],
            "dataset_label": DATASET_LABELS[summary["_dataset_key"]],
            "model_family": MODEL_FAMILIES.get(summary["model"], summary["model"]),
            "model": summary["model"],
            "model_label": MODEL_LABELS.get(summary["model"], summary["model"]),
            "strategy": summary["strategy"],
            "strategy_label": STRATEGY_LABELS.get(summary["strategy"], summary["strategy"]),
            "experiment_id": summary["experiment_id"],
            "items": summary.get("items"),
            "fatal_items_for_stage2": summary.get("fatal_items_for_stage2"),
            "self_cons_samples": summary.get("self_cons_samples"),
            "completed_at": summary.get("completed_at"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "stage1_total": metric(s1, "total"),
            "stage1_correct": metric(s1, "correct"),
            "stage1_accuracy": metric(s1, "accuracy"),
            "stage1_balanced_accuracy": metric(s1, "balanced_accuracy"),
            "stage1_macro_f1": metric(s1, "macro_f1"),
            "stage1_missing_predictions": metric(s1, "missing_predictions"),
            "stage2_oracle_fatal_total": gate.get("oracle_fatal_total"),
            "stage2_gated_fatal_total": gate.get("stage1_detected_fatal_total"),
            "stage2_gate_fatal_detection_recall": gate.get("fatal_detection_recall"),
        }
        for scope in STAGE2_SCOPES:
            s2_3 = summary.get(f"stage2_three_class_{scope}")
            s2_fine = summary.get(f"stage2_fine_{scope}")
            row.update({
                f"stage2_{scope}_three_total": metric(s2_3, "total"),
                f"stage2_{scope}_three_correct": metric(s2_3, "correct"),
                f"stage2_{scope}_three_accuracy": metric(s2_3, "accuracy"),
                f"stage2_{scope}_three_balanced_accuracy": metric(s2_3, "balanced_accuracy"),
                f"stage2_{scope}_three_macro_f1": metric(s2_3, "macro_f1"),
                f"stage2_{scope}_three_top2_accuracy": topk(s2_3, "top2_accuracy"),
                f"stage2_{scope}_three_missing_predictions": metric(s2_3, "missing_predictions"),
                f"stage2_{scope}_fine_total": metric(s2_fine, "total"),
                f"stage2_{scope}_fine_correct": metric(s2_fine, "correct"),
                f"stage2_{scope}_fine_accuracy": metric(s2_fine, "accuracy"),
                f"stage2_{scope}_fine_balanced_accuracy": metric(s2_fine, "balanced_accuracy"),
                f"stage2_{scope}_fine_macro_f1": metric(s2_fine, "macro_f1"),
                f"stage2_{scope}_fine_top3_accuracy": topk(s2_fine, "top3_accuracy"),
                f"stage2_{scope}_fine_missing_predictions": metric(s2_fine, "missing_predictions"),
            })
        for cls in STAGE1_CLASSES:
            safe = cls.replace(" ", "_").replace("/", "").lower()
            row[f"stage1_{safe}_support"] = class_metric(s1, cls, "support")
            row[f"stage1_{safe}_precision"] = class_metric(s1, cls, "precision")
            row[f"stage1_{safe}_recall"] = class_metric(s1, cls, "recall")
            row[f"stage1_{safe}_f1"] = class_metric(s1, cls, "f1")
        for cls in THREE_CLASS_CLASSES:
            safe = cls.lower().replace(" & ", "_").replace(" / ", "_").replace(" ", "_")
            for scope in STAGE2_SCOPES:
                s2_3 = summary.get(f"stage2_three_class_{scope}")
                row[f"stage2_{scope}_three_{safe}_support"] = class_metric(s2_3, cls, "support")
                row[f"stage2_{scope}_three_{safe}_precision"] = class_metric(s2_3, cls, "precision")
                row[f"stage2_{scope}_three_{safe}_recall"] = class_metric(s2_3, cls, "recall")
                row[f"stage2_{scope}_three_{safe}_f1"] = class_metric(s2_3, cls, "f1")
        rows.append(row)
    return rows


def build_wide_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["strategy"], row["dataset"]): row
        for row in long_rows
    }
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            out: dict[str, Any] = {
                "model_family": MODEL_FAMILIES.get(model, model),
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
            }
            for dataset in DATASET_ORDER:
                src = by_key[(model, strategy, dataset)]
                prefix = dataset
                keys = [
                    "items",
                    "fatal_items_for_stage2",
                    "stage1_total",
                    "stage1_correct",
                    "stage1_accuracy",
                    "stage1_balanced_accuracy",
                    "stage1_macro_f1",
                    "stage2_oracle_fatal_total",
                    "stage2_gated_fatal_total",
                    "stage2_gate_fatal_detection_recall",
                ]
                for scope in STAGE2_SCOPES:
                    keys.extend([
                        f"stage2_{scope}_three_total",
                        f"stage2_{scope}_three_correct",
                        f"stage2_{scope}_three_accuracy",
                        f"stage2_{scope}_three_balanced_accuracy",
                        f"stage2_{scope}_three_macro_f1",
                        f"stage2_{scope}_three_top2_accuracy",
                        f"stage2_{scope}_fine_total",
                        f"stage2_{scope}_fine_correct",
                        f"stage2_{scope}_fine_accuracy",
                        f"stage2_{scope}_fine_balanced_accuracy",
                        f"stage2_{scope}_fine_macro_f1",
                        f"stage2_{scope}_fine_top3_accuracy",
                    ])
                for key in keys:
                    out[f"{prefix}_{key}"] = src.get(key)
            rows.append(out)
    return rows


def build_class_rows(
    summaries: list[dict[str, Any]],
    stage_key: str,
    classes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        stage = summary.get(stage_key) or {}
        for class_name in classes:
            stats = (stage.get("per_class") or {}).get(class_name) or {}
            rows.append({
                "dataset": summary["_dataset_key"],
                "dataset_label": DATASET_LABELS[summary["_dataset_key"]],
                "model_family": MODEL_FAMILIES.get(summary["model"], summary["model"]),
                "model": summary["model"],
                "model_label": MODEL_LABELS.get(summary["model"], summary["model"]),
                "strategy": summary["strategy"],
                "strategy_label": STRATEGY_LABELS.get(summary["strategy"], summary["strategy"]),
                "experiment_id": summary["experiment_id"],
                "class": class_name,
                "support": stats.get("support"),
                "tp": stats.get("tp"),
                "fp": stats.get("fp"),
                "fn": stats.get("fn"),
                "precision": stats.get("precision"),
                "recall": stats.get("recall"),
                "f1": stats.get("f1"),
            })
    return rows


def build_scoped_stage2_class_rows(
    summaries: list[dict[str, Any]],
    metric_prefix: str,
    classes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for scope in STAGE2_SCOPES:
            stage = summary.get(f"{metric_prefix}_{scope}") or {}
            for class_name in classes:
                stats = (stage.get("per_class") or {}).get(class_name) or {}
                rows.append({
                    "dataset": summary["_dataset_key"],
                    "dataset_label": DATASET_LABELS[summary["_dataset_key"]],
                    "model_family": MODEL_FAMILIES.get(summary["model"], summary["model"]),
                    "model": summary["model"],
                    "model_label": MODEL_LABELS.get(summary["model"], summary["model"]),
                    "strategy": summary["strategy"],
                    "strategy_label": STRATEGY_LABELS.get(summary["strategy"], summary["strategy"]),
                    "experiment_id": summary["experiment_id"],
                    "evaluation_scope": scope,
                    "class": class_name,
                    "support": stats.get("support"),
                    "tp": stats.get("tp"),
                    "fp": stats.get("fp"),
                    "fn": stats.get("fn"),
                    "precision": stats.get("precision"),
                    "recall": stats.get("recall"),
                    "f1": stats.get("f1"),
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(wide_rows: list[dict[str, Any]], *, scope: str = "end_to_end") -> str:
    if scope == "end_to_end":
        title = "Main Experiment Results (End-to-End Stage 2)"
        scope_note = "Stage 2 in this compact table is end-to-end: true fatal samples missed by Stage 1 count as Stage 2 wrong."
        filename_note = "CSV/XLSX companion files also include the older oracle-fatal and conditional gated Stage 2 views."
        stage_label = "E2E"
    elif scope == "gated":
        title = "Main Experiment Results (Gated Stage 2)"
        scope_note = "Stage 2 in this compact table is conditional gated: only true fatal samples that Stage 1 predicted as fatal are evaluated."
        filename_note = "The Fatal Gate column is the gated Stage 2 denominator."
        stage_label = "Gated"
    elif scope == "oracle":
        title = "Main Experiment Results (Oracle-Fatal Stage 2)"
        scope_note = "Stage 2 in this compact table uses the older oracle-fatal view: all true fatal samples are evaluated regardless of Stage 1 prediction."
        filename_note = "Use this only when explicitly describing an oracle routing setting."
        stage_label = "Oracle"
    else:
        raise ValueError(f"Unknown markdown scope: {scope}")

    header = [
        "Model",
        "Strategy",
        "Silver S1 Acc/BAcc/F1",
        "Silver Fatal Gate",
        f"Silver S2 {stage_label} 3C Acc/BAcc/F1/Top2",
        f"Silver S2 {stage_label} Fine Acc/BAcc/F1/Top3",
        "Gold S1 Acc/BAcc/F1",
        "Gold Fatal Gate",
        f"Gold S2 {stage_label} 3C Acc/BAcc/F1/Top2",
        f"Gold S2 {stage_label} Fine Acc/BAcc/F1/Top3",
    ]
    lines = [
        f"# {title}",
        "",
        "Stage 1 is problem detection over `valid`, `minor_issue`, and `fatal_violation`.",
        scope_note,
        filename_note,
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in wide_rows:
        def triple(prefix: str, stage: str) -> str:
            return " / ".join([
                pct(row[f"{prefix}_{stage}_accuracy"]),
                pct(row[f"{prefix}_{stage}_balanced_accuracy"]),
                pct(row[f"{prefix}_{stage}_macro_f1"]),
            ])

        def gate(prefix: str) -> str:
            return (
                f"{row[f'{prefix}_stage2_gated_fatal_total']}/"
                f"{row[f'{prefix}_stage2_oracle_fatal_total']} "
                f"({pct(row[f'{prefix}_stage2_gate_fatal_detection_recall'])})"
            )

        def quad(prefix: str, stage: str, topk_name: str) -> str:
            return " / ".join([
                pct(row[f"{prefix}_{stage}_accuracy"]),
                pct(row[f"{prefix}_{stage}_balanced_accuracy"]),
                pct(row[f"{prefix}_{stage}_macro_f1"]),
                pct(row[f"{prefix}_{stage}_{topk_name}"]),
            ])

        values = [
            row["model_label"],
            row["strategy_label"],
            triple("silver5k", "stage1"),
            gate("silver5k"),
            quad("silver5k", f"stage2_{scope}_three", "top2_accuracy"),
            quad("silver5k", f"stage2_{scope}_fine", "top3_accuracy"),
            triple("gold683", "stage1"),
            gate("gold683"),
            quad("gold683", f"stage2_{scope}_three", "top2_accuracy"),
            quad("gold683", f"stage2_{scope}_fine", "top3_accuracy"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "CSV/XLSX companion files contain counts, per-class precision/recall/F1, and run metadata.",
        "",
    ])
    return "\n".join(lines)


def write_xlsx(
    path: Path,
    wide_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    stage2_three_rows: list[dict[str, Any]],
    stage2_fine_rows: list[dict[str, Any]],
) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    def add_sheet(name: str, rows: list[dict[str, Any]]) -> None:
        ws = wb.create_sheet(name)
        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([row.get(header) for header in headers])
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        header_fill = PatternFill("solid", fgColor="1F4E79")
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for column_idx, header in enumerate(headers, start=1):
            letter = get_column_letter(column_idx)
            width = min(max(len(header) + 2, 12), 34)
            ws.column_dimensions[letter].width = width
            if any(token in header for token in ["accuracy", "macro_f1", "precision", "recall", "_f1"]):
                for cell in ws[letter][1:]:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0%"
            if header in {"elapsed_seconds", "items", "fatal_items_for_stage2", "support", "tp", "fp", "fn"} or header.endswith("_total") or header.endswith("_correct"):
                for cell in ws[letter][1:]:
                    if isinstance(cell.value, int):
                        cell.number_format = "0"
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=False)

    add_sheet("Main Table", wide_rows)
    add_sheet("Long Results", long_rows)
    add_sheet("Stage1 Per-Class", stage1_rows)
    add_sheet("Stage2 Three-Class", stage2_three_rows)
    add_sheet("Stage2 Fine-12", stage2_fine_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_readme(path: Path) -> None:
    path.write_text(
        "\n".join([
            "# Main Experiment Tables",
            "",
            "Generated from `experiments/main_20260510_main/*/summary.json` and `per_item_results.jsonl`.",
            "",
            "Stage 2 scopes:",
            "",
            "- `oracle`: all ground-truth `fatal_violation` samples, regardless of Stage 1 prediction.",
            "- `gated`: only ground-truth fatal samples where Stage 1 predicted `fatal_violation`.",
            "- `end_to_end`: all ground-truth fatal samples; Stage 1 misses count as Stage 2 wrong.",
            "",
            "Files:",
            "",
            "- `main_experiment_results.md`: compact paper-facing table using end-to-end Stage 2.",
            "- `main_experiment_results_gated.md`: compact table using conditional gated Stage 2.",
            "- `main_experiment_results_oracle.md`: compact table using the older oracle-fatal Stage 2.",
            "- `main_experiment_results_wide.csv`: one row per model/strategy, with Silver and Gold metrics side by side.",
            "- `main_experiment_results_long.csv`: one row per dataset/model/strategy.",
            "- `main_experiment_results.xlsx`: workbook with main, long, and per-class sheets.",
            "- `stage1_per_class.csv`: Stage 1 precision/recall/F1 by label.",
            "- `stage2_three_class_per_class.csv`: Stage 2 3-class precision/recall/F1 by scope.",
            "- `stage2_fine12_per_class.csv`: Stage 2 fine-grained 12-class precision/recall/F1 by scope.",
            "",
            "Per-class accuracy in presentation tables should use recall (`tp / support`).",
            "",
        ])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summaries = load_summaries(args.root)
    if len(summaries) != 48:
        raise SystemExit(f"Expected 48 main runs, found {len(summaries)}")

    long_rows = build_long_rows(summaries)
    wide_rows = build_wide_rows(long_rows)
    stage1_rows = build_class_rows(summaries, "stage1_problem_detection", STAGE1_CLASSES)
    stage2_three_rows = build_scoped_stage2_class_rows(summaries, "stage2_three_class", THREE_CLASS_CLASSES)
    stage2_fine_rows = build_scoped_stage2_class_rows(summaries, "stage2_fine", FINE12_CLASSES)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "main_experiment_results_wide.csv", wide_rows)
    write_csv(args.output / "main_experiment_results_long.csv", long_rows)
    write_csv(args.output / "stage1_per_class.csv", stage1_rows)
    write_csv(args.output / "stage2_three_class_per_class.csv", stage2_three_rows)
    write_csv(args.output / "stage2_fine12_per_class.csv", stage2_fine_rows)
    (args.output / "main_experiment_results.md").write_text(markdown_table(wide_rows, scope="end_to_end"))
    (args.output / "main_experiment_results_gated.md").write_text(markdown_table(wide_rows, scope="gated"))
    (args.output / "main_experiment_results_oracle.md").write_text(markdown_table(wide_rows, scope="oracle"))
    write_xlsx(
        args.output / "main_experiment_results.xlsx",
        wide_rows,
        long_rows,
        stage1_rows,
        stage2_three_rows,
        stage2_fine_rows,
    )
    write_readme(args.output / "README.md")
    print(f"Wrote tables to {args.output}")


if __name__ == "__main__":
    main()
