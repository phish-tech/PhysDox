"""Small dependency-free classification metrics used by experiments."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def confusion_matrix(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    """Return a JSON-friendly confusion matrix indexed by reference then prediction."""
    matrix = {label: {pred: 0 for pred in labels + ["__missing__"]} for label in labels}
    for true, pred in zip(y_true, y_pred):
        if true not in matrix:
            continue
        key = pred if pred in labels else "__missing__"
        matrix[true][key] += 1
    return matrix


def classification_metrics(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    """Compute accuracy, balanced accuracy, macro-F1, and per-class stats."""
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
    """Compute top-k accuracy from ranked predictions."""
    allowed = set(labels)
    hits = 0
    total = len(y_true)
    for true, ranked in zip(y_true, y_pred_ranked):
        cleaned = [pred for pred in ranked if pred in allowed]
        if true in cleaned[:k]:
            hits += 1
    return {"k": k, "total": total, "hits": hits, "accuracy": hits / total if total else 0.0}
