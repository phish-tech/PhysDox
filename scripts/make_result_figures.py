"""Rebuild the result figures shown in the README from the released result tables.

    uv run --extra plots python scripts/make_result_figures.py              # downloads tables from Hugging Face
    uv run --extra plots python scripts/make_result_figures.py --tables PATH_TO/tables

Writes PNGs to ``assets/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from hallucination_benchmark.data import HF_PUBLIC, cache_dir, download, hf_url  # noqa: E402

TABLES_PATH = "experiment_outputs/experiments/main_20260510_main/tables"
MODEL_ORDER = ["Claude Opus 4.7", "DeepSeek V4 Flash", "DeepSeek V4 Pro", "MiMO v2.5", "MiMO v2.5 Pro", "GPT-5.5"]
STRATEGY_ORDER = ["Zero-Shot", "CoT", "Self-Cons", "Tool-Aug"]
STRATEGY_COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
CLASS_COLORS = {"valid": "#1abc9c", "minor_issue": "#f39c12", "fatal_violation": "#c0392b"}
CLASS_NAMES = {"valid": "valid", "minor_issue": "minor issue", "fatal_violation": "fatal violation"}
SHORT = {"Claude Opus 4.7": "Claude\nOpus 4.7", "DeepSeek V4 Flash": "DeepSeek\nV4 Flash",
         "DeepSeek V4 Pro": "DeepSeek\nV4 Pro", "MiMO v2.5": "MiMO\nv2.5", "MiMO v2.5 Pro": "MiMO\nv2.5 Pro",
         "GPT-5.5": "GPT-5.5"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load_tables(tables: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = ("main_experiment_results_wide.csv", "stage1_per_class.csv")
    if tables:
        base = Path(tables)
        return tuple(pd.read_csv(base / n) for n in names)  # type: ignore[return-value]
    out = []
    for n in names:
        dest = cache_dir() / "tables" / n
        if not dest.exists():
            download(hf_url(HF_PUBLIC, f"{TABLES_PATH}/{n}"), dest)
        out.append(pd.read_csv(dest))
    return out[0], out[1]


def fig_stage1_f1(wide: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (col, title) in zip(axes, [("gold683_stage1_macro_f1", "Gold (683, expert-curated)"),
                                       ("silver5k_stage1_macro_f1", "Silver (5,000)")]):
        width = 0.2
        for j, strat in enumerate(STRATEGY_ORDER):
            vals = [wide[(wide.model_label == m) & (wide.strategy_label == strat)][col].iloc[0] * 100
                    for m in MODEL_ORDER]
            ax.bar([i + (j - 1.5) * width for i in range(len(MODEL_ORDER))], vals, width,
                   color=STRATEGY_COLORS[j], label=strat)
        ax.axhline(31, color="grey", ls="--", lw=1, label="uniform random guess (about 31)")
        ax.set_xticks(range(len(MODEL_ORDER)), [SHORT[m] for m in MODEL_ORDER], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 75)
    axes[0].set_ylabel("Stage-1 macro-F1 (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", fontsize=9, frameon=False)
    fig.suptitle("Severity detection: no model clears 55 macro-F1 on Gold, and the inference strategy barely matters",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_class_recall(per_class: pd.DataFrame, out: Path) -> None:
    g = per_class[per_class.dataset == "gold683"]
    rec = g.pivot_table(index="model_label", columns="class", values="recall", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.26
    for j, cls in enumerate(["valid", "minor_issue", "fatal_violation"]):
        vals = [rec.loc[m, cls] for m in MODEL_ORDER]
        bars = ax.bar([i + (j - 1) * width for i in range(len(MODEL_ORDER))], vals, width,
                      color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(MODEL_ORDER)), [SHORT[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("Recall on Gold (%), mean over 4 strategies")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, frameon=False, loc="upper center")
    ax.set_title("Models fail in opposite directions: GPT-5.5 rejects almost everything, MiMO accepts too much")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_cascade(wide: pd.DataFrame, out: Path) -> None:
    m = wide.groupby("model_label")[["gold683_stage2_oracle_three_accuracy",
                                     "gold683_stage2_end_to_end_three_accuracy"]].mean() * 100
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.36
    xs = range(len(MODEL_ORDER))
    o = [m.loc[k, "gold683_stage2_oracle_three_accuracy"] for k in MODEL_ORDER]
    e = [m.loc[k, "gold683_stage2_end_to_end_three_accuracy"] for k in MODEL_ORDER]
    ax.bar([i - width / 2 for i in xs], o, width, color="#7fb3d5", label="Stage 2 given the true fatal label (oracle)")
    ax.bar([i + width / 2 for i in xs], e, width, color="#c0392b", label="Stage 2 end-to-end (Stage-1 misses count as wrong)")
    for i, (a, b) in enumerate(zip(o, e)):
        ax.text(i - width / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs), [SHORT[k] for k in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("3-class diagnosis accuracy on Gold (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Diagnosis accuracy falls once Stage-1 misses carry over")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=None, help="Local tables/ directory. Default: download from Hugging Face.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "assets"))
    args = ap.parse_args()
    wide, per_class = load_tables(args.tables)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_stage1_f1(wide, out / "results_stage1_f1.png")
    fig_class_recall(per_class, out / "results_class_recall.png")
    fig_cascade(wide, out / "results_cascade.png")
    print("wrote", *sorted(p.name for p in out.glob("results_*.png")))


if __name__ == "__main__":
    main()
