#!/usr/bin/env python3
"""
Prompting Strategy Experiment
==============================
Compare Baseline vs CoT vs Self-Consistency vs Tool-Augmented on 100 items.

Usage:
  python3 experiment.py
  python3 experiment.py --models mimo mimo-v2.5
  python3 experiment.py --sc-runs 5
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from hallucination_benchmark.config import EXPERIMENTS_DIR, LEGACY_DATA_DIR, get_model_config
from hallucination_benchmark.evaluator import RateLimiter
from hallucination_benchmark.llm_client import llm_call, extract_json
from hallucination_benchmark.rule_checkers import run_checks

DATA_FILE = os.path.join(LEGACY_DATA_DIR, "experiment_100.json")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "prompting")

# ============================================================
# Prompts
# ============================================================

BASELINE_PROMPT = """You are an IEEE journal reviewer with deep expertise in hardware engineering and signal processing.
Review the experimental protocol below and determine whether it is physically feasible.
Focus on:
1. Does the sampling rate satisfy the Nyquist theorem?
2. Does the sensor bandwidth cover the target signal frequency?
3. Are the hardware parameters self-consistent (accuracy, resolution, power, etc.)?
4. Is the signal processing pipeline reasonable?

Output JSON: {"is_physically_viable": true/false, "identified_error": "error description or 'none'", "reasoning": "reasoning process"}"""

COT_PROMPT = """You are an IEEE journal reviewer with deep expertise in hardware engineering and signal processing.
Review the experimental protocol below and determine whether it is physically feasible.

Think step by step:
1. First, identify the target signal and its frequency range.
2. Check if the sampling rate satisfies the Nyquist theorem (fs >= 2 * fmax).
3. Check if the sensor bandwidth covers the signal frequency.
4. Check if the hardware parameters (ADC resolution, power, latency) are self-consistent.
5. Check if the claimed accuracy is within physical limits.
6. Check if the sensor placement matches the target signal source.
7. Check if any physical barriers (metal, opaque materials) block the signal path.
8. Finally, conclude whether the protocol is physically viable.

For each step, state your finding before moving to the next step.

Output JSON: {"is_physically_viable": true/false, "identified_error": "error description or 'none'", "reasoning": "step-by-step reasoning"}"""

TOOL_PROMPT = """You are an IEEE journal reviewer with deep expertise in hardware engineering and signal processing.
Review the experimental protocol below and determine whether it is physically feasible.

You have been given automated rule-check results below. Use them as supplementary evidence, but also apply your own judgment.

Rule-check results:
{rule_results}

Based on the protocol and the rule-check results, determine if the protocol is physically viable.

Output JSON: {{"is_physically_viable": true/false, "identified_error": "error description or 'none'", "reasoning": "reasoning process", "rule_check_used": true/false}}"""


# ============================================================
# Evaluation functions
# ============================================================

def eval_baseline(item, model_key, rate_limiter):
    """Baseline: standard L2 evaluation."""
    rate_limiter.acquire()
    content, reasoning = llm_call(
        BASELINE_PROMPT, item["protocol"],
        model_key=model_key, temperature=0.1, max_tokens=4000, timeout=120,
    )
    result = extract_json(content) or extract_json(reasoning)
    return judge(item, result)


def eval_cot(item, model_key, rate_limiter):
    """Chain-of-thought evaluation."""
    rate_limiter.acquire()
    content, reasoning = llm_call(
        COT_PROMPT, item["protocol"],
        model_key=model_key, temperature=0.1, max_tokens=4000, timeout=120,
    )
    result = extract_json(content) or extract_json(reasoning)
    return judge(item, result)


def eval_tool(item, model_key, rate_limiter):
    """Tool-augmented: run rule checkers, then evaluate with results."""
    # Run rule checkers
    domain = item.get("domain", "")
    constraint = item.get("violated_constraint", "none")
    if constraint == "none":
        constraint = "sampling_theorem"  # run a generic check

    rule_results = run_checks(item["protocol"], domain, constraint)
    rule_text = json.dumps(rule_results, ensure_ascii=False, indent=2)

    rate_limiter.acquire()
    content, reasoning = llm_call(
        TOOL_PROMPT.format(rule_results=rule_text), item["protocol"],
        model_key=model_key, temperature=0.1, max_tokens=4000, timeout=120,
    )
    result = extract_json(content) or extract_json(reasoning)
    return judge(item, result)


def eval_sc(item, model_key, rate_limiter, n_runs=5):
    """Self-consistency: run multiple times, majority vote."""
    votes = []
    for _ in range(n_runs):
        rate_limiter.acquire()
        content, reasoning = llm_call(
            BASELINE_PROMPT, item["protocol"],
            model_key=model_key, temperature=0.7, max_tokens=4000, timeout=120,
        )
        result = extract_json(content) or extract_json(reasoning)
        if result:
            is_viable = result.get("is_physically_viable")
            votes.append(is_viable)

    if not votes:
        return {"correct": False, "model_response": None}

    # Majority vote
    counter = Counter(votes)
    majority = counter.most_common(1)[0][0]

    # Build synthetic result
    synthetic = {"is_physically_viable": majority, "reasoning": f"SC majority vote: {counter}"}
    return judge(item, synthetic)


def judge(item, result):
    """Determine if model's judgment is correct."""
    true_label = item["label"]
    is_viable = result.get("is_physically_viable") if result else None

    if true_label == "valid":
        correct = (is_viable is True)
    elif true_label in ("fatal_violation", "minor_issue"):
        correct = (is_viable is False)
    elif true_label == "ambiguous":
        correct = True
    else:
        correct = False

    return {"correct": correct, "model_response": result}


# ============================================================
# Run experiment
# ============================================================

TECHNIQUES = {
    "baseline": lambda item, mk, rl, sc_runs: eval_baseline(item, mk, rl),
    "cot": lambda item, mk, rl, sc_runs: eval_cot(item, mk, rl),
    "self_consistency": lambda item, mk, rl, sc_runs: eval_sc(item, mk, rl, sc_runs),
    "tool_augmented": lambda item, mk, rl, sc_runs: eval_tool(item, mk, rl),
}


def run_experiment(items, model_key, technique, concurrency, rpm, sc_runs=5):
    """Run one experiment: model x technique."""
    model_cfg = get_model_config(model_key)
    model_name = model_cfg["model"]
    rate_limiter = RateLimiter(rpm=rpm)
    eval_fn = TECHNIQUES[technique]

    print(f"\n  {model_name} x {technique}...", end=" ", flush=True)

    results = []
    start = time.time()

    def _eval(item):
        return item, eval_fn(item, model_key, rate_limiter, sc_runs)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_eval, item): item for item in items}
        for f in as_completed(futures):
            item, result = f.result()
            results.append({**item, "_exp_result": result})

    elapsed = time.time() - start

    # Stats
    correct = sum(1 for r in results if r["_exp_result"]["correct"])
    total = len(results)
    errors = sum(1 for r in results if r["_exp_result"]["model_response"] is None)

    # Per-label
    label_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        label = r["label"]
        label_stats[label]["total"] += 1
        if r["_exp_result"]["correct"]:
            label_stats[label]["correct"] += 1

    # Per-category
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r.get("error_category", "unknown")
        cat_stats[cat]["total"] += 1
        if r["_exp_result"]["correct"]:
            cat_stats[cat]["correct"] += 1

    stats = {
        "model": model_name,
        "technique": technique,
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / (total - errors) * 100 if total - errors > 0 else 0,
        "elapsed": round(elapsed),
        "label_stats": {k: dict(v) for k, v in label_stats.items()},
        "cat_stats": {k: dict(v) for k, v in cat_stats.items()},
    }

    print(f"{correct}/{total-errors} ({stats['accuracy']:.1f}%) [{elapsed:.0f}s]", flush=True)

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Prompting Strategy Experiment")
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--models", nargs="+", default=["mimo", "mimo-v2.5"])
    parser.add_argument("--techniques", nargs="+",
                        default=["baseline", "cot", "self_consistency", "tool_augmented"])
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--rpm", type=int, default=100)
    parser.add_argument("--sc-runs", type=int, default=5, help="Runs for self-consistency")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.data) as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items from {args.data}")
    print(f"Models: {args.models}")
    print(f"Techniques: {args.techniques}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    all_stats = []

    for model_key in args.models:
        model_cfg = get_model_config(model_key)
        print(f"\n{'='*60}")
        print(f"Model: {model_cfg['model']}")
        print(f"{'='*60}")

        for technique in args.techniques:
            results, stats = run_experiment(
                items, model_key, technique,
                args.concurrency, args.rpm, args.sc_runs,
            )
            all_results[f"{model_key}_{technique}"] = results
            all_stats.append(stats)

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':25s} {'Technique':20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>9s} {'Time':>6s}")
    print(f"{'-'*25} {'-'*20} {'-'*8} {'-'*6} {'-'*9} {'-'*6}")
    for s in all_stats:
        print(f"{s['model']:25s} {s['technique']:20s} {s['correct']:8d} {s['total']:6d} {s['accuracy']:8.1f}% {s['elapsed']:5d}s")

    # Per-label breakdown
    print(f"\n{'='*80}")
    print(f"PER-LABEL BREAKDOWN")
    print(f"{'='*80}")
    for s in all_stats:
        print(f"\n  {s['model']} x {s['technique']}:")
        for label in ["fatal_violation", "valid", "minor_issue"]:
            ls = s["label_stats"].get(label, {"correct": 0, "total": 0})
            acc = ls["correct"] / ls["total"] * 100 if ls["total"] > 0 else 0
            print(f"    {label:20s} {ls['correct']}/{ls['total']} ({acc:.0f}%)")

    # Per-category breakdown
    print(f"\n{'='*80}")
    print(f"PER-CATEGORY BREAKDOWN")
    print(f"{'='*80}")
    for s in all_stats:
        print(f"\n  {s['model']} x {s['technique']}:")
        for cat, cs in sorted(s["cat_stats"].items()):
            acc = cs["correct"] / cs["total"] * 100 if cs["total"] > 0 else 0
            print(f"    {cat:40s} {cs['correct']}/{cs['total']} ({acc:.0f}%)")

    # Save
    output_file = args.output or os.path.join(OUTPUT_DIR, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Record full experimental config
    experiment_config = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data,
        "data_size": len(items),
        "data_label_distribution": dict(Counter(i["label"] for i in items)),
        "models": {},
        "techniques": args.techniques,
        "concurrency": args.concurrency,
        "rpm": args.rpm,
        "sc_runs": args.sc_runs,
        "prompts": {
            "baseline": BASELINE_PROMPT,
            "cot": COT_PROMPT,
            "tool_augmented": TOOL_PROMPT,
        },
    }
    for mk in args.models:
        cfg = get_model_config(mk)
        experiment_config["models"][mk] = {
            "key": mk,
            "model_name": cfg["model"],
            "base_url": cfg["base_url"],
        }

    output_data = {
        "experiment_config": experiment_config,
        "stats": all_stats,
        "results": all_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
