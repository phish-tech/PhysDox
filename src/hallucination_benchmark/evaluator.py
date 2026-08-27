#!/usr/bin/env python3
"""
Benchmark Evaluator for Hardware-Grounded Hallucination
========================================================
Evaluates LLMs on the generated sabotage benchmark with:
- Binary detection (correct/incorrect)
- Multi-label classification (valid/fatal/minor/ambiguous)
- Per-category, per-difficulty, per-domain breakdown
- L1/L2/L3 level analysis
- Checkpoint resume for large evaluations
- Concurrent API calls

Usage:
  uv run benchmark-evaluate data/current/benchmark_balanced.json
  uv run benchmark-evaluate data/current/benchmark_balanced.json --level L2
  uv run benchmark-evaluate data/current/benchmark_balanced.json --level all --max-items 50
  uv run benchmark-evaluate data/current/benchmark_balanced.json --model openai
  uv run benchmark-evaluate data/current/benchmark_balanced.json --model mimo --concurrency 8
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from hallucination_benchmark.config import get_model_config
from hallucination_benchmark.llm_client import llm_call, extract_json


# ============================================================
# Rate Limiter (sliding window, thread-safe)
# ============================================================

class RateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, rpm: int = 100):
        self.rpm = rpm
        self.lock = threading.Lock()
        self.timestamps: list[float] = []

    def acquire(self):
        """Wait until a request slot is available."""
        while True:
            with self.lock:
                now = time.time()
                # Remove timestamps older than 60s
                self.timestamps = [t for t in self.timestamps if now - t < 60]
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return
            time.sleep(0.2)


# ============================================================
# Evaluation Prompts (L1 / L2 / L3)
# ============================================================

L1_SYSTEM_PROMPT = """You are a biomedical engineering researcher.
Read the experimental protocol below and determine whether the sensors and equipment used are appropriate for collecting the target signal.
This is a basic matching task: do the sensor types match the target signal?

Output JSON: {"sensor_match": true/false, "reasoning": "brief explanation"}"""

L2_SYSTEM_PROMPT = """You are an IEEE journal reviewer with deep expertise in hardware engineering and signal processing.
Review the experimental protocol below and determine whether it is physically feasible.
Focus on:
1. Does the sampling rate satisfy the Nyquist theorem?
2. Does the sensor bandwidth cover the target signal frequency?
3. Are the hardware parameters self-consistent (accuracy, resolution, power, etc.)?
4. Is the signal processing pipeline reasonable?

Output JSON: {"is_physically_viable": true/false, "identified_error": "error description or 'none'", "reasoning": "reasoning process"}"""

L3_SYSTEM_PROMPT = """You are a senior biomedical engineering professor evaluating the completeness of an experimental design.
Review the protocol below, assess not only feasibility but also:
1. If a sensor is unavailable, are there alternative approaches?
2. Are there unstated assumptions or limitations in the design?
3. Is the validation methodology sufficient?

Output JSON: {
  "is_physically_viable": true/false,
  "identified_error": "error description or 'none'",
  "alternative_suggestions": "alternative suggestions or 'none'",
  "hidden_assumptions": "unstated assumptions or 'none'",
  "reasoning": "reasoning process"
}"""


# ============================================================
# Evaluation Logic
# ============================================================

def evaluate_single(
    item: dict,
    level: str = "L2",
    model_key: str = None,
    model_name_override: str = None,
    max_retries: int = 3,
    rate_limiter: RateLimiter = None,
) -> dict:
    """Evaluate a single benchmark item with retry logic."""
    protocol = item["protocol"]
    true_label = item["label"]

    prompts = {"L1": L1_SYSTEM_PROMPT, "L2": L2_SYSTEM_PROMPT, "L3": L3_SYSTEM_PROMPT}
    system_prompt = prompts.get(level, L2_SYSTEM_PROMPT)

    content = None
    reasoning = ""
    for attempt in range(max_retries):
        if rate_limiter:
            rate_limiter.acquire()
        content, reasoning = llm_call(
            system_prompt,
            protocol,
            model_key=model_key,
            model_name_override=model_name_override,
            temperature=0.1,
            max_tokens=4000,
            timeout=120,
        )
        if content:
            break
        if attempt < max_retries - 1:
            time.sleep((attempt + 1) * 2)

    result = extract_json(content)
    if not result and reasoning:
        result = extract_json(reasoning)

    if level == "L1":
        model_judgment = result.get("sensor_match") if result else None
        correct = (model_judgment == (true_label == "valid"))
    else:
        is_viable = result.get("is_physically_viable") if result else None
        if true_label == "valid":
            correct = (is_viable is True)
        elif true_label in ("fatal_violation", "minor_issue"):
            correct = (is_viable is False)
        elif true_label == "ambiguous":
            correct = True
        else:
            correct = False

    return {
        "level": level,
        "model_response": result,
        "true_label": true_label,
        "correct": correct,
        "raw_content": (content or "")[:500],
    }


# ============================================================
# Checkpoint
# ============================================================

def _get_checkpoint_path(benchmark_file: str, level: str, model_name: str) -> str:
    model_slug = model_name.replace("/", "_")
    base = os.path.splitext(benchmark_file)[0]
    return f"{base}_ckpt_{level}_{model_slug}.json"


def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {len(data)} items already evaluated")
        return data
    return {}


def _save_checkpoint(path: str, checkpoint: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    os.replace(tmp, path)


# ============================================================
# Main evaluation (concurrent)
# ============================================================

def run_evaluation(
    benchmark_file: str,
    level: str = "L2",
    max_items: int = None,
    model_key: str = None,
    model_name_override: str = None,
    concurrency: int = 20,
    max_retries: int = 3,
    rpm: int = 100,
) -> tuple[list[dict], dict]:
    """Run full evaluation with concurrency, rate limiting, and checkpoint resume."""
    with open(benchmark_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    if max_items:
        items = items[:max_items]

    model_cfg = get_model_config(model_key, model_name_override=model_name_override)
    model_name = model_cfg["model"]

    ckpt_path = _get_checkpoint_path(benchmark_file, level, model_name)
    checkpoint = _load_checkpoint(ckpt_path)

    rate_limiter = RateLimiter(rpm=rpm)

    # Split already-done vs pending
    pending_indices = []
    for i, item in enumerate(items):
        item_id = item.get("id", str(i))
        if item_id not in checkpoint:
            pending_indices.append(i)

    print(f"\n{'='*60}")
    print(f"Evaluating {len(items)} items at {level} level")
    print(f"  Model: {model_name}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Rate limit: {rpm} RPM")
    print(f"  Already done: {len(items) - len(pending_indices)}")
    print(f"  Pending: {len(pending_indices)}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    start = time.time()

    # Evaluate pending items concurrently
    def _eval_one(idx):
        item = items[idx]
        item_id = item.get("id", str(idx))
        eval_result = evaluate_single(
            item,
            level,
            model_key=model_key,
            model_name_override=model_name_override,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        return idx, item_id, eval_result

    done_count = len(items) - len(pending_indices)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_eval_one, i): i for i in pending_indices}
        for f in as_completed(futures):
            idx, item_id, eval_result = f.result()
            checkpoint[item_id] = eval_result
            done_count += 1

            label = items[idx]["label"]
            domain = items[idx].get("domain", "")[:20]
            status = "ok" if eval_result["correct"] else ("API error" if eval_result["model_response"] is None else "FAIL")
            print(f"[{done_count}/{len(items)}] {domain} | {label[:15]} | {status}", flush=True)

            # Save checkpoint every 20 items
            if done_count % 20 == 0:
                _save_checkpoint(ckpt_path, checkpoint)

    # Final checkpoint save
    _save_checkpoint(ckpt_path, checkpoint)
    elapsed = time.time() - start

    # Build results list (ordered)
    results = []
    for i, item in enumerate(items):
        item_id = item.get("id", str(i))
        eval_result = checkpoint.get(item_id, {
            "level": level, "model_response": None, "true_label": item["label"],
            "correct": False, "raw_content": "",
        })
        results.append({**item, "eval_result": eval_result})

    # Statistics
    correct = sum(1 for r in results if r["eval_result"].get("correct"))
    errors = sum(1 for r in results if r["eval_result"].get("model_response") is None)
    valid_items = len(items) - errors
    accuracy = correct / valid_items * 100 if valid_items > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results ({level})  [{elapsed:.0f}s]")
    print(f"{'='*60}")
    print(f"Overall: {correct}/{valid_items} ({accuracy:.1f}%)")

    def _breakdown(key: str, label_if_missing: str = "unknown"):
        stats: dict[str, dict] = {}
        for item, r in zip(items, results):
            k = item.get(key, label_if_missing)
            if k not in stats:
                stats[k] = {"total": 0, "correct": 0}
            stats[k]["total"] += 1
            if r.get("eval_result", {}).get("correct", False):
                stats[k]["correct"] += 1
        return stats

    def _print_breakdown(stats: dict, title: str):
        print(f"\n{title}:")
        for k, s in sorted(stats.items()):
            pct = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"  {k}: {s['correct']}/{s['total']} ({pct:.0f}%)")

    _print_breakdown(_breakdown("label"), "Per-label")
    _print_breakdown(_breakdown("difficulty"), "Per-difficulty")
    _print_breakdown(_breakdown("domain"), "Per-domain")
    _print_breakdown(_breakdown("error_category", "N/A"), "Per-error-category")

    stats = {
        "accuracy": accuracy,
        "total": len(items),
        "correct": correct,
        "errors": errors,
        "elapsed_seconds": round(elapsed),
        "label_stats": _breakdown("label"),
        "diff_stats": _breakdown("difficulty"),
        "domain_stats": _breakdown("domain"),
        "cat_stats": _breakdown("error_category", "N/A"),
    }

    model_slug = model_name.replace("/", "_")
    output_file = benchmark_file.replace(".json", f"_eval_{level}_{model_slug}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {output_file}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"Removed checkpoint file")

    return results, stats


def run_multilevel_evaluation(
    benchmark_file: str,
    max_items: int = None,
    model_key: str = None,
    model_name_override: str = None,
    concurrency: int = 20,
    rpm: int = 100,
) -> dict:
    all_results = {}
    for level in ["L1", "L2", "L3"]:
        results, stats = run_evaluation(
            benchmark_file, level=level, max_items=max_items, model_key=model_key, model_name_override=model_name_override, concurrency=concurrency, rpm=rpm,
        )
        all_results[level] = {"results": results, "stats": stats}

    print(f"\n{'='*60}")
    print("Multi-Level Comparison")
    print(f"{'='*60}")
    for level in ["L1", "L2", "L3"]:
        acc = all_results[level]["stats"]["accuracy"]
        print(f"  {level}: {acc:.1f}% accuracy")

    return all_results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluator")
    parser.add_argument("benchmark", help="Path to benchmark JSON file")
    parser.add_argument("--level", choices=["L1", "L2", "L3", "all"], default="L2")
    parser.add_argument("--max-items", type=int, default=None, help="Limit number of items")
    parser.add_argument("--model", type=str, default=None, help="Model key from config.py (e.g. mimo, openai, custom)")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name (e.g. anthropic/claude-3.5-sonnet)")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent API calls (default: 20)")
    parser.add_argument("--rpm", type=int, default=100, help="Requests per minute limit (default: 100)")
    parser.add_argument("--retries", type=int, default=3, help="Max retries per item (default: 3)")
    args = parser.parse_args()

    if args.level == "all":
        run_multilevel_evaluation(args.benchmark, max_items=args.max_items, model_key=args.model, model_name_override=args.model_name, concurrency=args.concurrency, rpm=args.rpm)
    else:
        run_evaluation(args.benchmark, level=args.level, max_items=args.max_items, model_key=args.model, model_name_override=args.model_name, concurrency=args.concurrency, max_retries=args.retries, rpm=args.rpm)


if __name__ == "__main__":
    main()
