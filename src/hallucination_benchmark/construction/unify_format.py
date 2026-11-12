#!/usr/bin/env python3
"""
Protocol Format Unifier
========================
Rewrites all protocols to a standard format to eliminate format leakage.
Preserves technical content, errors, and labels.

Usage:
  python3 unify_format.py
  python3 unify_format.py --concurrency 20 --rpm 100
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from hallucination_benchmark.config import CURRENT_DATA_DIR, OUTPUT_DIR, RUNTIME_DIR, get_model_config
from hallucination_benchmark.evaluator import RateLimiter
from hallucination_benchmark.llm_client import llm_call

SOURCE_FILE = os.path.join(CURRENT_DATA_DIR, "benchmark_balanced.json")
OUTPUT_FILE = os.path.join(CURRENT_DATA_DIR, "benchmark_balanced_unified.json")
CHECKPOINT_FILE = os.path.join(RUNTIME_DIR, ".rewrite_checkpoint.json")

REWRITE_PROMPT = """You are a technical writer. Rewrite the following experimental protocol into a standardized format.

STANDARD FORMAT (use these exact section headers):
**Protocol:** [one-line title]
**Objective:** [research goal]
**Equipment:** [hardware, sensors, components with model numbers and specs]
**Sensor Placement & Configuration:** [where sensors are placed, electrode config, etc.]
**Data Acquisition Procedure:** [sampling rates, channels, recording protocol]
**Signal Processing & Validation:** [filters, algorithms, validation methodology]
**Safety & Feasibility:** [safety measures, practical considerations]

RULES:
1. Preserve ALL technical details exactly (sampling rates, frequencies, model numbers, accuracy claims).
2. Preserve any errors or problems in the original — do NOT fix them.
3. Do NOT add meta-commentary like "note:", "minor issue:", or "this is suboptimal".
4. Do NOT mention whether the protocol is valid or invalid.
5. Write in professional academic English.
6. Each section should be 1-3 sentences.

Original protocol:
{protocol}

Rewritten protocol (standard format only, no extra text):"""


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f)
    os.replace(tmp, CHECKPOINT_FILE)


def rewrite_protocol(protocol_text, model_key, rate_limiter, max_retries=3):
    """Rewrite a single protocol to standard format."""
    # Clean leakage text first (simple regex)
    cleaned = re.sub(r"(?i)\bnote:\s*minor\s+suboptimal\s+issue\.?\s*$", "", protocol_text)
    cleaned = re.sub(r"(?i)\bminor\s+suboptimal\s+issue\.?\s*$", "", cleaned)
    cleaned = cleaned.strip()

    for attempt in range(max_retries):
        rate_limiter.acquire()
        content, _ = llm_call(
            "You are a technical writer. Output only the rewritten protocol, no explanations.",
            REWRITE_PROMPT.format(protocol=cleaned[:3000]),
            model_key=model_key,
            temperature=0.3,
            max_tokens=2000,
            timeout=120,
        )
        if content and len(content) > 100:
            # Strip any trailing notes the LLM might add
            result = content.strip()
            # Remove "Note:" or "N.B.:" at the end
            result = re.sub(r"\n*(?:Note|N\.B\.|Important):.*$", "", result, flags=re.DOTALL | re.IGNORECASE)
            return result.strip()
        if attempt < max_retries - 1:
            time.sleep((attempt + 1) * 2)

    return None


def main():
    parser = argparse.ArgumentParser(description="Unify protocol format")
    parser.add_argument("--data", default=SOURCE_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--rpm", type=int, default=100)
    args = parser.parse_args()

    with open(args.data) as f:
        items = json.load(f)

    checkpoint = load_checkpoint()
    rate_limiter = RateLimiter(rpm=args.rpm)
    model_cfg = get_model_config(args.model)

    print(f"Loaded {len(items)} items")
    print(f"Checkpoint: {len(checkpoint)} already done")
    print(f"Model: {model_cfg['model']}")
    print(f"Concurrency: {args.concurrency}, RPM: {args.rpm}")

    # Find items that need rewriting
    pending = []
    for item in items:
        item_id = item["id"]
        if item_id not in checkpoint:
            pending.append(item)

    print(f"Pending: {len(pending)}")
    print(f"\n{'='*60}")

    start = time.time()
    done = len(checkpoint)

    def _rewrite(item):
        new_protocol = rewrite_protocol(item["protocol"], args.model, rate_limiter)
        return item["id"], new_protocol

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_rewrite, item): item for item in pending}
        for f in as_completed(futures):
            item_id, new_protocol = f.result()
            done += 1

            if new_protocol:
                checkpoint[item_id] = new_protocol
            else:
                # Keep original if rewrite fails
                checkpoint[item_id] = items[[i["id"] for i in items].index(item_id)]["protocol"]

            if done % 50 == 0 or done == len(items):
                elapsed = time.time() - start
                save_checkpoint(checkpoint)
                print(f"  [{done}/{len(items)}] ({elapsed:.0f}s)", flush=True)

    # Final save
    save_checkpoint(checkpoint)

    # Apply rewrites to items
    for item in items:
        item_id = item["id"]
        if item_id in checkpoint:
            item["protocol"] = checkpoint[item_id]

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    print(f"\nSaved {len(items)} items to {args.output} ({elapsed:.0f}s)")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Removed checkpoint")


if __name__ == "__main__":
    main()
