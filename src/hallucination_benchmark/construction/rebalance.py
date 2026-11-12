#!/usr/bin/env python3
"""
Dataset Rebalancer
==================
Fixes label imbalance, constraint concentration, and minor_issue quality.

Three phases:
  A: Generate missing constraint types (fatal_violation)
  B: Generate valid + minor_issue with complete fields
  C: Sample and merge to target distribution

Usage:
  python3 rebalance.py
  python3 rebalance.py --phase a
  python3 rebalance.py --phase b
  python3 rebalance.py --phase c
  python3 rebalance.py --concurrency 20 --rpm 100
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from hallucination_benchmark.config import OUTPUT_DIR, RELEASES_DIR, SEED_DATA_DIR, get_model_config
from hallucination_benchmark.evaluator import RateLimiter
from hallucination_benchmark.construction.generator import (
    _generate_one_protocol,
    _inject_one_sabotage,
    _verify_one_item,
    _generate_one_negative,
    _rule_based_sabotage,
    assign_difficulty,
)
from hallucination_benchmark.llm_client import llm_call, llm_json_call
from hallucination_benchmark.rule_checkers import run_checks
from hallucination_benchmark.construction.schema import make_item, make_clean_for_output, generate_id

# ============================================================
# Constants
# ============================================================

SOURCE_FILE = os.path.join(RELEASES_DIR, "hq_3459", "benchmark_3459_hq.json")
PHASE_A_OUTPUT = os.path.join(OUTPUT_DIR, "phase_a_new_fatal.json")
PHASE_B_OUTPUT = os.path.join(OUTPUT_DIR, "phase_b_valid_minor.json")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "benchmark_balanced.json")

# Target distribution
TARGET_TOTAL = 3500
TARGET_FATAL_RATIO = 0.47   # 47% fatal_violation → target ~1450 for 3100 total
TARGET_VALID_RATIO = 0.28   # 28% valid
TARGET_MINOR_RATIO = 0.25   # 25% minor_issue

# Target constraint distribution within fatal (even spread)
TARGET_CONSTRAINTS = [
    "sampling_theorem",
    "sensor_bandwidth",
    "anatomical_misplacement",
    "optical_obstruction",
    "ematerial_block",
    "adc_resolution",
    "accuracy_impossibility",
    "power_budget",
    "data_bandwidth",
    "snr_floor",
    "latency_constraint",
    "temperature_range",
]

# Missing constraints in current dataset
MISSING_CONSTRAINTS = [
    "ematerial_block",
    "adc_resolution",
    "accuracy_impossibility",
    "power_budget",
    "data_bandwidth",
    "snr_floor",
    "latency_constraint",
    "temperature_range",
]


# ============================================================
# Phase A: Generate missing constraint types
# ============================================================

PHASE_A_PROMPT = """You are a biomedical engineering researcher writing realistic experimental protocols.

Generate a detailed, physically valid experimental protocol for:
Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}
Typical parameters: {typical_params}

Requirements:
1. Write as if for a methods section of a peer-reviewed paper.
2. Include specific hardware model numbers, sampling rates, resolutions, bandwidths.
3. Include signal processing pipeline details.
4. Include validation methodology and expected performance metrics.
5. All parameters must be physically realistic and consistent.
6. Do NOT include any errors or violations.
7. Length: 200-400 words.

Output ONLY the protocol text, no headers or commentary."""


def phase_a_generate_missing(constraints, domains, model_key, concurrency, rpm, items_per_constraint=30):
    """Generate fatal_violation items for missing constraints."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing data to check what we have
    existing = load_existing()
    existing_constraints = defaultdict(int)
    for item in existing:
        if item["label"] == "fatal_violation":
            existing_constraints[item.get("violated_constraint", "")] += 1

    print(f"\n{'='*70}")
    print(f"Phase A: Generate missing constraint types")
    print(f"{'='*70}")
    print(f"  Missing constraints: {MISSING_CONSTRAINTS}")
    print(f"  Items per constraint: {items_per_constraint}")
    print(f"  Concurrency: {concurrency}, RPM: {rpm}")

    # Build tasks: for each missing constraint, find applicable domains
    tasks = []
    constraint_map = {c["key"]: c for c in constraints}
    for ck in MISSING_CONSTRAINTS:
        if ck not in constraint_map:
            print(f"  WARNING: constraint '{ck}' not in seed data, skipping")
            continue
        cinfo = constraint_map[ck]
        applicable_domains = [d for d in domains if d["key"] in cinfo.get("domains", [])]
        if not applicable_domains:
            print(f"  WARNING: no domains for '{ck}', skipping")
            continue
        # Distribute items across applicable domains
        for i in range(items_per_constraint):
            domain = applicable_domains[i % len(applicable_domains)]
            tasks.append((domain, cinfo))

    print(f"  Total tasks: {len(tasks)}")

    rate_limiter = RateLimiter(rpm=rpm)
    new_items = []
    start = time.time()

    def _process(task):
        domain, cinfo = task
        rate_limiter.acquire()

        # Generate correct protocol
        protocol = _generate_one_protocol(domain, model_key, temperature=0.8)
        if not protocol:
            return None

        # Inject sabotage
        sabotage = _inject_one_sabotage(protocol, cinfo, model_key)
        if not sabotage:
            sabotage = _rule_based_sabotage(protocol, cinfo["key"], cinfo)
        if not sabotage:
            return None

        # Rule check
        rule_results = run_checks(sabotage["sabotaged_protocol"], domain["key"], cinfo["key"])
        rule_passed = all(r.get("passed", True) for r in rule_results) if rule_results else True

        # Verify
        rate_limiter.acquire()
        verifier = _verify_one_item(
            sabotage["sabotaged_protocol"],
            sabotage.get("violated_principle", cinfo["principle"]),
            sabotage.get("why_invalid", ""),
            model_key,
        )
        quality = verifier.get("quality_score", 0)
        is_wrong = verifier.get("is_physically_wrong", False)
        is_realistic = verifier.get("is_protocol_realistic", False)
        verifier_passed = is_wrong and is_realistic and quality >= 3

        if not verifier_passed and not rule_passed:
            return None

        difficulty = assign_difficulty(cinfo["key"])
        plausibility = "high" if verifier.get("llm_would_fool") else "medium"

        item = make_item(
            domain=domain["key"],
            scenario=domain["description"],
            label="fatal_violation",
            protocol=sabotage["sabotaged_protocol"],
            error_category=cinfo["category"],
            violated_constraint=cinfo["key"],
            violated_principle=sabotage.get("violated_principle", cinfo["principle"]),
            why_invalid=sabotage.get("why_invalid", ""),
            minimal_fix=sabotage.get("minimal_fix", ""),
            difficulty=difficulty,
            surface_plausibility=plausibility,
            severity=cinfo.get("severity", "fatal"),
            source_correct_protocol=protocol,
            what_changed=sabotage.get("what_changed", ""),
        )
        return item

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            result = f.result()
            if result:
                new_items.append(result)
            if done % 50 == 0 or done == len(futures):
                elapsed = time.time() - start
                print(f"  Progress: {done}/{len(futures)}, kept: {len(new_items)} ({elapsed:.0f}s)", flush=True)

    # Save
    with open(PHASE_A_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(new_items, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - start
    print(f"\n  Saved {len(new_items)} items to {PHASE_A_OUTPUT} ({elapsed:.0f}s)")

    # Print constraint distribution
    c_counts = defaultdict(int)
    for item in new_items:
        c_counts[item["violated_constraint"]] += 1
    print(f"  Constraint distribution:")
    for k, v in sorted(c_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:30s} {v:5d}")

    return new_items


# ============================================================
# Phase B: Generate valid + minor_issue with complete fields
# ============================================================

MINOR_ISSUE_PROMPT = """You are a biomedical engineering researcher writing an experimental protocol with ONE minor suboptimal issue.

Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}

Write a protocol that is physically feasible but has ONE minor issue that could affect data quality. The issue should be SUBTLE - a careful reviewer would flag it as "needs improvement" but NOT "physically impossible."

Examples of minor issues:
- Sampling rate above Nyquist but with minimal margin (could cause edge artifacts)
- Electrode impedance slightly high but still functional
- Sensor placement suboptimal but still captures signal
- Validation sample size small but not zero
- Filter order too low, causing some spectral leakage
- No mention of environmental noise control
- Calibration protocol not described in detail

Requirements:
1. The protocol must be PHYSICALLY FEASIBLE (not fatal).
2. The issue must be identifiable and specific.
3. Write 200-400 words.

Output a JSON object:
{{
    "protocol": "the full protocol text",
    "minor_issue_description": "what the minor issue is",
    "violated_principle": "the best practice or guideline that is slightly violated",
    "why_suboptimal": "why this issue affects data quality (but doesn't make it impossible)",
    "minimal_fix": "how to fix the issue (one sentence)"
}}"""


VALID_PROTOCOL_PROMPT = """You are a biomedical engineering researcher writing a physically valid, well-designed experimental protocol.

Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}
Typical parameters: {typical_params}

Write a protocol that is:
1. Physically correct and feasible
2. Well-designed with proper validation
3. Uses appropriate hardware with realistic specs
4. Has correct sampling rates, sensor placement, and signal processing

Write 200-400 words. Output ONLY the protocol text."""


def phase_b_generate_valid_minor(constraints, domains, model_key, concurrency, rpm,
                                  n_valid_per_domain=50, n_minor_per_domain=40):
    """Generate high-quality valid and minor_issue items."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Phase B: Generate valid + minor_issue")
    print(f"{'='*70}")
    print(f"  Valid per domain: {n_valid_per_domain}")
    print(f"  Minor per domain: {n_minor_per_domain}")
    print(f"  Concurrency: {concurrency}, RPM: {rpm}")

    rate_limiter = RateLimiter(rpm=rpm)
    start = time.time()

    # --- Generate valid items ---
    print(f"\n  Generating valid protocols...")
    valid_items = []

    def _gen_valid(domain):
        rate_limiter.acquire()
        prompt = VALID_PROTOCOL_PROMPT.format(
            domain=domain["key"],
            description=domain["description"],
            sensors=", ".join(domain["sensors"]),
            signals=", ".join(domain["signals"]),
            typical_params=json.dumps(domain["typical_params"]),
        )
        content, _ = llm_call(
            "You are a biomedical engineering researcher. Write only valid, physically correct protocols.",
            prompt, model_key=model_key, temperature=0.8, max_tokens=1500, timeout=120,
        )
        if not content:
            return None
        return make_item(
            domain=domain["key"],
            scenario=domain["description"],
            label="valid",
            protocol=content.strip(),
            error_category="N/A",
            violated_constraint="none",
            violated_principle="",
            why_invalid="",
            minimal_fix="",
            difficulty="easy",
            surface_plausibility="high",
            severity="none",
            source_correct_protocol="",
            what_changed="",
        )

    valid_tasks = []
    for domain in domains:
        for _ in range(n_valid_per_domain):
            valid_tasks.append(domain)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_gen_valid, d): d for d in valid_tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            result = f.result()
            if result:
                valid_items.append(result)
            if done % 50 == 0 or done == len(futures):
                print(f"    Valid progress: {done}/{len(futures)}, kept: {len(valid_items)}", flush=True)

    print(f"  Generated {len(valid_items)} valid items")

    # --- Generate minor_issue items ---
    print(f"\n  Generating minor_issue protocols...")
    minor_items = []

    def _gen_minor(domain):
        rate_limiter.acquire()
        prompt = MINOR_ISSUE_PROMPT.format(
            domain=domain["key"],
            description=domain["description"],
            sensors=", ".join(domain["sensors"]),
            signals=", ".join(domain["signals"]),
        )
        result = llm_json_call(
            "You are a biomedical engineering researcher. Output only JSON.",
            prompt, model_key=model_key, temperature=0.8, max_tokens=2000, timeout=120,
        )
        if not result or "protocol" not in result:
            return None

        return make_item(
            domain=domain["key"],
            scenario=domain["description"],
            label="minor_issue",
            protocol=result["protocol"],
            error_category="minor_issue",
            violated_constraint="suboptimal",
            violated_principle=result.get("violated_principle", ""),
            why_invalid=result.get("minor_issue_description", result.get("why_suboptimal", "")),
            minimal_fix=result.get("minimal_fix", ""),
            difficulty="medium",
            surface_plausibility="high",
            severity="minor",
            source_correct_protocol="",
            what_changed=result.get("minor_issue_description", ""),
        )

    minor_tasks = []
    for domain in domains:
        for _ in range(n_minor_per_domain):
            minor_tasks.append(domain)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_gen_minor, d): d for d in minor_tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            result = f.result()
            if result:
                minor_items.append(result)
            if done % 50 == 0 or done == len(futures):
                print(f"    Minor progress: {done}/{len(futures)}, kept: {len(minor_items)}", flush=True)

    print(f"  Generated {len(minor_items)} minor_issue items")

    # Combine with existing Phase B data (accumulate)
    all_new = valid_items + minor_items
    existing_b = load_phase_b()
    if existing_b:
        existing_keys = {i["protocol"][:200] for i in existing_b}
        added = 0
        for item in all_new:
            key = item["protocol"][:200]
            if key not in existing_keys:
                existing_b.append(item)
                existing_keys.add(key)
                added += 1
        all_new = existing_b
        print(f"  Accumulated: {added} new + {len(existing_b) - added} existing = {len(all_new)} total")
    else:
        print(f"  New batch: {len(all_new)} items")

    with open(PHASE_B_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_new, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - start
    print(f"\n  Saved {len(all_new)} items to {PHASE_B_OUTPUT} ({elapsed:.0f}s)")

    return all_new


# ============================================================
# Phase C: Sample and merge to target distribution
# ============================================================

def load_existing():
    """Load existing dataset."""
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_phase_a():
    """Load Phase A output if exists."""
    if os.path.exists(PHASE_A_OUTPUT):
        with open(PHASE_A_OUTPUT, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_phase_b():
    """Load Phase B output if exists."""
    if os.path.exists(PHASE_B_OUTPUT):
        with open(PHASE_B_OUTPUT, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def phase_c_rebalance(fatal_count=None):
    """Sample and merge all data to target distribution.
    
    Args:
        fatal_count: If set, use this exact number of fatal items.
                     If None, compute from TARGET_FATAL_RATIO.
    """
    print(f"\n{'='*70}")
    print(f"Phase C: Rebalance to target distribution")
    print(f"{'='*70}")

    existing = load_existing()
    new_fatal = load_phase_a()
    new_valid_minor = load_phase_b()

    print(f"  Existing:         {len(existing)}")
    print(f"  New fatal (A):    {len(new_fatal)}")
    print(f"  New valid/minor (B): {len(new_valid_minor)}")

    # Separate by label
    all_fatal = [i for i in existing if i["label"] == "fatal_violation"] + [i for i in new_fatal if i["label"] == "fatal_violation"]
    all_valid = [i for i in existing if i["label"] == "valid"] + [i for i in new_valid_minor if i["label"] == "valid"]
    all_minor = [i for i in existing if i["label"] == "minor_issue"] + [i for i in new_valid_minor if i["label"] == "minor_issue"]

    print(f"\n  Total available:")
    print(f"    fatal_violation: {len(all_fatal)}")
    print(f"    valid:           {len(all_valid)}")
    print(f"    minor_issue:     {len(all_minor)}")

    # --- Sample fatal: balance constraints ---
    if fatal_count:
        target_fatal = fatal_count
    else:
        target_fatal = int(TARGET_TOTAL * TARGET_FATAL_RATIO)
    target_per_constraint = target_fatal // len(TARGET_CONSTRAINTS)

    # Group fatal by constraint
    fatal_by_constraint = defaultdict(list)
    for item in all_fatal:
        ck = item.get("violated_constraint", "unknown")
        fatal_by_constraint[ck].append(item)

    print(f"\n  Fatal constraint pool:")
    for ck in TARGET_CONSTRAINTS:
        available = len(fatal_by_constraint.get(ck, []))
        print(f"    {ck:30s} {available:5d} available, target {target_per_constraint}")

    sampled_fatal = []
    random.seed(42)
    for ck in TARGET_CONSTRAINTS:
        pool = fatal_by_constraint.get(ck, [])
        n = min(target_per_constraint, len(pool))
        if n > 0:
            sampled = random.sample(pool, n)
            sampled_fatal.extend(sampled)

    # Fill remaining with any fatal items
    remaining = target_fatal - len(sampled_fatal)
    if remaining > 0:
        used_ids = {i["id"] for i in sampled_fatal}
        leftover = [i for i in all_fatal if i["id"] not in used_ids]
        if leftover:
            extra = random.sample(leftover, min(remaining, len(leftover)))
            sampled_fatal.extend(extra)

    print(f"\n  Sampled fatal: {len(sampled_fatal)}")

    # --- Sample valid ---
    target_valid = int(TARGET_TOTAL * TARGET_VALID_RATIO)
    if len(all_valid) >= target_valid:
        sampled_valid = random.sample(all_valid, target_valid)
    else:
        sampled_valid = all_valid
        print(f"  WARNING: only {len(all_valid)} valid available, target {target_valid}")

    # --- Sample minor ---
    target_minor = int(TARGET_TOTAL * TARGET_MINOR_RATIO)
    if len(all_minor) >= target_minor:
        sampled_minor = random.sample(all_minor, target_minor)
    else:
        sampled_minor = all_minor
        print(f"  WARNING: only {len(all_minor)} minor available, target {target_minor}")

    # --- Combine ---
    final = sampled_fatal + sampled_valid + sampled_minor
    random.shuffle(final)

    # Deduplicate
    seen = set()
    deduped = []
    for item in final:
        key = item["protocol"][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # Clean for output
    clean = [make_clean_for_output(i) for i in deduped]

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    # Print final stats
    print(f"\n{'='*70}")
    print(f"Final Dataset: {FINAL_OUTPUT}")
    print(f"{'='*70}")
    print(f"  Total: {len(clean)}")

    label_counts = defaultdict(int)
    for item in clean:
        label_counts[item["label"]] += 1
    print(f"\n  Labels:")
    for k in ["fatal_violation", "valid", "minor_issue"]:
        v = label_counts.get(k, 0)
        print(f"    {k:20s} {v:5d}  ({v/len(clean)*100:.1f}%)")

    fatal_items = [i for i in clean if i["label"] == "fatal_violation"]
    c_counts = defaultdict(int)
    for item in fatal_items:
        c_counts[item.get("violated_constraint", "")] += 1
    print(f"\n  Fatal constraints:")
    for k, v in sorted(c_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:30s} {v:5d}  ({v/len(fatal_items)*100:.1f}%)")

    cat_counts = defaultdict(int)
    for item in clean:
        cat_counts[item.get("error_category", "")] += 1
    print(f"\n  Error categories:")
    for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:40s} {v:5d}  ({v/len(clean)*100:.1f}%)")

    diff_counts = defaultdict(int)
    for item in clean:
        diff_counts[item.get("difficulty", "")] += 1
    print(f"\n  Difficulty:")
    for k in ["easy", "medium", "hard"]:
        v = diff_counts.get(k, 0)
        print(f"    {k:20s} {v:5d}  ({v/len(clean)*100:.1f}%)")

    # Check minor_issue quality
    minors = [i for i in clean if i["label"] == "minor_issue"]
    if minors:
        print(f"\n  Minor issue quality:")
        for field in ["why_invalid", "minimal_fix", "violated_principle", "what_changed"]:
            filled = sum(1 for i in minors if i.get(field, "").strip())
            print(f"    {field:30s} {filled}/{len(minors)} filled ({filled/len(minors)*100:.0f}%)")

    # Save manifest
    manifest = {
        "rebalanced_at": datetime.now().isoformat(),
        "source_file": SOURCE_FILE,
        "phase_a_file": PHASE_A_OUTPUT,
        "phase_b_file": PHASE_B_OUTPUT,
        "total": len(clean),
        "label_distribution": dict(label_counts),
        "constraint_distribution": dict(c_counts),
        "category_distribution": dict(cat_counts),
        "difficulty_distribution": dict(diff_counts),
    }
    manifest_path = FINAL_OUTPUT.replace(".json", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n  Manifest: {manifest_path}")

    return clean


# ============================================================
# CLI
# ============================================================

def load_seed_data(seed_dir):
    with open(os.path.join(seed_dir, "constraints.json")) as f:
        constraints = json.load(f)
    with open(os.path.join(seed_dir, "domains.json")) as f:
        domains = json.load(f)
    return constraints, domains


def main():
    parser = argparse.ArgumentParser(description="Dataset Rebalancer")
    parser.add_argument("--phase", choices=["a", "b", "c", "all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--seed-data", default=SEED_DATA_DIR)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--rpm", type=int, default=100)
    parser.add_argument("--items-per-constraint", type=int, default=30,
                        help="Fatal items per missing constraint (Phase A)")
    parser.add_argument("--valid-per-domain", type=int, default=50,
                        help="Valid items per domain (Phase B)")
    parser.add_argument("--minor-per-domain", type=int, default=40,
                        help="Minor items per domain (Phase B)")
    parser.add_argument("--fatal-count", type=int, default=None,
                        help="Exact number of fatal items to keep (Phase C)")
    args = parser.parse_args()

    constraints, domains = load_seed_data(args.seed_data)

    if args.phase in ("a", "all"):
        phase_a_generate_missing(constraints, domains, args.model, args.concurrency, args.rpm, args.items_per_constraint)

    if args.phase in ("b", "all"):
        phase_b_generate_valid_minor(constraints, domains, args.model, args.concurrency, args.rpm,
                                      args.valid_per_domain, args.minor_per_domain)

    if args.phase in ("c", "all"):
        phase_c_rebalance(fatal_count=args.fatal_count)


if __name__ == "__main__":
    main()
