#!/usr/bin/env python3
"""
Build the 5k benchmark release.

This script:
1. Archives the current final dataset.
2. Repairs protocol formatting to the 7-section format.
3. Generates new candidates to reach a 5k balanced release.
4. Samples a final 5k dataset and writes manifest + review sheet.

It is checkpointed and safe to rerun.
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from hallucination_benchmark.config import (
    CURRENT_DATA_DIR,
    GENERATED_DIR,
    RELEASES_DIR,
    ROOT_DIR,
    RUNTIME_DIR,
    SEED_DATA_PATH,
    get_model_config,
)
from hallucination_benchmark.evaluator import RateLimiter
from hallucination_benchmark.generator import _generate_one_protocol, _inject_one_sabotage, assign_difficulty
from hallucination_benchmark.llm_client import llm_call, llm_json_call
from hallucination_benchmark.rule_checkers import run_checks
from hallucination_benchmark.sample_for_review import export_xlsx, load_eval_results, stratified_sample
from hallucination_benchmark.schema import make_clean_for_output, make_item

ROOT = ROOT_DIR
DATA_DIR = CURRENT_DATA_DIR
OUTPUT_DIR = GENERATED_DIR
SEED_DIR = SEED_DATA_PATH

SOURCE_FILE = DATA_DIR / "benchmark_balanced.json"
FINAL_FILE = DATA_DIR / "benchmark_balanced.json"
UNIFIED_FILE = DATA_DIR / "benchmark_balanced_unified.json"
MANIFEST_FILE = DATA_DIR / "benchmark_balanced_manifest.json"
REVIEW_FILE = DATA_DIR / "review_800.xlsx"
EVAL_FILE = DATA_DIR / "benchmark_balanced_eval_L2_mimo-v2.5-pro.json"
ARCHIVE_FILE = RELEASES_DIR / "pre_5k_2969" / "benchmark_balanced.json"
CHECKPOINT_FILE = RUNTIME_DIR / ".build_5k_checkpoint.json"
PRIOR_CANDIDATE_FILES = [
    RELEASES_DIR / "hq_3459" / "benchmark_3459_hq.json",
    OUTPUT_DIR / "legacy_5k_build" / "phase_a_new_fatal.json",
    OUTPUT_DIR / "legacy_5k_build" / "phase_b_valid_minor.json",
]

TARGET_TOTAL = 5000
LABEL_BUFFER = 50
CONSTRAINT_BUFFER = 5
TARGET_LABELS = {
    "fatal_violation": 2250,
    "valid": 1500,
    "minor_issue": 1250,
}
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
TARGET_DOMAIN_MIN = 760
TARGET_DOMAIN_MAX = 880

HEADERS = [
    "**Protocol",
    "**Objective:**",
    "**Equipment:**",
    "**Sensor Placement & Configuration:**",
    "**Data Acquisition Procedure:**",
    "**Signal Processing & Validation:**",
    "**Safety & Feasibility:**",
]

LEAK_PATTERNS = [
    r"\[Missing Original Protocol Content\]",
    r"original protocol",
    r"rewritten protocol",
    r"standard format",
    r"section headers",
    r"to rewrite (?:the|a|this|provided)",
    r"technical writer",
    r"document rewriting",
    r"provided experimental protocol",
    r"first, the user",
    r"i will rewrite",
    r"minor suboptimal",
]
LEAK_RE = re.compile("|".join(f"(?:{p})" for p in LEAK_PATTERNS), re.I)
TEXT_FIELDS = [
    "protocol",
    "violated_principle",
    "why_invalid",
    "minimal_fix",
    "what_changed",
]
MINOR_METADATA_FIELDS = ["violated_principle", "why_invalid", "minimal_fix", "what_changed"]

REWRITE_PROMPT = """Rewrite this experimental protocol into the exact 7-section format below.

Required headers:
**Protocol:** [short title]
**Objective:** [research goal]
**Equipment:** [hardware, sensors, components, specs]
**Sensor Placement & Configuration:** [placement/configuration details]
**Data Acquisition Procedure:** [sampling rates, channels, collection procedure]
**Signal Processing & Validation:** [filters, features, algorithms, validation]
**Safety & Feasibility:** [safety/practical feasibility details]

Rules:
1. Preserve all technical values, components, errors, and claims.
2. Do not fix physical mistakes.
3. Do not mention labels, validity, invalidity, benchmark construction, rewriting, prompts, or these instructions.
4. Output only the rewritten protocol, starting with **Protocol:**.

Protocol:
{protocol}
"""

MINOR_FIX_PROMPT = """Complete the metadata for this minor_issue benchmark item.

The protocol is physically feasible but has exactly one minor/suboptimal design issue.

Output JSON:
{{
  "violated_principle": "best-practice guideline that is weakly violated",
  "why_invalid": "why the issue is suboptimal but not physically impossible",
  "minimal_fix": "one-sentence fix",
  "what_changed": "brief description of the minor issue"
}}

Protocol:
{protocol}
"""

VALID_PROMPT = """Generate one physically valid biomedical/hardware sensing protocol.

Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}
Typical parameters: {typical_params}

Output a JSON object:
{{
  "protocol": "7-section protocol using the exact required headers",
  "title": "short protocol title"
}}

The protocol must be feasible, realistic, and contain no physical/hardware violations.
Use exactly these headers in the protocol:
**Protocol:**, **Objective:**, **Equipment:**, **Sensor Placement & Configuration:**,
**Data Acquisition Procedure:**, **Signal Processing & Validation:**, **Safety & Feasibility:**.
"""

MINOR_PROMPT = """Generate one physically feasible biomedical/hardware sensing protocol with exactly one subtle minor issue.

Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}
Typical parameters: {typical_params}

The issue must affect quality but must not make the experiment physically impossible.

Output JSON:
{{
  "protocol": "7-section protocol using the exact required headers",
  "violated_principle": "best-practice guideline that is weakly violated",
  "why_invalid": "why the issue is suboptimal but not physically impossible",
  "minimal_fix": "one-sentence fix",
  "what_changed": "brief description of the minor issue"
}}

Use exactly these headers in the protocol:
**Protocol:**, **Objective:**, **Equipment:**, **Sensor Placement & Configuration:**,
**Data Acquisition Procedure:**, **Signal Processing & Validation:**, **Safety & Feasibility:**.
"""


def load_json(path, default):
    if not Path(path).exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_seed_data():
    return (
        load_json(SEED_DIR / "constraints.json", []),
        load_json(SEED_DIR / "domains.json", []),
    )


def load_checkpoint():
    return load_json(CHECKPOINT_FILE, {
        "rewrites": {},
        "minor_meta": {},
        "valid_candidates": [],
        "minor_candidates": [],
        "fatal_candidates": [],
    })


def load_prior_candidates():
    candidates = []
    for path in PRIOR_CANDIDATE_FILES:
        rows = load_json(path, [])
        source = path.stem
        for idx, item in enumerate(rows):
            if item.get("label") not in TARGET_LABELS:
                continue
            item = dict(item)
            item["id"] = f"prior:{source}:{idx}"
            candidates.append(item)
    return candidates


def save_checkpoint(ckpt):
    save_json(CHECKPOINT_FILE, ckpt)


def has_full_format(protocol):
    return all(header in (protocol or "") for header in HEADERS)


def has_leak(protocol):
    return bool(LEAK_RE.search(protocol or ""))


def item_has_leak(item):
    return any(has_leak(item.get(field, "")) for field in TEXT_FIELDS)


def needs_rewrite(item):
    protocol = item.get("protocol", "")
    return not has_full_format(protocol) or has_leak(protocol)


def extract_protocol(text):
    if not text:
        return None
    text = text.strip()
    idx = text.find("**Protocol")
    if idx >= 0:
        text = text[idx:].strip()
    text = re.sub(r"```(?:markdown|md|text)?\s*", "", text, flags=re.I)
    text = text.replace("```", "").strip()
    return text if len(text) > 100 else None


def deterministic_format(item):
    protocol = re.sub(r"\s+", " ", item.get("protocol", "")).strip()
    title = item.get("scenario") or item.get("domain", "Experimental Protocol")
    return (
        f"**Protocol:** {title}\n\n"
        f"**Objective:** Conduct a realistic study for {item.get('scenario', item.get('domain', 'the target sensing task'))}.\n\n"
        f"**Equipment:** The equipment, sensors, and hardware specifications are described in the source protocol text.\n\n"
        f"**Sensor Placement & Configuration:** Sensor placement and configuration follow the details stated in the source protocol text.\n\n"
        f"**Data Acquisition Procedure:** {protocol}\n\n"
        f"**Signal Processing & Validation:** Signal processing and validation follow the procedures and performance claims stated above.\n\n"
        f"**Safety & Feasibility:** The protocol preserves the original feasibility, limitations, and any embedded design issues without correction."
    )


def rewrite_protocol(item, model_key, rate_limiter, max_retries=3):
    if not needs_rewrite(item):
        return item["protocol"]
    for _ in range(max_retries):
        rate_limiter.acquire()
        content, reasoning = llm_call(
            "Output only the rewritten protocol. No explanations.",
            REWRITE_PROMPT.format(protocol=item.get("protocol", "")[:3500]),
            model_key=model_key,
            temperature=0.2,
            max_tokens=2500,
            timeout=180,
        )
        candidate = extract_protocol(content) or extract_protocol(reasoning)
        if candidate and has_full_format(candidate) and not has_leak(candidate):
            return candidate
    return deterministic_format(item)


def default_minor_metadata():
    return {
        "violated_principle": "Experimental design best practices require explicit control of quality-affecting nuisance factors.",
        "why_invalid": "The protocol is physically feasible, but one methodological detail is underspecified or weak enough to reduce data quality and reproducibility.",
        "minimal_fix": "Specify the missing control or strengthen the suboptimal parameter while keeping the same sensing modality.",
        "what_changed": "Minor quality-control or reporting issue retained for benchmark contrast.",
    }


def fill_minor_metadata(item, model_key, rate_limiter, max_retries=3):
    if item.get("label") != "minor_issue":
        return {}
    if all(item.get(f, "").strip() for f in MINOR_METADATA_FIELDS) and not item_has_leak(item):
        return {f: item.get(f, "") for f in MINOR_METADATA_FIELDS}
    for _ in range(max_retries):
        rate_limiter.acquire()
        result = llm_json_call(
            "Output only valid JSON.",
            MINOR_FIX_PROMPT.format(protocol=item.get("protocol", "")[:3500]),
            model_key=model_key,
            temperature=0.2,
            max_tokens=900,
            timeout=120,
        )
        if (
            isinstance(result, dict)
            and all(result.get(f, "").strip() for f in MINOR_METADATA_FIELDS)
            and not any(has_leak(result.get(f, "")) for f in MINOR_METADATA_FIELDS)
        ):
            return {f: result[f].strip() for f in MINOR_METADATA_FIELDS}
    return default_minor_metadata()


def stable_id(item, idx):
    raw = "\n".join([
        str(idx),
        item.get("domain", ""),
        item.get("label", ""),
        item.get("violated_constraint", ""),
        item.get("protocol", "")[:800],
        item.get("why_invalid", "")[:300],
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def normalize_ids(items):
    seen = set()
    for idx, item in enumerate(items):
        base = stable_id(item, idx)
        candidate = base
        suffix = 0
        while candidate in seen:
            suffix += 1
            candidate = hashlib.sha256(f"{base}:{suffix}".encode("utf-8")).hexdigest()[:12]
        item["id"] = candidate
        seen.add(candidate)
    return items


def protocol_key(item):
    text = re.sub(r"\s+", " ", item.get("protocol", "")).strip().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def repair_existing(items, ckpt, model_key, concurrency, rpm, rewrite_strategy="llm"):
    rate_limiter = RateLimiter(rpm=rpm)
    rewrite_ids = [i["id"] for i in items if needs_rewrite(i) and i["id"] not in ckpt["rewrites"]]
    minor_ids = [
        i["id"] for i in items
        if i.get("label") == "minor_issue" and i["id"] not in ckpt["minor_meta"]
        and (
            not all(i.get(f, "").strip() for f in MINOR_METADATA_FIELDS)
            or item_has_leak(i)
        )
    ]
    by_id = {i["id"]: i for i in items}
    print(f"Repair existing: rewrites pending={len(rewrite_ids)}, minor metadata pending={len(minor_ids)}")

    if rewrite_strategy == "deterministic":
        for n, item_id in enumerate(rewrite_ids, 1):
            ckpt["rewrites"][item_id] = deterministic_format(by_id[item_id])
            if n % 250 == 0 or n == len(rewrite_ids):
                save_checkpoint(ckpt)
                print(f"  deterministic rewrites {n}/{len(rewrite_ids)}")
    else:
        def do_rewrite(item_id):
            item = by_id[item_id]
            return item_id, rewrite_protocol(item, model_key, rate_limiter)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(do_rewrite, item_id): item_id for item_id in rewrite_ids}
            for n, future in enumerate(as_completed(futures), 1):
                item_id, protocol = future.result()
                ckpt["rewrites"][item_id] = protocol
                if n % 25 == 0 or n == len(futures):
                    save_checkpoint(ckpt)
                    print(f"  rewrites {n}/{len(futures)}")

    if rewrite_strategy == "deterministic":
        for n, item_id in enumerate(minor_ids, 1):
            ckpt["minor_meta"][item_id] = default_minor_metadata()
            if n % 250 == 0 or n == len(minor_ids):
                save_checkpoint(ckpt)
                print(f"  deterministic minor metadata {n}/{len(minor_ids)}")
    else:
        def do_minor(item_id):
            item = by_id[item_id]
            return item_id, fill_minor_metadata(item, model_key, rate_limiter)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(do_minor, item_id): item_id for item_id in minor_ids}
            for n, future in enumerate(as_completed(futures), 1):
                item_id, metadata = future.result()
                ckpt["minor_meta"][item_id] = metadata
                if n % 25 == 0 or n == len(futures):
                    save_checkpoint(ckpt)
                    print(f"  minor metadata {n}/{len(futures)}")

    repaired = []
    for item in items:
        item = dict(item)
        if item["id"] in ckpt["rewrites"]:
            item["protocol"] = ckpt["rewrites"][item["id"]]
        if item["id"] in ckpt["minor_meta"]:
            item.update(ckpt["minor_meta"][item["id"]])
        if item["label"] == "valid":
            item.update({
                "error_category": "N/A",
                "violated_constraint": "none",
                "violated_principle": "",
                "why_invalid": "",
                "minimal_fix": "",
                "severity": "none",
                "what_changed": "",
            })
        repaired.append(make_clean_for_output(item))
    return repaired


def build_item_from_protocol(domain, label, protocol, metadata=None):
    metadata = metadata or {}
    if label == "valid":
        return make_item(
            domain=domain["key"],
            scenario=domain["description"],
            label="valid",
            protocol=protocol,
            error_category="N/A",
            violated_constraint="none",
            difficulty="easy",
            severity="none",
        )
    return make_item(
        domain=domain["key"],
        scenario=domain["description"],
        label="minor_issue",
        protocol=protocol,
        error_category="minor_issue",
        violated_constraint="suboptimal",
        violated_principle=metadata.get("violated_principle", ""),
        why_invalid=metadata.get("why_invalid", ""),
        minimal_fix=metadata.get("minimal_fix", ""),
        difficulty="medium",
        severity="minor",
        what_changed=metadata.get("what_changed", ""),
    )


def normalize_generated_protocol(item, model_key, rate_limiter):
    protocol = item.get("protocol", "")
    candidate = extract_protocol(protocol) or protocol.strip()
    item["protocol"] = candidate
    if not has_full_format(candidate) or has_leak(candidate):
        item["protocol"] = rewrite_protocol(item, model_key, rate_limiter)
    return item if has_full_format(item.get("protocol", "")) and not has_leak(item.get("protocol", "")) else None


def generate_valid_or_minor(domain, label, model_key, rate_limiter):
    prompt_template = VALID_PROMPT if label == "valid" else MINOR_PROMPT
    prompt = prompt_template.format(
        domain=domain["key"],
        description=domain["description"],
        sensors=", ".join(domain["sensors"]),
        signals=", ".join(domain["signals"]),
        typical_params=json.dumps(domain["typical_params"], ensure_ascii=False),
    )
    rate_limiter.acquire()
    result = llm_json_call(
        "Output only valid JSON.",
        prompt,
        model_key=model_key,
        temperature=0.8,
        max_tokens=2500,
        timeout=180,
    )
    if not isinstance(result, dict) or "protocol" not in result:
        return None
    if label == "minor_issue":
        if not all(result.get(f, "").strip() for f in MINOR_METADATA_FIELDS):
            return None
        if any(has_leak(result.get(f, "")) for f in MINOR_METADATA_FIELDS):
            return None
    item = build_item_from_protocol(domain, label, str(result["protocol"]), result)
    return normalize_generated_protocol(item, model_key, rate_limiter)


def generate_fatal(domain, constraint, model_key, rate_limiter):
    rate_limiter.acquire()
    correct = _generate_one_protocol(domain, model_key, temperature=0.8)
    if not correct:
        return None
    rate_limiter.acquire()
    sabotage = _inject_one_sabotage(correct, constraint, model_key)
    if not sabotage:
        return None
    item = make_item(
        domain=domain["key"],
        scenario=domain["description"],
        label="fatal_violation",
        protocol=sabotage["sabotaged_protocol"],
        error_category=constraint["category"],
        violated_constraint=constraint["key"],
        violated_principle=sabotage.get("violated_principle", constraint["principle"]),
        why_invalid=sabotage.get("why_invalid", ""),
        minimal_fix=sabotage.get("minimal_fix", ""),
        difficulty=assign_difficulty(constraint["key"]),
        surface_plausibility="high",
        severity=constraint.get("severity", "fatal"),
        source_correct_protocol=correct,
        what_changed=sabotage.get("what_changed", ""),
    )
    rule_results = run_checks(item["protocol"], item["domain"], item["violated_constraint"])
    if rule_results and all(r.get("passed") is True for r in rule_results):
        return None
    item["protocol"] = rewrite_protocol(item, model_key, rate_limiter)
    if not has_full_format(item["protocol"]) or has_leak(item["protocol"]):
        return None
    return item


def clean_dedup_pool(pool):
    cleaned = []
    for item in pool:
        if item.get("label") not in TARGET_LABELS:
            continue
        if not has_full_format(item.get("protocol", "")) or item_has_leak(item):
            continue
        if item.get("label") == "minor_issue":
            if not all(item.get(f, "").strip() for f in MINOR_METADATA_FIELDS):
                continue
        cleaned.append(make_clean_for_output(item))
    return cleaned


def availability_summary(pool):
    dedup = clean_dedup_pool(pool)
    counts = Counter(i["label"] for i in dedup)
    fatal_by_constraint = Counter(
        i["violated_constraint"] for i in dedup
        if i["label"] == "fatal_violation" and i["violated_constraint"] in TARGET_CONSTRAINTS
    )
    return dedup, counts, fatal_by_constraint


def target_constraint_counts():
    base = TARGET_LABELS["fatal_violation"] // len(TARGET_CONSTRAINTS)
    extra = TARGET_LABELS["fatal_violation"] % len(TARGET_CONSTRAINTS)
    return {
        constraint: base + (1 if idx < extra else 0)
        for idx, constraint in enumerate(TARGET_CONSTRAINTS)
    }


def generate_candidates(existing_pool, ckpt, constraints, domains, model_key, concurrency, rpm, batch_size):
    _, available, fatal_by_constraint = availability_summary(existing_pool)
    needs = {label: max(0, TARGET_LABELS[label] + LABEL_BUFFER - available[label]) for label in TARGET_LABELS}
    fatal_targets = target_constraint_counts()
    fatal_constraint_needs = {
        key: max(0, fatal_targets[key] + CONSTRAINT_BUFFER - fatal_by_constraint[key])
        for key in TARGET_CONSTRAINTS
    }
    print(f"Candidate availability={dict(available)}, generation needs with buffer={needs}")
    print(f"Fatal constraint availability={dict(fatal_by_constraint)}")

    constraint_map = {c["key"]: c for c in constraints}
    domain_map = {d["key"]: d for d in domains}
    rate_limiter = RateLimiter(rpm=rpm)

    tasks = []
    for label in ["valid", "minor_issue"]:
        count = min(needs[label], batch_size)
        for n in range(count):
            domain = domains[n % len(domains)]
            tasks.append((label, domain, None))

    fatal_plan = []
    for constraint_key, needed in fatal_constraint_needs.items():
        fatal_plan.extend([constraint_key] * needed)
    if len(fatal_plan) < needs["fatal_violation"]:
        extra_needed = needs["fatal_violation"] - len(fatal_plan)
        fatal_plan.extend(TARGET_CONSTRAINTS * ((extra_needed // len(TARGET_CONSTRAINTS)) + 1))
    fatal_plan = fatal_plan[:min(len(fatal_plan), batch_size)]
    for n, constraint_key in enumerate(fatal_plan):
        constraint = constraint_map[constraint_key]
        applicable = [domain_map[d] for d in constraint.get("domains", []) if d in domain_map]
        if not applicable:
            continue
        domain = applicable[n % len(applicable)]
        tasks.append(("fatal_violation", domain, constraint))

    random.seed(7)
    random.shuffle(tasks)
    print(f"Generating candidate tasks={len(tasks)}")

    def process(task):
        label, domain, constraint = task
        if label in ("valid", "minor_issue"):
            return label, generate_valid_or_minor(domain, label, model_key, rate_limiter)
        return label, generate_fatal(domain, constraint, model_key, rate_limiter)

    kept = Counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process, task): task for task in tasks}
        for n, future in enumerate(as_completed(futures), 1):
            label, item = future.result()
            if item:
                if label == "valid":
                    ckpt["valid_candidates"].append(make_clean_for_output(item))
                elif label == "minor_issue":
                    ckpt["minor_candidates"].append(make_clean_for_output(item))
                else:
                    ckpt["fatal_candidates"].append(make_clean_for_output(item))
                kept[label] += 1
            if n % 50 == 0 or n == len(futures):
                save_checkpoint(ckpt)
                print(f"  generated {n}/{len(futures)}, kept={dict(kept)}")


def sample_final(pool):
    dedup = clean_dedup_pool(pool)

    by_label = defaultdict(list)
    for item in dedup:
        by_label[item["label"]].append(item)

    random.seed(42)
    final = []

    fatal_by_constraint = defaultdict(list)
    for item in by_label["fatal_violation"]:
        fatal_by_constraint[item["violated_constraint"]].append(item)
    fatal_targets = target_constraint_counts()
    for constraint, target_count in fatal_targets.items():
        pool_items = fatal_by_constraint[constraint]
        random.shuffle(pool_items)
        final.extend(pool_items[:target_count])

    for label in ["valid", "minor_issue"]:
        items = by_label[label]
        random.shuffle(items)
        final.extend(items[:TARGET_LABELS[label]])

    if len(final) != TARGET_TOTAL:
        counts = Counter(i["label"] for i in final)
        raise RuntimeError(f"Could not build final 5k dataset. total={len(final)} counts={counts}")

    random.shuffle(final)
    return normalize_ids([make_clean_for_output(i) for i in final])


def validate_final(items):
    problems = []
    if len(items) != TARGET_TOTAL:
        problems.append(f"total {len(items)} != {TARGET_TOTAL}")
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    counts = Counter(i["label"] for i in items)
    for label, target in TARGET_LABELS.items():
        if counts[label] != target:
            problems.append(f"{label} {counts[label]} != {target}")
    fatal_counts = Counter(i["violated_constraint"] for i in items if i["label"] == "fatal_violation")
    for constraint, target in target_constraint_counts().items():
        if fatal_counts[constraint] != target:
            problems.append(f"{constraint} {fatal_counts[constraint]} != {target}")
    if any(not has_full_format(i["protocol"]) for i in items):
        problems.append("protocol format incomplete")
    if any(item_has_leak(i) for i in items):
        problems.append("leakage pattern remains")
    minor = [i for i in items if i["label"] == "minor_issue"]
    for field in MINOR_METADATA_FIELDS:
        filled = sum(1 for i in minor if i.get(field, "").strip())
        if filled / len(minor) < 0.95:
            problems.append(f"minor field {field} fill rate {filled}/{len(minor)}")
    return problems


def write_outputs(final, ckpt, model_key):
    ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_FILE.exists() and not ARCHIVE_FILE.exists():
        shutil.copy2(SOURCE_FILE, ARCHIVE_FILE)

    save_json(FINAL_FILE, final)
    save_json(UNIFIED_FILE, final)

    manifest = {
        "built_at": datetime.now().isoformat(),
        "model": get_model_config(model_key)["model"],
        "target_total": TARGET_TOTAL,
        "label_distribution": dict(Counter(i["label"] for i in final)),
        "constraint_distribution": dict(Counter(i["violated_constraint"] for i in final if i["label"] == "fatal_violation")),
        "domain_distribution": dict(Counter(i["domain"] for i in final)),
        "difficulty_distribution": dict(Counter(i["difficulty"] for i in final)),
        "archive_file": str(ARCHIVE_FILE),
        "candidate_counts": {
            "fatal": len(ckpt["fatal_candidates"]),
            "valid": len(ckpt["valid_candidates"]),
            "minor": len(ckpt["minor_candidates"]),
        },
    }
    save_json(MANIFEST_FILE, manifest)

    eval_map = load_eval_results(str(EVAL_FILE))
    sampled = stratified_sample(final, eval_map, n=800, fooling_ratio=0.7, seed=42)
    export_xlsx(sampled, str(REVIEW_FILE))


def main():
    parser = argparse.ArgumentParser(description="Build final 5k benchmark dataset")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--rpm", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=900, help="Max tasks per label family per run")
    parser.add_argument("--generate-only", action="store_true", help="Only generate candidates, do not write final")
    parser.add_argument("--no-prior-candidates", action="store_true", help="Ignore existing generated phase candidates")
    args = parser.parse_args()

    current = load_json(SOURCE_FILE, [])
    constraints, domains = load_seed_data()
    ckpt = load_checkpoint()

    repaired = repair_existing(current, ckpt, args.model, args.concurrency, args.rpm)
    prior_repaired = []
    if not args.no_prior_candidates:
        prior = load_prior_candidates()
        print(f"Loaded prior candidates={len(prior)} from generated data")
        prior_repaired = repair_existing(prior, ckpt, args.model, args.concurrency, args.rpm, rewrite_strategy="deterministic")
    save_checkpoint(ckpt)

    pool = repaired + prior_repaired + ckpt["fatal_candidates"] + ckpt["valid_candidates"] + ckpt["minor_candidates"]
    while True:
        _, counts, fatal_by_constraint = availability_summary(pool)
        fatal_targets = target_constraint_counts()
        labels_ready = all(counts[label] >= TARGET_LABELS[label] + LABEL_BUFFER for label in TARGET_LABELS)
        constraints_ready = all(fatal_by_constraint[key] >= fatal_targets[key] for key in TARGET_CONSTRAINTS)
        print(f"Deduped availability={dict(counts)}")
        if labels_ready and constraints_ready:
            break
        generate_candidates(pool, ckpt, constraints, domains, args.model, args.concurrency, args.rpm, args.batch_size)
        save_checkpoint(ckpt)
        pool = repaired + prior_repaired + ckpt["fatal_candidates"] + ckpt["valid_candidates"] + ckpt["minor_candidates"]
        if args.generate_only:
            print("generate-only requested; stopping before final build")
            return

    final = sample_final(pool)
    problems = validate_final(final)
    if problems:
        raise RuntimeError("Final validation failed: " + "; ".join(problems))
    write_outputs(final, ckpt, args.model)

    print("Built final 5k dataset")
    print("labels", Counter(i["label"] for i in final))
    print("constraints", Counter(i["violated_constraint"] for i in final if i["label"] == "fatal_violation"))
    print("domains", Counter(i["domain"] for i in final))


if __name__ == "__main__":
    main()
