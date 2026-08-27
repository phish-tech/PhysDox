#!/usr/bin/env python3
"""
Rebuild the benchmark as a clean 5k release candidate.

This rebuild is stricter than build_5k_dataset.py:
1. Protocol text must be unique after whitespace normalization.
2. Wrapper text and placeholders are rejected.
3. All seven protocol sections must be present and substantive.
4. Generation, filtering, and final quality metrics are logged for paper use.

The script is checkpointed and safe to rerun.
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from hallucination_benchmark.config import (
    CURRENT_DATA_DIR,
    EXPERIMENTS_DIR,
    GENERATED_DIR,
    RELEASES_DIR,
    ROOT_DIR,
    RUNTIME_DIR,
    SEED_DATA_PATH,
    get_model_config,
)
from hallucination_benchmark.evaluator import RateLimiter
from hallucination_benchmark.llm_client import llm_json_call
from hallucination_benchmark.sample_for_review import export_xlsx, load_eval_results, stratified_sample
from hallucination_benchmark.schema import make_clean_for_output, make_item

ROOT = ROOT_DIR
DATA_DIR = CURRENT_DATA_DIR
OUTPUT_DIR = GENERATED_DIR
SEED_DIR = SEED_DATA_PATH
DRAFT_RELEASE_DIR = RELEASES_DIR / "draft_5k_pre_clean"

SOURCE_FILE = DATA_DIR / "benchmark_balanced.json"
FINAL_FILE = DATA_DIR / "benchmark_balanced.json"
UNIFIED_FILE = DATA_DIR / "benchmark_balanced_unified.json"
MANIFEST_FILE = DATA_DIR / "benchmark_balanced_manifest.json"
REVIEW_FILE = DATA_DIR / "review_800.xlsx"
EVAL_FILE = DATA_DIR / "benchmark_balanced_eval_L2_mimo-v2.5-pro.json"
CHECKPOINT_FILE = RUNTIME_DIR / ".clean_5k_checkpoint.json"
PRE_CLEAN_ARCHIVE = DRAFT_RELEASE_DIR / "benchmark_balanced.json"

PRIOR_CANDIDATE_FILES = [
    RELEASES_DIR / "pre_5k_2969" / "benchmark_balanced.json",
    RELEASES_DIR / "hq_3459" / "benchmark_3459_hq.json",
    OUTPUT_DIR / "legacy_5k_build" / "phase_a_new_fatal.json",
    OUTPUT_DIR / "legacy_5k_build" / "phase_b_valid_minor.json",
]

TARGET_TOTAL = 5000
LABEL_BUFFER = 80
CONSTRAINT_BUFFER = 8
MIN_TAIL_BATCH = 0
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
MINOR_METADATA_FIELDS = ["violated_principle", "why_invalid", "minimal_fix", "what_changed"]

HEADERS = [
    "**Protocol:**",
    "**Objective:**",
    "**Equipment:**",
    "**Sensor Placement & Configuration:**",
    "**Data Acquisition Procedure:**",
    "**Signal Processing & Validation:**",
    "**Safety & Feasibility:**",
]
SECTION_MIN_WORDS = {
    "Protocol": 3,
    "Objective": 18,
    "Equipment": 22,
    "Sensor Placement & Configuration": 22,
    "Data Acquisition Procedure": 30,
    "Signal Processing & Validation": 25,
    "Safety & Feasibility": 18,
}
MIN_TOTAL_WORDS = 180

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

VALID_PROMPT = """Generate one original, physically valid biomedical/hardware sensing protocol.

Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}
Typical parameters: {typical_params}
Study variant: {variant}

Output JSON:
{{
  "protocol": "full Markdown protocol with exactly the required seven sections",
  "title": "short title"
}}

Hard requirements:
1. The protocol must be physically feasible and realistic.
2. Use exactly these section headers:
   **Protocol:**, **Objective:**, **Equipment:**, **Sensor Placement & Configuration:**,
   **Data Acquisition Procedure:**, **Signal Processing & Validation:**, **Safety & Feasibility:**.
3. Every section must contain concrete details; do not use placeholders, brackets, or phrases like "source protocol text", "as above", or "procedure below".
4. Include specific hardware specs, sampling rates, bit depths, placement, validation, and safety details.
5. The protocol should be 180-350 words.
6. Do not mention labels, benchmark construction, prompts, instructions, or JSON.
"""

MINOR_PROMPT = """Generate one original biomedical/hardware sensing protocol with exactly one subtle minor design issue.

Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}
Typical parameters: {typical_params}
Study variant: {variant}

The protocol must remain physically feasible; the issue should reduce data quality or reproducibility but not make the experiment impossible.

Output JSON:
{{
  "protocol": "full Markdown protocol with exactly the required seven sections",
  "violated_principle": "best-practice guideline that is weakly violated",
  "why_invalid": "why this is suboptimal but still physically feasible",
  "minimal_fix": "one-sentence fix",
  "what_changed": "brief description of the minor issue"
}}

Hard requirements:
1. Use exactly these section headers:
   **Protocol:**, **Objective:**, **Equipment:**, **Sensor Placement & Configuration:**,
   **Data Acquisition Procedure:**, **Signal Processing & Validation:**, **Safety & Feasibility:**.
2. Every section must contain concrete details; do not use placeholders, brackets, or phrases like "source protocol text", "as above", or "procedure below".
3. Include realistic hardware specs, placement, sampling, processing, validation, and safety details.
4. The protocol should be 180-350 words.
5. Do not mention labels, benchmark construction, prompts, instructions, or JSON.
"""

FATAL_PROMPT = """Generate one original biomedical/hardware sensing protocol containing exactly one fatal physical or hardware violation.

Domain: {domain}
Description: {description}
Sensors: {sensors}
Target signals: {signals}
Typical parameters: {typical_params}
Study variant: {variant}

Fatal constraint:
- Key: {constraint_key}
- Principle: {principle}
- Example violation: {example_violation}

Output JSON:
{{
  "protocol": "full Markdown protocol with exactly the required seven sections",
  "violated_principle": "the violated physical/hardware principle",
  "why_invalid": "why the protocol is physically impossible or hardware-infeasible",
  "minimal_fix": "one-sentence fix",
  "what_changed": "brief description of the injected fatal flaw"
}}

Hard requirements:
1. The protocol must look like a plausible experimental methods section, except for exactly one fatal flaw tied to the specified constraint.
2. Use exactly these section headers:
   **Protocol:**, **Objective:**, **Equipment:**, **Sensor Placement & Configuration:**,
   **Data Acquisition Procedure:**, **Signal Processing & Validation:**, **Safety & Feasibility:**.
3. Every section must contain concrete details; do not use placeholders, brackets, or phrases like "source protocol text", "as above", or "procedure below".
4. Include specific hardware specs, placement, sampling, processing, validation, and safety details.
5. The protocol should be 180-350 words.
6. Do not mention labels, benchmark construction, prompts, instructions, or JSON.
"""


def load_json(path, default):
    path = Path(path)
    if not path.exists():
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


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def word_count(text):
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_protocol(text):
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"```(?:json|markdown|md|text)?\s*", "", text, flags=re.I)
    text = text.replace("```", "").strip()
    idx = text.find("**Protocol:**")
    if idx >= 0:
        text = text[idx:].strip()
    return text


def section_contents(protocol):
    contents = {}
    for idx, header in enumerate(HEADERS):
        match = re.search(re.escape(header), protocol, re.I)
        if not match:
            return None
        start = match.end()
        end = len(protocol)
        for next_header in HEADERS[idx + 1:]:
            next_match = re.search(re.escape(next_header), protocol[start:], re.I)
            if next_match:
                end = start + next_match.start()
                break
        name = header.replace("**", "").replace(":", "")
        contents[name] = protocol[start:end].strip()
    return contents


def quality_issues(item):
    issues = []
    protocol = item.get("protocol", "") or ""
    if item.get("label") not in TARGET_LABELS:
        issues.append("invalid_label")
    if FORBIDDEN_RE.search(protocol):
        issues.append("forbidden_text")
    if word_count(protocol) < MIN_TOTAL_WORDS:
        issues.append("short_protocol")
    sections = section_contents(protocol)
    if sections is None:
        issues.append("missing_header")
    else:
        for name, content in sections.items():
            min_words = SECTION_MIN_WORDS[name]
            if word_count(content) < min_words:
                issues.append(f"weak_section:{name}")
    if item.get("label") == "minor_issue":
        for field in MINOR_METADATA_FIELDS:
            if not (item.get(field, "") or "").strip():
                issues.append(f"missing_minor_meta:{field}")
    for field in ["violated_principle", "why_invalid", "minimal_fix", "what_changed"]:
        if FORBIDDEN_RE.search(item.get(field, "") or ""):
            issues.append(f"forbidden_meta:{field}")
    return issues


def is_quality_item(item):
    return not quality_issues(item)


def protocol_key(item):
    return hashlib.sha256(normalize_text(item.get("protocol", "")).encode("utf-8")).hexdigest()


def stable_id(item, idx):
    raw = "\n".join([
        str(idx),
        item.get("domain", ""),
        item.get("label", ""),
        item.get("violated_constraint", ""),
        normalize_text(item.get("protocol", ""))[:1200],
        normalize_text(item.get("why_invalid", ""))[:400],
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


def load_seed_data():
    return (
        load_json(SEED_DIR / "constraints.json", []),
        load_json(SEED_DIR / "domains.json", []),
    )


def load_checkpoint():
    ckpt = load_json(CHECKPOINT_FILE, {})
    ckpt.setdefault("experiment_id", datetime.now().strftime("clean_5k_%Y%m%d_%H%M%S"))
    ckpt.setdefault("candidates", [])
    ckpt.setdefault("generated_attempts", 0)
    ckpt.setdefault("rejected", {})
    return ckpt


def save_checkpoint(ckpt):
    save_json(CHECKPOINT_FILE, ckpt)


def experiment_dir(ckpt):
    path = EXPERIMENTS_DIR / ckpt["experiment_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_event(ckpt, event):
    event = {"ts": datetime.now().isoformat(), **event}
    with open(experiment_dir(ckpt) / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def clean_for_label(item):
    item = make_clean_for_output(item)
    if item["label"] == "valid":
        item.update({
            "error_category": "N/A",
            "violated_constraint": "none",
            "violated_principle": "",
            "why_invalid": "",
            "minimal_fix": "",
            "severity": "none",
            "source_correct_protocol": "",
            "what_changed": "",
        })
    elif item["label"] == "minor_issue":
        item.update({
            "error_category": "minor_issue",
            "violated_constraint": "suboptimal",
            "difficulty": item.get("difficulty") or "medium",
            "surface_plausibility": item.get("surface_plausibility") or "high",
            "severity": "minor",
            "source_correct_protocol": "",
        })
    return item


def load_source_pool():
    pool = []
    sources = [SOURCE_FILE, *PRIOR_CANDIDATE_FILES]
    for path in sources:
        rows = load_json(path, [])
        for idx, item in enumerate(rows):
            if item.get("label") not in TARGET_LABELS:
                continue
            item = dict(item)
            item["_source"] = str(path)
            item["_source_idx"] = idx
            pool.append(item)
    old_ckpt = load_json(RUNTIME_DIR / ".build_5k_checkpoint.json", {})
    for key in ["fatal_candidates", "valid_candidates", "minor_candidates"]:
        for idx, item in enumerate(old_ckpt.get(key, [])):
            if item.get("label") in TARGET_LABELS:
                item = dict(item)
                item["_source"] = f"legacy_checkpoint:{key}"
                item["_source_idx"] = idx
                pool.append(item)
    return pool


def source_quality_report(items):
    reason = Counter()
    clean = []
    for item in items:
        issues = quality_issues(item)
        if issues:
            reason[issues[0]] += 1
        else:
            clean.append(item)

    seen = set()
    unique = []
    duplicate_count = 0
    for item in clean:
        key = protocol_key(item)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(item)

    return {
        "raw_total": len(items),
        "quality_rejection_reasons": dict(reason),
        "clean_raw": len(clean),
        "clean_raw_label_distribution": dict(Counter(i["label"] for i in clean)),
        "clean_unique": len(unique),
        "clean_unique_label_distribution": dict(Counter(i["label"] for i in unique)),
        "clean_unique_fatal_constraints": dict(Counter(i.get("violated_constraint") for i in unique if i["label"] == "fatal_violation")),
        "duplicate_protocols_removed_after_quality": duplicate_count,
    }


def strict_unique_pool(items):
    seen = set()
    out = []
    for item in items:
        item = clean_for_label(item)
        if not is_quality_item(item):
            continue
        key = protocol_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def target_constraint_counts():
    base = TARGET_LABELS["fatal_violation"] // len(TARGET_CONSTRAINTS)
    extra = TARGET_LABELS["fatal_violation"] % len(TARGET_CONSTRAINTS)
    return {
        constraint: base + (1 if idx < extra else 0)
        for idx, constraint in enumerate(TARGET_CONSTRAINTS)
    }


def availability(pool):
    counts = Counter(i["label"] for i in pool)
    constraints = Counter(i["violated_constraint"] for i in pool if i["label"] == "fatal_violation")
    return counts, constraints


def make_variant(domain, label, n, constraint_key=None):
    settings = [
        "pilot study with 12 participants",
        "cross-session validation with two visits",
        "wearable prototype comparison",
        "controlled laboratory protocol",
        "ambulatory recording session",
        "leave-one-subject-out validation",
        "multi-device calibration study",
        "motion artifact stress test",
        "low-power embedded implementation",
        "clinical screening workflow",
        "sports physiology monitoring task",
        "home-use feasibility study",
    ]
    return f"{domain['key']} | {label} | {constraint_key or 'none'} | variant {n} | {settings[n % len(settings)]}"


def build_item_from_json(result, domain, label, constraint=None):
    protocol = extract_protocol(result.get("protocol", ""))
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
    if label == "minor_issue":
        return make_item(
            domain=domain["key"],
            scenario=domain["description"],
            label="minor_issue",
            protocol=protocol,
            error_category="minor_issue",
            violated_constraint="suboptimal",
            violated_principle=str(result.get("violated_principle", "")).strip(),
            why_invalid=str(result.get("why_invalid", "")).strip(),
            minimal_fix=str(result.get("minimal_fix", "")).strip(),
            difficulty="medium",
            severity="minor",
            what_changed=str(result.get("what_changed", "")).strip(),
        )
    return make_item(
        domain=domain["key"],
        scenario=domain["description"],
        label="fatal_violation",
        protocol=protocol,
        error_category=constraint["category"],
        violated_constraint=constraint["key"],
        violated_principle=str(result.get("violated_principle", constraint["principle"])).strip(),
        why_invalid=str(result.get("why_invalid", "")).strip(),
        minimal_fix=str(result.get("minimal_fix", "")).strip(),
        difficulty="hard" if constraint["key"] in {"accuracy_impossibility", "snr_floor", "latency_constraint"} else "medium",
        surface_plausibility="high",
        severity=constraint.get("severity", "fatal"),
        source_correct_protocol="",
        what_changed=str(result.get("what_changed", "")).strip(),
    )


def generate_one(task, model_key, rate_limiter):
    label, domain, constraint, variant_n = task
    base = {
        "domain": domain["key"],
        "description": domain["description"],
        "sensors": ", ".join(domain["sensors"]),
        "signals": ", ".join(domain["signals"]),
        "typical_params": json.dumps(domain["typical_params"], ensure_ascii=False),
        "variant": make_variant(domain, label, variant_n, constraint["key"] if constraint else None),
    }
    if label == "valid":
        prompt = VALID_PROMPT.format(**base)
    elif label == "minor_issue":
        prompt = MINOR_PROMPT.format(**base)
    else:
        prompt = FATAL_PROMPT.format(
            **base,
            constraint_key=constraint["key"],
            principle=constraint["principle"],
            example_violation=constraint["example_violation"],
        )
    rate_limiter.acquire()
    result = llm_json_call(
        "You are a biomedical engineering methods writer. Return only valid JSON.",
        prompt,
        model_key=model_key,
        temperature=0.85,
        max_tokens=2600,
        timeout=220,
    )
    if not isinstance(result, dict):
        return label, None, ["json_parse_failed"]
    item = clean_for_label(build_item_from_json(result, domain, label, constraint))
    issues = quality_issues(item)
    if issues:
        return label, None, issues
    return label, item, []


def generation_needs(pool):
    counts, constraint_counts = availability(pool)
    needs = {label: max(0, TARGET_LABELS[label] + LABEL_BUFFER - counts[label]) for label in TARGET_LABELS}
    target_constraints = target_constraint_counts()
    constraint_needs = {
        key: max(0, target_constraints[key] + CONSTRAINT_BUFFER - constraint_counts[key])
        for key in TARGET_CONSTRAINTS
    }
    return needs, constraint_needs


def build_tasks(pool, constraints, domains, batch_size, start_n):
    needs, constraint_needs = generation_needs(pool)
    domain_map = {d["key"]: d for d in domains}
    constraint_map = {c["key"]: c for c in constraints}
    tasks = []

    for label in ["valid", "minor_issue"]:
        count = min(needs[label], batch_size)
        if 0 < count < MIN_TAIL_BATCH:
            count = min(MIN_TAIL_BATCH, batch_size)
        for offset in range(count):
            domain = domains[(start_n + offset) % len(domains)]
            tasks.append((label, domain, None, start_n + offset))

    fatal_plan = []
    for constraint_key, needed in constraint_needs.items():
        fatal_plan.extend([constraint_key] * needed)
    if 0 < len(fatal_plan) < MIN_TAIL_BATCH:
        original = list(fatal_plan)
        while len(fatal_plan) < MIN_TAIL_BATCH:
            fatal_plan.extend(original)
    if len(fatal_plan) < needs["fatal_violation"]:
        extra = needs["fatal_violation"] - len(fatal_plan)
        fatal_plan.extend(TARGET_CONSTRAINTS * ((extra // len(TARGET_CONSTRAINTS)) + 1))
    for offset, constraint_key in enumerate(fatal_plan[:batch_size]):
        constraint = constraint_map[constraint_key]
        applicable = [domain_map[d] for d in constraint.get("domains", []) if d in domain_map]
        if not applicable:
            continue
        domain = applicable[(start_n + offset) % len(applicable)]
        tasks.append(("fatal_violation", domain, constraint, start_n + offset))

    random.Random(17 + start_n).shuffle(tasks)
    return tasks


def sample_final(pool):
    by_label = defaultdict(list)
    for item in pool:
        by_label[item["label"]].append(item)

    random.seed(42)
    final = []
    fatal_by_constraint = defaultdict(list)
    for item in by_label["fatal_violation"]:
        fatal_by_constraint[item["violated_constraint"]].append(item)
    for constraint, target in target_constraint_counts().items():
        candidates = fatal_by_constraint[constraint]
        random.shuffle(candidates)
        final.extend(candidates[:target])
    for label in ["valid", "minor_issue"]:
        candidates = by_label[label]
        random.shuffle(candidates)
        final.extend(candidates[:TARGET_LABELS[label]])
    if len(final) != TARGET_TOTAL:
        raise RuntimeError(f"Cannot sample final: total={len(final)} labels={Counter(i['label'] for i in final)}")
    random.shuffle(final)
    return normalize_ids([clean_for_label(i) for i in final])


def final_quality_report(items):
    protocols = [normalize_text(i.get("protocol", "")) for i in items]
    duplicate_rows = len(protocols) - len(set(protocols))
    issues = Counter()
    for item in items:
        for issue in quality_issues(item):
            issues[issue] += 1
    return {
        "total": len(items),
        "unique_ids": len({i["id"] for i in items}),
        "unique_protocols": len(set(protocols)),
        "duplicate_protocol_rows": duplicate_rows,
        "label_distribution": dict(Counter(i["label"] for i in items)),
        "constraint_distribution": dict(Counter(i["violated_constraint"] for i in items if i["label"] == "fatal_violation")),
        "domain_distribution": dict(Counter(i["domain"] for i in items)),
        "quality_issues": dict(issues),
        "wrapper_hits": sum(1 for i in items if re.search(r"source protocol text|procedure below|as written above", i["protocol"], re.I)),
        "bracket_placeholder_hits": sum(1 for i in items if re.search(r"\[[^\]]{1,80}\]", i["protocol"])),
        "short_under_80_words": sum(1 for i in items if word_count(i["protocol"]) < 80),
    }


def validate_final(items):
    report = final_quality_report(items)
    problems = []
    if report["total"] != TARGET_TOTAL:
        problems.append(f"total {report['total']} != {TARGET_TOTAL}")
    if report["unique_ids"] != TARGET_TOTAL:
        problems.append("ids are not unique")
    if report["unique_protocols"] != TARGET_TOTAL:
        problems.append("protocols are not unique")
    if report["quality_issues"]:
        problems.append(f"quality issues remain: {report['quality_issues']}")
    counts = Counter(i["label"] for i in items)
    for label, target in TARGET_LABELS.items():
        if counts[label] != target:
            problems.append(f"{label} {counts[label]} != {target}")
    constraints = Counter(i["violated_constraint"] for i in items if i["label"] == "fatal_violation")
    for key, target in target_constraint_counts().items():
        if constraints[key] != target:
            problems.append(f"{key} {constraints[key]} != {target}")
    return problems, report


def write_experiment_summary(ckpt, source_report, final_report, model_key, args):
    exp_dir = experiment_dir(ckpt)
    summary = {
        "experiment_id": ckpt["experiment_id"],
        "built_at": datetime.now().isoformat(),
        "model": get_model_config(model_key)["model"],
        "target_total": TARGET_TOTAL,
        "args": vars(args),
        "quality_rules": {
            "min_total_words": MIN_TOTAL_WORDS,
            "section_min_words": SECTION_MIN_WORDS,
            "forbidden_patterns": FORBIDDEN_PATTERNS,
            "protocol_uniqueness": "exact normalized protocol text",
        },
        "source_report": source_report,
        "final_report": final_report,
        "checkpoint_generated_attempts": ckpt.get("generated_attempts", 0),
        "checkpoint_rejected": ckpt.get("rejected", {}),
    }
    save_json(exp_dir / "summary.json", summary)
    with open(exp_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(f"# Clean 5k Rebuild: {ckpt['experiment_id']}\n\n")
        f.write(f"- Built at: {summary['built_at']}\n")
        f.write(f"- Model: {summary['model']}\n")
        f.write(f"- Target: {TARGET_TOTAL}\n")
        f.write(f"- Generated attempts: {summary['checkpoint_generated_attempts']}\n\n")
        f.write("## Final Quality\n\n")
        for key, value in final_report.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n## Source Quality\n\n")
        for key, value in source_report.items():
            f.write(f"- {key}: {value}\n")


def write_outputs(final, ckpt, source_report, final_report, model_key, args):
    PRE_CLEAN_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_FILE.exists() and not PRE_CLEAN_ARCHIVE.exists():
        shutil.copy2(SOURCE_FILE, PRE_CLEAN_ARCHIVE)
    save_json(FINAL_FILE, final)
    save_json(UNIFIED_FILE, final)
    manifest = {
        "built_at": datetime.now().isoformat(),
        "model": get_model_config(model_key)["model"],
        "target_total": TARGET_TOTAL,
        "label_distribution": final_report["label_distribution"],
        "constraint_distribution": final_report["constraint_distribution"],
        "domain_distribution": final_report["domain_distribution"],
        "quality_report": final_report,
        "source_report": source_report,
        "experiment_id": ckpt["experiment_id"],
        "archive_file": str(PRE_CLEAN_ARCHIVE),
    }
    save_json(MANIFEST_FILE, manifest)
    write_experiment_summary(ckpt, source_report, final_report, model_key, args)

    eval_map = load_eval_results(str(EVAL_FILE))
    sampled = stratified_sample(final, eval_map, n=800, fooling_ratio=0.7, seed=42)
    export_xlsx(sampled, str(REVIEW_FILE))


def main():
    global LABEL_BUFFER, CONSTRAINT_BUFFER, MIN_TAIL_BATCH

    parser = argparse.ArgumentParser(description="Clean rebuild final 5k benchmark")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--rpm", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--label-buffer", type=int, default=LABEL_BUFFER)
    parser.add_argument("--constraint-buffer", type=int, default=CONSTRAINT_BUFFER)
    parser.add_argument("--min-tail-batch", type=int, default=MIN_TAIL_BATCH)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    LABEL_BUFFER = args.label_buffer
    CONSTRAINT_BUFFER = args.constraint_buffer
    MIN_TAIL_BATCH = args.min_tail_batch

    constraints, domains = load_seed_data()
    ckpt = load_checkpoint()
    source_pool = load_source_pool()
    source_report = source_quality_report(source_pool)
    log_event(ckpt, {"event": "source_quality", "report": source_report})

    pool = strict_unique_pool(source_pool + ckpt["candidates"])
    print("Initial strict unique availability", dict(Counter(i["label"] for i in pool)))
    print("Initial fatal constraints", dict(Counter(i["violated_constraint"] for i in pool if i["label"] == "fatal_violation")))

    while True:
        pool = strict_unique_pool(source_pool + ckpt["candidates"])
        counts, constraints_available = availability(pool)
        needs, constraint_needs = generation_needs(pool)
        labels_ready = all(counts[label] >= TARGET_LABELS[label] + LABEL_BUFFER for label in TARGET_LABELS)
        constraints_ready = all(
            constraints_available[key] >= target_constraint_counts()[key] + CONSTRAINT_BUFFER
            for key in TARGET_CONSTRAINTS
        )
        print("Availability", dict(counts), "needs", needs)
        print("Fatal constraints", dict(constraints_available))
        log_event(ckpt, {
            "event": "availability",
            "counts": dict(counts),
            "constraint_counts": dict(constraints_available),
            "needs": needs,
            "constraint_needs": constraint_needs,
        })
        if labels_ready and constraints_ready:
            break

        tasks = build_tasks(pool, constraints, domains, args.batch_size, ckpt["generated_attempts"])
        if not tasks:
            raise RuntimeError("No generation tasks but dataset is not ready")
        print(f"Generating tasks={len(tasks)}")
        if args.generate_only:
            print("generate-only requested; stopping before generation")
            return
        rate_limiter = RateLimiter(rpm=args.rpm)
        kept = Counter()
        rejected = Counter()
        seen = {protocol_key(i) for i in pool}

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(generate_one, task, args.model, rate_limiter): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                label, item, issues = future.result()
                ckpt["generated_attempts"] += 1
                if item and protocol_key(item) not in seen:
                    seen.add(protocol_key(item))
                    ckpt["candidates"].append(clean_for_label(item))
                    kept[label] += 1
                else:
                    reason = issues[0] if issues else "duplicate_protocol"
                    rejected[reason] += 1
                    ckpt["rejected"][reason] = ckpt["rejected"].get(reason, 0) + 1
                if n % 50 == 0 or n == len(futures):
                    save_checkpoint(ckpt)
                    print(f"  generated {n}/{len(futures)} kept={dict(kept)} rejected={dict(rejected)}")
                    log_event(ckpt, {
                        "event": "generation_progress",
                        "completed": n,
                        "total": len(futures),
                        "kept": dict(kept),
                        "rejected": dict(rejected),
                    })
        save_checkpoint(ckpt)

    final = sample_final(pool)
    problems, final_report = validate_final(final)
    if problems:
        write_experiment_summary(ckpt, source_report, final_report, args.model, args)
        raise RuntimeError("Final validation failed: " + "; ".join(problems))
    write_outputs(final, ckpt, source_report, final_report, args.model, args)
    print("Built clean 5k dataset")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
