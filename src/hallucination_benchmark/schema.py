"""
Unified schema definition and validation for the benchmark dataset.
"""

import re
import hashlib
from typing import Optional

# ============================================================
# Valid enum values
# ============================================================
VALID_LABELS = {"valid", "fatal_violation", "minor_issue", "ambiguous"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_PLAUSIBILITIES = {"low", "medium", "high"}
VALID_SEVERITIES = {"none", "minor", "major", "fatal"}

VALID_ERROR_CATEGORIES = {
    "signal_processing_sampling_violation",
    "physical_medium_em_propagation_violation",
    "physiological_anatomical_misplacement",
    "hardware_system_performance_paradox",
    "N/A",
    "minor_issue",
}

# ============================================================
# Benchmark item schema
# ============================================================
BENCHMARK_ITEM_SCHEMA = {
    "id":                {"type": "str",  "required": True,  "auto": True},
    "domain":            {"type": "str",  "required": True},
    "scenario":          {"type": "str",  "required": True},
    "label":             {"type": "str",  "required": True,  "enum": VALID_LABELS},
    "protocol":          {"type": "str",  "required": True},
    "error_category":    {"type": "str",  "required": True,  "enum": VALID_ERROR_CATEGORIES},
    "violated_constraint": {"type": "str", "required": True},
    "violated_principle":  {"type": "str", "required": True},
    "why_invalid":       {"type": "str",  "required": True},
    "minimal_fix":       {"type": "str",  "required": True},
    "difficulty":        {"type": "str",  "required": True,  "enum": VALID_DIFFICULTIES},
    "surface_plausibility": {"type": "str", "required": True, "enum": VALID_PLAUSIBILITIES},
    "severity":          {"type": "str",  "required": True,  "enum": VALID_SEVERITIES},
    "source_correct_protocol": {"type": "str", "required": True},
    "what_changed":      {"type": "str",  "required": True},
}


def generate_id(domain: str, constraint: str, idx: int) -> str:
    """Generate deterministic 12-char hex ID."""
    raw = f"{domain}:{constraint}:{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def make_item(
    *,
    domain: str,
    scenario: str,
    label: str,
    protocol: str,
    error_category: str = "N/A",
    violated_constraint: str = "none",
    violated_principle: str = "",
    why_invalid: str = "",
    minimal_fix: str = "",
    difficulty: str = "medium",
    surface_plausibility: str = "high",
    severity: str = "none",
    source_correct_protocol: str = "",
    what_changed: str = "",
    item_id: Optional[str] = None,
    idx: int = 0,
) -> dict:
    """Construct a benchmark item with auto-generated ID if not provided."""
    if item_id is None:
        item_id = generate_id(domain, violated_constraint, idx)
    return {
        "id": item_id,
        "domain": domain,
        "scenario": scenario,
        "label": label,
        "protocol": protocol,
        "error_category": error_category,
        "violated_constraint": violated_constraint,
        "violated_principle": violated_principle,
        "why_invalid": why_invalid,
        "minimal_fix": minimal_fix,
        "difficulty": difficulty,
        "surface_plausibility": surface_plausibility,
        "severity": severity,
        "source_correct_protocol": source_correct_protocol,
        "what_changed": what_changed,
    }


def validate_item(item: dict) -> list[str]:
    """Validate a benchmark item against the schema.

    Returns a list of error strings. Empty list = valid.
    """
    errors = []

    for field, spec in BENCHMARK_ITEM_SCHEMA.items():
        if spec["required"] and field not in item:
            errors.append(f"Missing required field: {field}")
            continue
        if field not in item:
            continue
        val = item[field]
        if not isinstance(val, str):
            errors.append(f"Field '{field}' must be str, got {type(val).__name__}")
        if "enum" in spec and val not in spec["enum"]:
            errors.append(f"Field '{field}' = '{val}' not in {spec['enum']}")

    # Semantic checks
    if item.get("label") == "valid":
        if item.get("violated_constraint") not in ("none", ""):
            errors.append("Valid samples should have violated_constraint='none'")
    if item.get("label") == "fatal_violation":
        if item.get("severity") == "none":
            errors.append("Fatal violation should have severity != 'none'")

    return errors


def make_clean_for_output(item: dict) -> dict:
    """Strip any internal fields and return only the clean schema fields."""
    return {k: item.get(k, "") for k in BENCHMARK_ITEM_SCHEMA}
