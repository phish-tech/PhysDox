#!/usr/bin/env python3
"""Two-stage evaluation for the benchmark.

Stage 1: classify protocol as valid, minor_issue, or fatal_violation.
Stage 2: for true fatal_violation items, diagnose the 3-class physical
or hardware error category, with optional fine-grained constraints.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from hallucination_benchmark.config import CURRENT_DATA_DIR, EXPERIMENTS_DIR, get_model_config
from hallucination_benchmark.evaluator import RateLimiter
from hallucination_benchmark.llm_client import extract_json, llm_call
from hallucination_benchmark.metrics import classification_metrics, top_k_accuracy
from hallucination_benchmark.rule_checkers import CHECKER_MAP

DATA_FILE = CURRENT_DATA_DIR / "benchmark_balanced.json"
STRATEGIES = ["zero-shot", "cot", "self-cons", "tool-aug"]

LABELS = ["valid", "minor_issue", "fatal_violation"]
FINE_CONSTRAINTS = [
    "sampling_theorem",
    "sensor_bandwidth",
    "adc_resolution",
    "snr_floor",
    "latency_constraint",
    "power_budget",
    "anatomical_misplacement",
    "ematerial_block",
    "optical_obstruction",
    "temperature_range",
    "data_bandwidth",
    "accuracy_impossibility",
]
THREE_CLASS_CATEGORIES = [
    "Signal & Sampling",
    "Perception & Anatomy",
    "Physical / Hardware",
]
FINE_TO_THREE_CLASS = {
    "sampling_theorem": "Signal & Sampling",
    "sensor_bandwidth": "Signal & Sampling",
    "snr_floor": "Signal & Sampling",
    "data_bandwidth": "Signal & Sampling",
    "adc_resolution": "Signal & Sampling",
    "latency_constraint": "Signal & Sampling",
    "anatomical_misplacement": "Perception & Anatomy",
    "optical_obstruction": "Perception & Anatomy",
    "power_budget": "Physical / Hardware",
    "temperature_range": "Physical / Hardware",
    "accuracy_impossibility": "Physical / Hardware",
    "ematerial_block": "Physical / Hardware",
}

TRANSPORT_ERROR_MARKERS = (
    "API call failed",
    "HTTP Error",
    "HTTP Error 402",
    "Bad Gateway",
    "Forbidden",
    "Too Many Requests",
    "Insufficient Balance",
    "余额不足",
    "预扣费额度失败",
    "timed out",
    "timeout",
    "Remote end closed connection",
    "Connection reset",
    "Temporary failure",
)

NON_RETRYABLE_TRANSPORT_MARKERS = (
    "HTTP Error 400",
    "HTTP Error 401",
    "HTTP Error 402",
    "HTTP Error 403",
    "HTTP Error 404",
    "Insufficient Balance",
    "余额不足",
    "预扣费额度失败",
)

STAGE1_PROMPT = """You are an expert IEEE reviewer for biomedical and hardware sensing protocols.
Classify the protocol into exactly one label:
- valid: physically feasible and methodologically acceptable.
- minor_issue: physically feasible but contains a non-fatal suboptimal design issue.
- fatal_violation: physically or hardware infeasible; data collection or claimed analysis cannot work.

Do not explain. The first character of your response must be {.
Return exactly one JSON object:
{"label": "valid|minor_issue|fatal_violation", "reasoning": "brief reason"}"""

STAGE1_COT_PROMPT = """You are an expert IEEE reviewer for biomedical and hardware sensing protocols.
Classify the protocol into exactly one label:
- valid: physically feasible and methodologically acceptable.
- minor_issue: physically feasible but contains a non-fatal suboptimal design issue.
- fatal_violation: physically or hardware infeasible; data collection or claimed analysis cannot work.

Think step by step internally about signal bandwidth, sampling, sensor placement, power, latency,
material obstruction, and physiological feasibility. Do not expose chain-of-thought.
The first character of your response must be {.
Return exactly one JSON object:
{"label": "valid|minor_issue|fatal_violation", "reasoning": "brief concise rationale"}"""

STAGE1_TOOL_PROMPT = """You are an expert IEEE reviewer for biomedical and hardware sensing protocols.
Classify the protocol into exactly one label:
- valid: physically feasible and methodologically acceptable.
- minor_issue: physically feasible but contains a non-fatal suboptimal design issue.
- fatal_violation: physically or hardware infeasible; data collection or claimed analysis cannot work.

You may use the deterministic tool-check report appended after the protocol. The tool report is noisy
and incomplete; treat it as supporting evidence, not ground truth.
Do not explain. The first character of your response must be {.
Return exactly one JSON object:
{"label": "valid|minor_issue|fatal_violation", "reasoning": "brief reason"}"""

STAGE2_PROMPT = """You are diagnosing the root physical or hardware error in a fatal experimental protocol.
Choose ranked 3-class categories from this fixed list:
Signal & Sampling, Perception & Anatomy, Physical / Hardware.

Also choose ranked fine-grained constraints from this fixed list:
sampling_theorem, sensor_bandwidth, adc_resolution, snr_floor, latency_constraint, power_budget,
anatomical_misplacement, ematerial_block, optical_obstruction, temperature_range, data_bandwidth,
accuracy_impossibility.

Do not explain. The first character of your response must be {.
Return exactly one JSON object:
{
  "three_class_categories": ["most likely", "second"],
  "fine_constraints": ["most likely", "second", "third"],
  "reasoning": "brief reason"
}"""

STAGE2_COT_PROMPT = """You are diagnosing the root physical or hardware error in a fatal experimental protocol.
Choose ranked 3-class categories from this fixed list:
Signal & Sampling, Perception & Anatomy, Physical / Hardware.

Also choose ranked fine-grained constraints from this fixed list:
sampling_theorem, sensor_bandwidth, adc_resolution, snr_floor, latency_constraint, power_budget,
anatomical_misplacement, ematerial_block, optical_obstruction, temperature_range, data_bandwidth,
accuracy_impossibility.

Think step by step internally about the protocol's physical bottleneck. Do not expose chain-of-thought.
The first character of your response must be {.
Return exactly one JSON object:
{
  "three_class_categories": ["most likely", "second"],
  "fine_constraints": ["most likely", "second", "third"],
  "reasoning": "brief concise rationale"
}"""

STAGE2_TOOL_PROMPT = """You are diagnosing the root physical or hardware error in a fatal experimental protocol.
Choose ranked 3-class categories from this fixed list:
Signal & Sampling, Perception & Anatomy, Physical / Hardware.

Also choose ranked fine-grained constraints from this fixed list:
sampling_theorem, sensor_bandwidth, adc_resolution, snr_floor, latency_constraint, power_budget,
anatomical_misplacement, ematerial_block, optical_obstruction, temperature_range, data_bandwidth,
accuracy_impossibility.

You may use the deterministic tool-check report appended after the protocol. The tool report is noisy
and incomplete; treat it as supporting evidence, not ground truth.
Do not explain. The first character of your response must be {.
Return exactly one JSON object:
{
  "three_class_categories": ["most likely", "second"],
  "fine_constraints": ["most likely", "second", "third"],
  "reasoning": "brief reason"
}"""

FINE_ALIASES = {
    "sampling theorem": "sampling_theorem",
    "nyquist": "sampling_theorem",
    "sampling_theorem": "sampling_theorem",
    "sensor bandwidth": "sensor_bandwidth",
    "sensor_bandwidth": "sensor_bandwidth",
    "bandwidth": "sensor_bandwidth",
    "adc resolution": "adc_resolution",
    "adc_resolution": "adc_resolution",
    "resolution": "adc_resolution",
    "snr floor": "snr_floor",
    "snr_floor": "snr_floor",
    "signal to noise": "snr_floor",
    "latency": "latency_constraint",
    "latency constraint": "latency_constraint",
    "latency_constraint": "latency_constraint",
    "power budget": "power_budget",
    "power_budget": "power_budget",
    "battery": "power_budget",
    "anatomical misplacement": "anatomical_misplacement",
    "anatomical_misplacement": "anatomical_misplacement",
    "misplacement": "anatomical_misplacement",
    "wrong placement": "anatomical_misplacement",
    "wrong muscle": "anatomical_misplacement",
    "calf": "anatomical_misplacement",
    "gastrocnemius": "anatomical_misplacement",
    "ematerial block": "ematerial_block",
    "ematerial_block": "ematerial_block",
    "electromagnetic material": "ematerial_block",
    "metal": "ematerial_block",
    "optical obstruction": "optical_obstruction",
    "optical_obstruction": "optical_obstruction",
    "obstruction": "optical_obstruction",
    "temperature range": "temperature_range",
    "temperature_range": "temperature_range",
    "data bandwidth": "data_bandwidth",
    "data_bandwidth": "data_bandwidth",
    "throughput": "data_bandwidth",
    "accuracy impossibility": "accuracy_impossibility",
    "accuracy_impossibility": "accuracy_impossibility",
    "accuracy": "accuracy_impossibility",
}

THREE_CLASS_ALIASES = {
    "signal & sampling": "Signal & Sampling",
    "signal and sampling": "Signal & Sampling",
    "signal sampling": "Signal & Sampling",
    "sampling": "Signal & Sampling",
    "nyquist": "Signal & Sampling",
    "signal processing": "Signal & Sampling",
    "perception & anatomy": "Perception & Anatomy",
    "perception and anatomy": "Perception & Anatomy",
    "perception anatomy": "Perception & Anatomy",
    "anatomy": "Perception & Anatomy",
    "anatomical": "Perception & Anatomy",
    "optical": "Perception & Anatomy",
    "placement": "Perception & Anatomy",
    "physical / hardware": "Physical / Hardware",
    "physical hardware": "Physical / Hardware",
    "physical and hardware": "Physical / Hardware",
    "hardware": "Physical / Hardware",
    "physical": "Physical / Hardware",
    "power": "Physical / Hardware",
    "temperature": "Physical / Hardware",
    "material": "Physical / Hardware",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _coerce_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
    return [str(value)]


def normalize_label(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "valid": "valid",
        "minor": "minor_issue",
        "minor_issue": "minor_issue",
        "minorissue": "minor_issue",
        "fatal": "fatal_violation",
        "fatal_violation": "fatal_violation",
        "fatalviolation": "fatal_violation",
        "invalid": "fatal_violation",
    }
    return aliases.get(text)


def normalize_fine(value) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    slug = _slug(text)
    if slug in FINE_CONSTRAINTS:
        return slug
    lowered = text.lower().replace("_", " ")
    for alias, canonical in FINE_ALIASES.items():
        if alias in lowered or _slug(alias) == slug:
            return canonical
    return None


def normalize_three_class(value) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in THREE_CLASS_CATEGORIES:
        return text
    lowered = text.lower()
    slug = _slug(text)
    for category in THREE_CLASS_CATEGORIES:
        if _slug(category) == slug:
            return category
    for alias, canonical in THREE_CLASS_ALIASES.items():
        if alias in lowered:
            return canonical
    fine = normalize_fine(text)
    return FINE_TO_THREE_CLASS.get(fine) if fine else None


def _ranked_unique(values: list[str], normalizer, limit: int) -> list[str]:
    ranked = []
    for value in values:
        normalized = normalizer(value)
        if normalized and normalized not in ranked:
            ranked.append(normalized)
    return ranked[:limit]


def _tail(text: str, chars: int = 3000) -> str:
    return (text or "")[-chars:]


def parse_stage1(result: dict | None, raw_text: str = "") -> str | None:
    if not isinstance(result, dict):
        result = {}
    for key in ("label", "classification", "verdict", "problem_type"):
        label = normalize_label(result.get(key))
        if label:
            return label
    viable = result.get("is_physically_viable")
    if viable is True:
        return "valid"
    if viable is False:
        issue_type = normalize_label(result.get("issue_type") or result.get("severity"))
        return issue_type or "fatal_violation"

    text = _tail(raw_text)
    patterns = [
        r'"label"\s*:\s*"([^"]+)"',
        r"\blabel\s*[:=]\s*(valid|minor[_\s-]?issue|fatal[_\s-]?violation)\b",
        r"\bclassification\s*[:=]\s*(valid|minor[_\s-]?issue|fatal[_\s-]?violation)\b",
        r"\bclassified\s+as\s+(valid|minor[_\s-]?issue|fatal[_\s-]?violation)\b",
        r"\btherefore,\s*(?:the\s+protocol\s+is\s+)?(valid|minor[_\s-]?issue|fatal[_\s-]?violation)\b",
        r"\b(final|conclusion).*?\b(valid|minor[_\s-]?issue|fatal[_\s-]?violation)\b",
        r"\bi\s+think\s+(?:it(?:'s| is)?|the\s+protocol\s+is|this\s+protocol\s+is)\s+"
        r"(valid|minor[_\s-]?issue|fatal[_\s-]?violation)\b",
        r"\b(?:the\s+protocol|this\s+protocol|it)\s+seems\s+"
        r"(?:physically\s+)?(valid|minor[_\s-]?issue|fatal[_\s-]?violation)\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.I | re.S))
        if matches:
            candidate = matches[-1].groups()[-1]
            label = normalize_label(candidate)
            if label:
                return label
    return None


def parse_stage2(result: dict | None, raw_text: str = "") -> tuple[list[str], list[str]]:
    if not isinstance(result, dict):
        result = {}
    fine_values = []
    for key in ("fine_constraints", "violated_constraints", "constraints", "top_constraints", "constraint"):
        fine_values.extend(_coerce_list(result.get(key)))
    three_class_values = []
    for key in (
        "three_class_categories",
        "three_class_category",
        "root_categories",
        "root_category",
        "categories",
        "top_categories",
        "category",
    ):
        three_class_values.extend(_coerce_list(result.get(key)))

    fine_ranked = _ranked_unique(fine_values, normalize_fine, 3)
    three_class_ranked = _ranked_unique(three_class_values, normalize_three_class, 2)
    if not fine_ranked:
        tail = _tail(raw_text).lower()
        hits = []
        for alias, canonical in FINE_ALIASES.items():
            pos = tail.find(alias)
            if pos >= 0 and canonical not in [hit[1] for hit in hits]:
                hits.append((pos, canonical))
        fine_ranked = [canonical for _, canonical in sorted(hits)[:3]]
    if not three_class_ranked:
        tail = _tail(raw_text).lower()
        hits = []
        for alias, canonical in THREE_CLASS_ALIASES.items():
            pos = tail.find(alias)
            if pos >= 0 and canonical not in [hit[1] for hit in hits]:
                hits.append((pos, canonical))
        three_class_ranked = [canonical for _, canonical in sorted(hits)[:2]]
    if not three_class_ranked and fine_ranked:
        three_class_ranked = _ranked_unique(
            [FINE_TO_THREE_CLASS[fine] for fine in fine_ranked],
            normalize_three_class,
            2,
        )
    return fine_ranked, three_class_ranked


def strategy_prompt(stage: int, strategy: str) -> str:
    if stage == 1:
        if strategy == "cot":
            return STAGE1_COT_PROMPT
        if strategy == "tool-aug":
            return STAGE1_TOOL_PROMPT
        return STAGE1_PROMPT
    if strategy == "cot":
        return STAGE2_COT_PROMPT
    if strategy == "tool-aug":
        return STAGE2_TOOL_PROMPT
    return STAGE2_PROMPT


def tool_report(item: dict) -> str:
    rows = []
    protocol = item.get("protocol", "")
    domain = item.get("domain") or item.get("scenario") or ""
    for constraint, checker in CHECKER_MAP.items():
        try:
            result = checker(protocol, domain)
        except Exception as exc:
            result = {"check": constraint, "result": "tool_error", "detail": str(exc), "passed": None}
        if result.get("passed") is False or result.get("result") not in {"ok", "manual"}:
            rows.append({
                "constraint": constraint,
                "result": result.get("result"),
                "passed": result.get("passed"),
                "detail": result.get("detail") or result.get("details"),
            })
    return json.dumps({
        "domain": domain,
        "note": "Noisy deterministic checks; absence of a violation does not imply validity.",
        "findings": rows[:12],
    }, ensure_ascii=False)


def user_prompt_for(item: dict, strategy: str) -> str:
    protocol = item["protocol"]
    if strategy != "tool-aug":
        return protocol
    return "\n\n".join([
        "PROTOCOL:",
        protocol,
        "DETERMINISTIC TOOL-CHECK REPORT:",
        tool_report(item),
    ])


def call_json(
    prompt: str,
    user_prompt: str,
    model_key: str,
    model_name: str | None,
    rate_limiter: RateLimiter,
    retries: int,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    adaptive_controller=None,
):
    content = None
    reasoning = ""
    last_error = ""
    attempts = max(1, retries)
    for attempt in range(attempts):
        rate_limiter.acquire()
        content, reasoning = llm_call(
            prompt,
            user_prompt,
            model_key=model_key,
            model_name_override=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        if content and not any(marker in content for marker in TRANSPORT_ERROR_MARKERS):
            break
        if content:
            last_error = content
            content = None
        else:
            last_error = reasoning or "empty response"
            content = None
        if any(marker in last_error for marker in NON_RETRYABLE_TRANSPORT_MARKERS):
            break
        if attempt < attempts - 1:
            time.sleep(_retry_delay(attempt))
    if content is None:
        if adaptive_controller:
            if last_error == "empty response" or "empty response" in last_error:
                adaptive_controller.record("empty_response")
            else:
                adaptive_controller.record("transport_error")
        raise RuntimeError(f"API call failed after {attempt + 1} attempts: {last_error[:1000]}")
    parsed = extract_json(content) or extract_json(reasoning)
    return parsed, (content or reasoning or "")[:8000]


class AdaptiveRpmController:
    """Adjust request rate from recent API and format health."""

    def __init__(
        self,
        rate_limiter: RateLimiter,
        run_log: Path,
        *,
        enabled: bool,
        initial_rpm: int,
        min_rpm: int,
        max_rpm: int,
        window: int = 80,
        adjust_every: int = 20,
    ):
        self.rate_limiter = rate_limiter
        self.run_log = run_log
        self.enabled = enabled
        self.min_rpm = max(1, min_rpm)
        self.max_rpm = max(self.min_rpm, max_rpm)
        self.current_rpm = max(self.min_rpm, min(initial_rpm, self.max_rpm))
        self.window = max(20, window)
        self.adjust_every = max(5, adjust_every)
        self.outcomes = deque(maxlen=self.window)
        self.since_adjust = 0
        self.lock = threading.Lock()
        self._set_rpm(self.current_rpm)

    def _set_rpm(self, rpm: int):
        rpm = max(self.min_rpm, min(int(rpm), self.max_rpm))
        with self.rate_limiter.lock:
            self.rate_limiter.rpm = rpm
        self.current_rpm = rpm

    def record(self, outcome: str):
        if not self.enabled:
            return
        with self.lock:
            self.outcomes.append(outcome)
            self.since_adjust += 1
            if self.since_adjust < self.adjust_every or len(self.outcomes) < self.adjust_every:
                return
            self.since_adjust = 0
            counts = Counter(self.outcomes)
            total = len(self.outcomes)
            transport = counts["transport_error"] + counts["empty_response"]
            format_errors = counts["format_error"]
            bad_rate = (transport + format_errors) / total
            transport_rate = transport / total
            old_rpm = self.current_rpm
            reason = None

            if transport_rate >= 0.03 or bad_rate >= 0.12:
                new_rpm = max(self.min_rpm, int(old_rpm * 0.72))
                reason = "backoff_transport_or_format_errors"
            elif bad_rate <= 0.02 and old_rpm < self.max_rpm and total >= self.window // 2:
                new_rpm = min(self.max_rpm, max(old_rpm + 1, int(old_rpm * 1.15)))
                reason = "recover_stable_success_rate"
            else:
                return

            if new_rpm == old_rpm:
                return
            self._set_rpm(new_rpm)
            append_log(self.run_log, {
                "event": "adaptive_rpm_adjust",
                "old_rpm": old_rpm,
                "new_rpm": new_rpm,
                "reason": reason,
                "window": total,
                "success": counts["success"],
                "format_error": format_errors,
                "transport_error": counts["transport_error"],
                "empty_response": counts["empty_response"],
                "bad_rate": round(bad_rate, 4),
                "transport_rate": round(transport_rate, 4),
            })

    def snapshot(self) -> dict:
        counts = Counter(self.outcomes)
        total = len(self.outcomes)
        return {
            "enabled": self.enabled,
            "current_rpm": self.current_rpm,
            "min_rpm": self.min_rpm,
            "max_rpm": self.max_rpm,
            "window_observations": total,
            "window_success": counts["success"],
            "window_format_error": counts["format_error"],
            "window_transport_error": counts["transport_error"],
            "window_empty_response": counts["empty_response"],
        }


def _rank_by_votes(rankings: list[list[str]], allowed: list[str], limit: int) -> list[str]:
    allowed_set = set(allowed)
    first_votes = Counter(ranked[0] for ranked in rankings if ranked and ranked[0] in allowed_set)
    all_votes = Counter(
        value
        for ranked in rankings
        for value in ranked[:limit]
        if value in allowed_set
    )
    ordered = []
    for value, _ in first_votes.most_common():
        if value not in ordered:
            ordered.append(value)
    for value, _ in all_votes.most_common():
        if value not in ordered:
            ordered.append(value)
    return ordered[:limit]


def _retry_delay(attempt: int) -> float:
    return min(90, (2 ** attempt) * 5) + random.uniform(0, 2)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_log(path: Path, event: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), **event}, ensure_ascii=False) + "\n")


def _ranked_three_classes(fine_ranked: list[str]) -> list[str]:
    ranked = []
    for fine in fine_ranked:
        category = FINE_TO_THREE_CLASS.get(fine)
        if category and category not in ranked:
            ranked.append(category)
    return ranked


def three_class_reference(item: dict) -> str | None:
    return item.get("three_class_category") or FINE_TO_THREE_CLASS.get(item.get("violated_constraint"))


def analyze_three_class(per_item: list[dict]) -> dict:
    fatal_rows = [row for row in per_item if row["label"] == "fatal_violation"]
    y_true = [three_class_reference(row) for row in fatal_rows]
    y_pred = []
    ranked = []
    for row in fatal_rows:
        stage2 = row.get("stage2") or {}
        three_ranked = stage2.get("three_class_ranked") or []
        if not three_ranked:
            three_ranked = _ranked_three_classes(stage2.get("fine_ranked") or [])
        y_pred.append(stage2.get("three_class_prediction") or (three_ranked[0] if three_ranked else None))
        ranked.append(three_ranked)

    metrics = classification_metrics(y_true, y_pred, THREE_CLASS_CATEGORIES)
    metrics["top2_accuracy"] = top_k_accuracy(y_true, ranked, THREE_CLASS_CATEGORIES, 2)
    by_category = {}
    for category in THREE_CLASS_CATEGORIES:
        rows = [(true, pred) for true, pred in zip(y_true, y_pred) if true == category]
        total = len(rows)
        correct = sum(1 for true, pred in rows if true == pred)
        by_category[category] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }

    return {
        "three_class_taxonomy": {
            category: [fine for fine, cat in FINE_TO_THREE_CLASS.items() if cat == category]
            for category in THREE_CLASS_CATEGORIES
        },
        "metrics": metrics,
        "by_three_class_category": by_category,
    }


def run_two_stage(
    data_file: Path,
    model_key: str,
    model_name_override: str | None,
    strategy: str,
    stage1_only: bool,
    self_cons_samples: int,
    concurrency: int,
    rpm: int,
    retries: int,
    max_items: int | None,
    output_dir: Path | None,
    allow_missing: bool = False,
    adaptive_rpm: bool = False,
    adaptive_min_rpm: int | None = None,
    adaptive_max_rpm: int | None = None,
) -> dict:
    with data_file.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if max_items is not None:
        items = items[:max_items]

    model_cfg = get_model_config(model_key, model_name_override=model_name_override)
    model_name = model_name_override or model_cfg["model"]
    experiment_id = output_dir.name if output_dir else f"two_stage_5k_{_slug(model_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = output_dir or EXPERIMENTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_log = exp_dir / "run_log.jsonl"
    checkpoint_file = exp_dir / "checkpoint.json"

    if checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    else:
        checkpoint = {}

    rate_limiter = RateLimiter(rpm=rpm)
    adaptive_controller = AdaptiveRpmController(
        rate_limiter,
        run_log,
        enabled=adaptive_rpm,
        initial_rpm=rpm,
        min_rpm=adaptive_min_rpm if adaptive_min_rpm is not None else max(1, rpm // 3),
        max_rpm=adaptive_max_rpm if adaptive_max_rpm is not None else rpm,
    )
    start = time.time()
    append_log(run_log, {
        "event": "start",
        "experiment_id": experiment_id,
        "data_file": str(data_file),
        "model": model_name,
        "strategy": strategy,
        "stage1_only": stage1_only,
        "self_cons_samples": self_cons_samples,
        "items": len(items),
        "concurrency": concurrency,
        "rpm": rpm,
        "adaptive_rpm": adaptive_controller.snapshot(),
    })

    def eval_stage1(item: dict) -> tuple[str, dict]:
        item_id = item["id"]
        cached = checkpoint.get(item_id, {})
        stage1 = cached.get("stage1") or {}
        cached_strategy = stage1.get("strategy", "zero-shot")
        if stage1.get("prediction") in LABELS and cached_strategy == strategy:
            return item_id, cached
        prompt = strategy_prompt(1, strategy)
        user_prompt = user_prompt_for(item, strategy)
        samples = []
        if strategy == "self-cons":
            for _ in range(self_cons_samples):
                parsed = None
                raw = ""
                pred = None
                for parse_attempt in range(max(1, retries)):
                    parsed, raw = call_json(
                        prompt,
                        user_prompt,
                        model_key,
                        model_name_override,
                        rate_limiter,
                        retries,
                        temperature=0.7,
                        adaptive_controller=adaptive_controller,
                    )
                    pred = parse_stage1(parsed, raw)
                    if pred in LABELS:
                        adaptive_controller.record("success")
                        break
                    adaptive_controller.record("format_error")
                    if parse_attempt < max(1, retries) - 1:
                        time.sleep(_retry_delay(parse_attempt))
                samples.append({
                    "raw_response": parsed,
                    "raw_text": raw[:2000],
                    "prediction": pred,
                })
            pred = _rank_by_votes([[sample["prediction"]] for sample in samples], LABELS, 1)
            pred = pred[0] if pred else None
            parsed = {"self_consistency_samples": samples}
            raw = "\n---SAMPLE---\n".join(sample["raw_text"] for sample in samples)[:8000]
        else:
            parsed = None
            raw = ""
            pred = None
            for parse_attempt in range(max(1, retries)):
                parsed, raw = call_json(
                    prompt,
                    user_prompt,
                    model_key,
                    model_name_override,
                    rate_limiter,
                    retries,
                    adaptive_controller=adaptive_controller,
                )
                pred = parse_stage1(parsed, raw)
                if pred in LABELS:
                    adaptive_controller.record("success")
                    break
                adaptive_controller.record("format_error")
                if parse_attempt < max(1, retries) - 1:
                    time.sleep(_retry_delay(parse_attempt))
        cached["stage1"] = {
            "strategy": strategy,
            "raw_response": parsed,
            "raw_text": raw[:8000],
            "prediction": pred,
            "reference": item["label"],
            "correct": pred == item["label"],
        }
        return item_id, cached

    pending_stage1 = [
        item for item in items
        if (checkpoint.get(item["id"], {}).get("stage1") or {}).get("prediction") not in LABELS
        or (checkpoint.get(item["id"], {}).get("stage1") or {}).get("strategy", "zero-shot") != strategy
    ]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(eval_stage1, item): item for item in pending_stage1}
        done = len(items) - len(pending_stage1)
        for future in as_completed(futures):
            item = futures[future]
            item_id = item["id"]
            try:
                item_id, cached = future.result()
            except RuntimeError as exc:
                cached = checkpoint.get(item_id, {})
                cached["stage1"] = {
                    "strategy": strategy,
                    "raw_response": None,
                    "raw_text": "",
                    "prediction": None,
                    "reference": item["label"],
                    "correct": False,
                    "error": str(exc)[:2000],
                }
                append_log(run_log, {
                    "event": "stage1_error",
                    "done": done,
                    "total": len(items),
                    "id": item_id,
                    "error": str(exc)[:1000],
                })
            checkpoint[item_id] = cached
            done += 1
            if done % 20 == 0:
                save_json(checkpoint_file, checkpoint)
            append_log(run_log, {"event": "stage1_item", "done": done, "total": len(items), "id": item_id})
    save_json(checkpoint_file, checkpoint)

    missing_stage1 = [
        item["id"] for item in items
        if ((checkpoint.get(item["id"], {}).get("stage1") or {}).get("prediction") not in LABELS)
    ]
    if missing_stage1 and not allow_missing:
        append_log(run_log, {
            "event": "aborted_missing_stage1",
            "missing": len(missing_stage1),
            "sample_ids": missing_stage1[:20],
        })
        raise RuntimeError(
            f"Stage 1 has {len(missing_stage1)} missing/unparseable predictions; "
            "checkpoint retained. Re-run later or pass --allow-missing to finalize anyway."
        )

    fatal_items = [item for item in items if item["label"] == "fatal_violation"]

    def eval_stage2(item: dict) -> tuple[str, dict]:
        item_id = item["id"]
        cached = checkpoint.get(item_id, {})
        stage2 = cached.get("stage2") or {}
        cached_strategy = stage2.get("strategy", "zero-shot")
        if stage2.get("three_class_prediction") in THREE_CLASS_CATEGORIES and cached_strategy == strategy:
            return item_id, cached
        prompt = strategy_prompt(2, strategy)
        user_prompt = user_prompt_for(item, strategy)
        samples = []
        if strategy == "self-cons":
            fine_samples = []
            three_samples = []
            for _ in range(self_cons_samples):
                parsed = None
                raw = ""
                fine_one = []
                three_one = []
                for parse_attempt in range(max(1, retries)):
                    parsed, raw = call_json(
                        prompt,
                        user_prompt,
                        model_key,
                        model_name_override,
                        rate_limiter,
                        retries,
                        temperature=0.7,
                        adaptive_controller=adaptive_controller,
                    )
                    fine_one, three_one = parse_stage2(parsed, raw)
                    if three_one:
                        adaptive_controller.record("success")
                        break
                    adaptive_controller.record("format_error")
                    if parse_attempt < max(1, retries) - 1:
                        time.sleep(_retry_delay(parse_attempt))
                fine_samples.append(fine_one)
                three_samples.append(three_one)
                samples.append({
                    "raw_response": parsed,
                    "raw_text": raw[:2000],
                    "fine_ranked": fine_one,
                    "three_class_ranked": three_one,
                })
            fine_ranked = _rank_by_votes(fine_samples, FINE_CONSTRAINTS, 3)
            three_class_ranked = _rank_by_votes(three_samples, THREE_CLASS_CATEGORIES, 2)
            parsed = {"self_consistency_samples": samples}
            raw = "\n---SAMPLE---\n".join(sample["raw_text"] for sample in samples)[:8000]
        else:
            parsed = None
            raw = ""
            fine_ranked = []
            three_class_ranked = []
            for parse_attempt in range(max(1, retries)):
                parsed, raw = call_json(
                    prompt,
                    user_prompt,
                    model_key,
                    model_name_override,
                    rate_limiter,
                    retries,
                    adaptive_controller=adaptive_controller,
                )
                fine_ranked, three_class_ranked = parse_stage2(parsed, raw)
                if three_class_ranked:
                    adaptive_controller.record("success")
                    break
                adaptive_controller.record("format_error")
                if parse_attempt < max(1, retries) - 1:
                    time.sleep(_retry_delay(parse_attempt))
        fine_ref = item["violated_constraint"]
        three_ref = three_class_reference(item)
        cached["stage2"] = {
            "strategy": strategy,
            "raw_response": parsed,
            "raw_text": raw[:8000],
            "three_class_ranked": three_class_ranked,
            "three_class_prediction": three_class_ranked[0] if three_class_ranked else None,
            "three_class_reference": three_ref,
            "three_class_correct": bool(three_class_ranked and three_class_ranked[0] == three_ref),
            "three_class_top2_correct": three_ref in three_class_ranked[:2],
            "fine_ranked": fine_ranked,
            "fine_prediction": fine_ranked[0] if fine_ranked else None,
            "fine_reference": fine_ref,
            "fine_correct": bool(fine_ranked and fine_ranked[0] == fine_ref),
            "fine_top3_correct": fine_ref in fine_ranked[:3],
        }
        return item_id, cached

    pending_stage2 = [
        item for item in fatal_items
        if (checkpoint.get(item["id"], {}).get("stage2") or {}).get("three_class_prediction") not in THREE_CLASS_CATEGORIES
        or (checkpoint.get(item["id"], {}).get("stage2") or {}).get("strategy", "zero-shot") != strategy
    ]
    if not stage1_only:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(eval_stage2, item): item for item in pending_stage2}
            done = len(fatal_items) - len(pending_stage2)
            for future in as_completed(futures):
                item = futures[future]
                item_id = item["id"]
                try:
                    item_id, cached = future.result()
                except RuntimeError as exc:
                    cached = checkpoint.get(item_id, {})
                    cached["stage2"] = {
                        "strategy": strategy,
                        "raw_response": None,
                        "raw_text": "",
                        "three_class_ranked": [],
                        "three_class_prediction": None,
                        "three_class_reference": three_class_reference(item),
                        "three_class_correct": False,
                        "three_class_top2_correct": False,
                        "fine_ranked": [],
                        "fine_prediction": None,
                        "fine_reference": item["violated_constraint"],
                        "fine_correct": False,
                        "fine_top3_correct": False,
                        "error": str(exc)[:2000],
                    }
                    append_log(run_log, {
                        "event": "stage2_error",
                        "done": done,
                        "total": len(fatal_items),
                        "id": item_id,
                        "error": str(exc)[:1000],
                    })
                checkpoint[item_id] = cached
                done += 1
                if done % 20 == 0:
                    save_json(checkpoint_file, checkpoint)
                append_log(run_log, {"event": "stage2_item", "done": done, "total": len(fatal_items), "id": item_id})
        save_json(checkpoint_file, checkpoint)

        missing_stage2 = [
            item["id"] for item in fatal_items
            if (
                (checkpoint.get(item["id"], {}).get("stage2") or {}).get("three_class_prediction")
                not in THREE_CLASS_CATEGORIES
            )
        ]
        if missing_stage2 and not allow_missing:
            append_log(run_log, {
                "event": "aborted_missing_stage2",
                "missing": len(missing_stage2),
                "sample_ids": missing_stage2[:20],
            })
            raise RuntimeError(
                f"Stage 2 has {len(missing_stage2)} missing/unparseable predictions; "
                "checkpoint retained. Re-run later or pass --allow-missing to finalize anyway."
            )

    elapsed = round(time.time() - start)
    per_item = []
    for item in items:
        cached = checkpoint.get(item["id"], {})
        stage1 = cached.get("stage1")
        if stage1:
            stage1["reference"] = item["label"]
            stage1["correct"] = stage1.get("prediction") == item["label"]
        stage2 = cached.get("stage2") if item["label"] == "fatal_violation" and not stage1_only else None
        if stage2:
            fine_ranked = stage2.get("fine_ranked") or []
            three_class_ranked = stage2.get("three_class_ranked") or []
            three_ref = three_class_reference(item)
            if not three_class_ranked and fine_ranked:
                three_class_ranked = _ranked_three_classes(fine_ranked)
                stage2["three_class_ranked"] = three_class_ranked
                stage2["three_class_prediction"] = three_class_ranked[0] if three_class_ranked else None
            stage2["three_class_reference"] = three_ref
            stage2["three_class_correct"] = bool(three_class_ranked and three_class_ranked[0] == three_ref)
            stage2["three_class_top2_correct"] = three_ref in three_class_ranked[:2]
            stage2["fine_reference"] = item["violated_constraint"]
            stage2["fine_correct"] = bool(fine_ranked and fine_ranked[0] == item["violated_constraint"])
            stage2["fine_top3_correct"] = item["violated_constraint"] in fine_ranked[:3]
        per_item.append({
            "id": item["id"],
            "domain": item.get("domain"),
            "label": item.get("label"),
            "three_class_category": three_class_reference(item),
            "violated_constraint": item.get("violated_constraint"),
            "difficulty": item.get("difficulty"),
            "stage1": stage1,
            "stage2": stage2,
        })

    stage1_true = [item["label"] for item in items]
    stage1_pred = [(checkpoint.get(item["id"], {}).get("stage1") or {}).get("prediction") for item in items]
    stage1_metrics = classification_metrics(stage1_true, stage1_pred, LABELS)
    if stage1_only:
        stage2_fine = None
        stage2_three = None
        three_class_analysis = None
    else:
        fatal_true_fine = [item["violated_constraint"] for item in fatal_items]
        fatal_pred_fine = [
            ((checkpoint.get(item["id"], {}).get("stage2") or {}).get("fine_prediction"))
            for item in fatal_items
        ]
        fatal_ranked_fine = [
            ((checkpoint.get(item["id"], {}).get("stage2") or {}).get("fine_ranked") or [])
            for item in fatal_items
        ]
        fatal_true_three = [three_class_reference(item) for item in fatal_items]
        fatal_pred_three = [
            ((checkpoint.get(item["id"], {}).get("stage2") or {}).get("three_class_prediction"))
            for item in fatal_items
        ]
        fatal_ranked_three = [
            ((checkpoint.get(item["id"], {}).get("stage2") or {}).get("three_class_ranked") or [])
            for item in fatal_items
        ]
        stage2_fine = classification_metrics(fatal_true_fine, fatal_pred_fine, FINE_CONSTRAINTS)
        stage2_fine["top3_accuracy"] = top_k_accuracy(fatal_true_fine, fatal_ranked_fine, FINE_CONSTRAINTS, 3)
        stage2_three = classification_metrics(fatal_true_three, fatal_pred_three, THREE_CLASS_CATEGORIES)
        stage2_three["top2_accuracy"] = top_k_accuracy(fatal_true_three, fatal_ranked_three, THREE_CLASS_CATEGORIES, 2)
        three_class_analysis = analyze_three_class(per_item)
        three_class_analysis["metrics"] = stage2_three

    summary = {
        "experiment_id": experiment_id,
        "completed_at": datetime.now().isoformat(),
        "data_file": str(data_file),
        "model": model_name,
        "model_key": model_key,
        "strategy": strategy,
        "stage1_only": stage1_only,
        "self_cons_samples": self_cons_samples if strategy == "self-cons" else 1,
        "items": len(items),
        "fatal_items_for_stage2": len(fatal_items),
        "concurrency": concurrency,
        "rpm": rpm,
        "adaptive_rpm": adaptive_controller.snapshot(),
        "retries": retries,
        "elapsed_seconds": elapsed,
        "stage1_problem_detection": stage1_metrics,
        "stage2_three_class_categories": stage2_three,
        "stage2_fine_constraints": stage2_fine,
    }
    save_json(exp_dir / "summary.json", summary)
    matrices = {
        "stage1": stage1_metrics["confusion_matrix"],
    }
    if not stage1_only:
        matrices["stage2_three_class"] = stage2_three["confusion_matrix"]
        matrices["stage2_fine"] = stage2_fine["confusion_matrix"]
    save_json(exp_dir / "confusion_matrices.json", matrices)
    if three_class_analysis is not None:
        save_json(exp_dir / "three_class_analysis.json", three_class_analysis)
    with (exp_dir / "per_item_results.jsonl").open("w", encoding="utf-8") as f:
        for row in per_item:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    readme_lines = [
        f"# Two-stage Evaluation: {experiment_id}",
        "",
        f"- Completed at: {summary['completed_at']}",
        f"- Dataset: {data_file}",
        f"- Model: {model_name}",
        f"- Strategy: {strategy}",
        f"- Stage 1 only: {stage1_only}",
        f"- Self-consistency samples: {summary['self_cons_samples']}",
        f"- Items: {len(items)}",
        f"- Fatal items for Stage 2: {0 if stage1_only else len(fatal_items)}",
        f"- Stage 1 accuracy: {stage1_metrics['accuracy']:.4f}",
        f"- Stage 1 balanced accuracy: {stage1_metrics['balanced_accuracy']:.4f}",
        f"- Stage 1 macro-F1: {stage1_metrics['macro_f1']:.4f}",
        f"- Stage 1 missing/unparseable predictions: {stage1_metrics['missing_predictions']}",
    ]
    if not stage1_only:
        readme_lines.extend([
            f"- Stage 2 three-class accuracy: {three_class_analysis['metrics']['accuracy']:.4f}",
            f"- Stage 2 three-class macro-F1: {three_class_analysis['metrics']['macro_f1']:.4f}",
            f"- Stage 2 three-class top-2: {three_class_analysis['metrics']['top2_accuracy']['accuracy']:.4f}",
            f"- Stage 2 fine accuracy: {stage2_fine['accuracy']:.4f}",
            f"- Stage 2 fine macro-F1: {stage2_fine['macro_f1']:.4f}",
            f"- Stage 2 fine top-3: {stage2_fine['top3_accuracy']['accuracy']:.4f}",
            "",
            "## Three-class Category Accuracy",
            "",
            *[
                f"- {category}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})"
                for category, stats in three_class_analysis["by_three_class_category"].items()
            ],
        ])
    readme_lines.append("")
    (exp_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    append_log(run_log, {"event": "completed", "elapsed_seconds": elapsed})
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run two-stage evaluation")
    parser.add_argument("--data", default=str(DATA_FILE))
    parser.add_argument("--model", default="mimo")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--strategy", choices=STRATEGIES, default="zero-shot")
    parser.add_argument("--stage1-only", action="store_true", help="Run only Stage 1 problem detection")
    parser.add_argument("--self-cons-samples", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--rpm", type=int, default=100)
    parser.add_argument(
        "--adaptive-rpm",
        action="store_true",
        help="Dynamically back off/recover RPM from recent transport and format errors.",
    )
    parser.add_argument(
        "--adaptive-min-rpm",
        type=int,
        default=None,
        help="Minimum RPM when --adaptive-rpm backs off (default: rpm // 3).",
    )
    parser.add_argument(
        "--adaptive-max-rpm",
        type=int,
        default=None,
        help="Maximum RPM when --adaptive-rpm recovers (default: --rpm).",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Finalize even if some predictions are missing or unparseable.",
    )
    args = parser.parse_args()
    summary = run_two_stage(
        data_file=Path(args.data),
        model_key=args.model,
        model_name_override=args.model_name,
        strategy=args.strategy,
        stage1_only=args.stage1_only,
        self_cons_samples=max(1, args.self_cons_samples),
        concurrency=args.concurrency,
        rpm=args.rpm,
        retries=args.retries,
        max_items=args.max_items,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        allow_missing=args.allow_missing,
        adaptive_rpm=args.adaptive_rpm,
        adaptive_min_rpm=args.adaptive_min_rpm,
        adaptive_max_rpm=args.adaptive_max_rpm,
    )
    print(json.dumps({
        "experiment_id": summary["experiment_id"],
        "strategy": summary["strategy"],
        "stage1_only": summary["stage1_only"],
        "stage1_accuracy": summary["stage1_problem_detection"]["accuracy"],
        "stage2_three_class_accuracy": (
            summary["stage2_three_class_categories"]["accuracy"]
            if summary["stage2_three_class_categories"] else None
        ),
        "stage2_fine_accuracy": (
            summary["stage2_fine_constraints"]["accuracy"]
            if summary["stage2_fine_constraints"] else None
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
