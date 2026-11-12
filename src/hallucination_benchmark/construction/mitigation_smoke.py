#!/usr/bin/env python3
"""Mitigation smoke test on the clean 5k benchmark.

Runs common prompting/mitigation strategies on a deterministic stratified
subset and writes a compact, paper-friendly experiment record.
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from hallucination_benchmark.config import CURRENT_DATA_DIR, EXPERIMENTS_DIR, get_model_config
from hallucination_benchmark.evaluator import RateLimiter
from hallucination_benchmark.llm_client import extract_json, llm_call
from hallucination_benchmark.rule_checkers import run_checks


BASELINE_PROMPT = """You are an IEEE journal reviewer with deep expertise in hardware engineering and signal processing.
Review the experimental protocol below and determine whether it is physically feasible.
Focus on:
1. Does the sampling rate satisfy the Nyquist theorem?
2. Does the sensor bandwidth cover the target signal frequency?
3. Are the hardware parameters self-consistent (accuracy, resolution, power, etc.)?
4. Is the signal processing pipeline reasonable?

Output JSON only:
{"is_physically_viable": true/false, "identified_error": "error description or 'none'", "reasoning": "brief reasoning"}"""


COT_PROMPT = """You are an IEEE journal reviewer with deep expertise in hardware engineering and signal processing.
Review the experimental protocol below and determine whether it is physically feasible.

Analyze these checks before concluding:
1. Target signal and frequency range.
2. Sampling rate vs. Nyquist theorem.
3. Sensor bandwidth vs. target signal frequency.
4. ADC resolution, SNR, power, latency, and data bandwidth consistency.
5. Sensor placement and physical signal path.
6. Whether the claimed validation or accuracy is physically plausible.

Output JSON only:
{"is_physically_viable": true/false, "identified_error": "error description or 'none'", "reasoning": "concise step-by-step reasoning"}"""


TOOL_PROMPT = """You are an IEEE journal reviewer with deep expertise in hardware engineering and signal processing.
Review the experimental protocol below and determine whether it is physically feasible.

Automated rule-check results are provided as supplementary evidence. Use them when relevant, but do not blindly follow them.

Rule-check results:
{rule_results}

Output JSON only:
{{"is_physically_viable": true/false, "identified_error": "error description or 'none'", "reasoning": "brief reasoning", "rule_check_used": true/false}}"""


TECHNIQUES = ("baseline", "cot", "self_consistency", "tool_augmented")


def load_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stratified_sample(items: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Sample by label and fatal constraint, preserving the clean 5k distribution."""
    if sample_size >= len(items):
        return list(items)

    rng = random.Random(seed)
    label_counts = Counter(item["label"] for item in items)
    raw_targets = {
        label: sample_size * count / len(items)
        for label, count in label_counts.items()
    }
    targets = {label: int(raw_targets[label]) for label in label_counts}
    remainder = sample_size - sum(targets.values())
    for label, _frac in sorted(
        ((label, raw_targets[label] - targets[label]) for label in label_counts),
        key=lambda x: x[1],
        reverse=True,
    )[:remainder]:
        targets[label] += 1

    selected: list[dict] = []
    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_label[item["label"]].append(item)

    for label, target in targets.items():
        bucket = by_label[label]
        if label != "fatal_violation":
            chosen = rng.sample(bucket, target)
            selected.extend(chosen)
            continue

        by_constraint: dict[str, list[dict]] = defaultdict(list)
        for item in bucket:
            by_constraint[item.get("violated_constraint", "unknown")].append(item)

        constraints = sorted(by_constraint)
        base = target // len(constraints)
        extra = target % len(constraints)
        fatal_chosen: list[dict] = []
        for i, constraint in enumerate(constraints):
            n = base + (1 if i < extra else 0)
            fatal_chosen.extend(rng.sample(by_constraint[constraint], n))
        selected.extend(fatal_chosen)

    rng.shuffle(selected)
    return selected


def expected_viability(label: str) -> bool:
    return label == "valid"


def judge(item: dict, response: dict | None) -> bool:
    if not response:
        return False
    return response.get("is_physically_viable") is expected_viability(item["label"])


def parse_response(content: str | None, reasoning: str | None) -> dict | None:
    return extract_json(content or "") or extract_json(reasoning or "")


def call_once(
    system_prompt: str,
    item: dict,
    *,
    model_key: str,
    model_name: str | None,
    temperature: float,
    rate_limiter: RateLimiter,
    max_retries: int,
) -> tuple[dict | None, str]:
    last_error = ""
    for attempt in range(max_retries):
        rate_limiter.acquire()
        content, reasoning = llm_call(
            system_prompt,
            item["protocol"],
            model_key=model_key,
            model_name_override=model_name,
            temperature=temperature,
            max_tokens=4000,
            timeout=120,
        )
        result = parse_response(content, reasoning)
        if result is not None:
            return result, ""
        last_error = (content or reasoning or "empty_or_unparseable_response")[:500]
        if attempt < max_retries - 1:
            time.sleep((attempt + 1) * 2)
    return None, last_error


def evaluate_item(
    item: dict,
    technique: str,
    *,
    model_key: str,
    model_name: str | None,
    rate_limiter: RateLimiter,
    max_retries: int,
    sc_runs: int,
) -> dict:
    if technique == "baseline":
        response, error = call_once(
            BASELINE_PROMPT,
            item,
            model_key=model_key,
            model_name=model_name,
            temperature=0.1,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
        )
        votes = None
    elif technique == "cot":
        response, error = call_once(
            COT_PROMPT,
            item,
            model_key=model_key,
            model_name=model_name,
            temperature=0.1,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
        )
        votes = None
    elif technique == "tool_augmented":
        constraint = item.get("violated_constraint") or "sampling_theorem"
        rule_results = run_checks(item["protocol"], item.get("domain", ""), constraint)
        prompt = TOOL_PROMPT.format(
            rule_results=json.dumps(rule_results, ensure_ascii=False, indent=2)
        )
        response, error = call_once(
            prompt,
            item,
            model_key=model_key,
            model_name=model_name,
            temperature=0.1,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
        )
        votes = None
    elif technique == "self_consistency":
        vote_values = []
        raw_votes = []
        errors = []
        for _ in range(sc_runs):
            response_i, error_i = call_once(
                BASELINE_PROMPT,
                item,
                model_key=model_key,
                model_name=model_name,
                temperature=0.7,
                rate_limiter=rate_limiter,
                max_retries=max_retries,
            )
            raw_votes.append(response_i)
            if response_i is None:
                errors.append(error_i)
                continue
            vote_values.append(response_i.get("is_physically_viable"))
        if vote_values:
            counts = Counter(vote_values)
            majority = counts.most_common(1)[0][0]
            response = {
                "is_physically_viable": majority,
                "identified_error": "none" if majority else "see self-consistency votes",
                "reasoning": f"majority_vote={dict(counts)}",
            }
            error = ""
        else:
            response = None
            error = "; ".join(errors)[:500] or "no_valid_votes"
        votes = raw_votes
    else:
        raise ValueError(f"Unknown technique: {technique}")

    return {
        "id": item.get("id"),
        "technique": technique,
        "label": item.get("label"),
        "domain": item.get("domain"),
        "violated_constraint": item.get("violated_constraint"),
        "expected_is_physically_viable": expected_viability(item["label"]),
        "model_response": response,
        "votes": votes,
        "error": error,
        "correct": judge(item, response),
    }


def summarize(results: list[dict], elapsed_seconds: float) -> dict:
    by_technique: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_technique[row["technique"]].append(row)

    stats = {}
    for technique, rows in by_technique.items():
        errors = sum(1 for row in rows if row["model_response"] is None)
        valid = len(rows) - errors
        correct = sum(1 for row in rows if row["correct"])
        label_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
        constraint_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
        for row in rows:
            for key, container in (
                (row["label"], label_stats),
                (row.get("violated_constraint") or "N/A", constraint_stats),
            ):
                container[key]["total"] += 1
                if row["correct"]:
                    container[key]["correct"] += 1
                if row["model_response"] is None:
                    container[key]["errors"] += 1
        stats[technique] = {
            "total": len(rows),
            "correct": correct,
            "errors": errors,
            "accuracy": correct / valid if valid else 0,
            "label_stats": dict(sorted(label_stats.items())),
            "constraint_stats": dict(sorted(constraint_stats.items())),
        }

    return {
        "elapsed_seconds": round(elapsed_seconds),
        "stats": stats,
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed run log line {line_no}: {path}")
                continue
            if row.get("model_response") is None:
                # Treat API/parse failures as incomplete so reruns retry them.
                continue
            rows.append(row)
    return rows


def make_readme(exp_dir: Path, config: dict, summary: dict) -> str:
    lines = [
        f"# 5k Mitigation Smoke Test: {exp_dir.name}",
        "",
        f"- Completed at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {config['dataset']}",
        f"- Subset: {config['sample_size']} stratified items, seed={config['seed']}",
        f"- Model: {config['model_name']}",
        f"- Techniques: {', '.join(config['techniques'])}",
        f"- Self-consistency runs: {config['sc_runs']}",
        f"- Concurrency: {config['concurrency']}",
        f"- RPM: {config['rpm']}",
        "",
        "## Results",
        "",
        "| Technique | Correct | Errors | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for technique in config["techniques"]:
        stat = summary["stats"].get(technique, {})
        total = stat.get("total", 0)
        correct = stat.get("correct", 0)
        errors = stat.get("errors", 0)
        valid = total - errors
        acc = stat.get("accuracy", 0) * 100
        lines.append(f"| {technique} | {correct}/{valid} | {errors} | {acc:.1f}% |")

    lines.extend(["", "## Notes", ""])
    lines.append(
        "- This is a mitigation smoke test, not the final full-5k run. "
        "It uses binary physical-viability correctness to match the previous L2 baseline."
    )
    lines.append(
        "- If a technique improves materially on this subset, rerun the same strategy on the full 5k."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mitigation smoke test on clean 5k.")
    parser.add_argument("--data", type=Path, default=CURRENT_DATA_DIR / "benchmark_balanced.json")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--model", default="mimo", help="Model key from config.py")
    parser.add_argument("--model-name", default=None, help="Optional model name override")
    parser.add_argument("--techniques", nargs="+", choices=TECHNIQUES, default=list(TECHNIQUES))
    parser.add_argument("--sc-runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--rpm", type=int, default=100)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    model_cfg = get_model_config(args.model, model_name_override=args.model_name)
    if not model_cfg["api_key"]:
        raise SystemExit(f"Missing API key for model key '{args.model}'")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = args.output_dir or EXPERIMENTS_DIR / f"mitigation_5k_smoke_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.data)
    subset = stratified_sample(items, args.sample_size, args.seed)

    new_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.data),
        "experiment_dir": str(exp_dir),
        "sample_size": len(subset),
        "seed": args.seed,
        "model_key": args.model,
        "model_name": model_cfg["model"],
        "techniques": args.techniques,
        "sc_runs": args.sc_runs,
        "concurrency": args.concurrency,
        "rpm": args.rpm,
        "retries": args.retries,
        "subset_label_distribution": dict(Counter(item["label"] for item in subset)),
        "subset_fatal_constraint_distribution": dict(
            Counter(
                item.get("violated_constraint", "unknown")
                for item in subset
                if item["label"] == "fatal_violation"
            )
        ),
        "prompts": {
            "baseline": BASELINE_PROMPT,
            "cot": COT_PROMPT,
            "tool_augmented": TOOL_PROMPT,
        },
    }
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        config.setdefault("resume_runs", []).append(
            {
                "resumed_at": new_config["created_at"],
                "concurrency": args.concurrency,
                "rpm": args.rpm,
                "retries": args.retries,
            }
        )
    else:
        config = new_config
    write_json(exp_dir / "config.json", config)
    write_json(exp_dir / "subset_500.json", subset)

    rate_limiter = RateLimiter(rpm=args.rpm)
    all_jobs = [(item, technique) for technique in args.techniques for item in subset]
    run_log = exp_dir / "run_log.jsonl"
    results: list[dict] = load_existing_results(run_log)
    completed_keys = {
        (row.get("technique"), row.get("id"))
        for row in results
        if row.get("technique") and row.get("id")
    }
    pending_jobs = [
        (item, technique)
        for item, technique in all_jobs
        if (technique, item.get("id")) not in completed_keys
    ]
    start = time.time()
    done = len(results)

    print(f"Experiment: {exp_dir}")
    print(f"Items: {len(subset)}; jobs: {len(all_jobs)}; pending: {len(pending_jobs)}; model: {model_cfg['model']}")
    print(f"Techniques: {', '.join(args.techniques)}")

    def _run(job: tuple[dict, str]) -> dict:
        item, technique = job
        return evaluate_item(
            item,
            technique,
            model_key=args.model,
            model_name=args.model_name,
            rate_limiter=rate_limiter,
            max_retries=args.retries,
            sc_runs=args.sc_runs,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_run, job) for job in pending_jobs]
        with run_log.open("a", encoding="utf-8") as log_f:
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                done += 1
                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_f.flush()
                status = "ok" if row["correct"] else ("api_error" if row["model_response"] is None else "wrong")
                print(
                    f"[{done}/{len(all_jobs)}] {row['technique']} | {row['label']} | {status}",
                    flush=True,
                )

    ordered_results = sorted(results, key=lambda r: (r["technique"], r["id"] or ""))
    summary = summarize(ordered_results, time.time() - start)
    summary["config"] = {k: v for k, v in config.items() if k != "prompts"}

    write_jsonl(exp_dir / "per_item_results.jsonl", ordered_results)
    write_json(exp_dir / "summary.json", summary)
    (exp_dir / "README.md").write_text(make_readme(exp_dir, config, summary), encoding="utf-8")
    print(f"Saved summary to {exp_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
