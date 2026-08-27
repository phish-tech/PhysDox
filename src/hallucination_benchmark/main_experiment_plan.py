#!/usr/bin/env python3
"""Print reproducible commands for the main paper experiment matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from hallucination_benchmark.config import get_model_config

SILVER_DATA = "data/current/benchmark_balanced.json"
GOLD_DATA = "data/gold/gold_0510_final_full.json"

DEFAULT_MODELS = [
    {"family": "Reasoning / domain", "key": "mimo", "name": "mimo-v2.5-pro", "rpm": 80, "concurrency": 20},
    {"family": "Non-reasoning / domain", "key": "mimo-v2.5", "name": "mimo-v2.5", "rpm": 80, "concurrency": 20},
    {
        "family": "Reasoning / proprietary",
        "key": "claude-opus-4-7",
        "name": "claude-opus-4-7",
        "rpm": 90,
        "concurrency": 15,
    },
    {
        "family": "Reasoning / proprietary",
        "key": "gpt-5.5",
        "name": "gpt-5.5",
        "rpm": 120,
        "concurrency": 20,
    },
    {
        "family": "Fast general",
        "key": "deepseek-v4-flash",
        "name": "deepseek-v4-flash",
        "rpm": 250,
        "concurrency": 50,
    },
    {
        "family": "Reasoning / general",
        "key": "deepseek-v4-pro",
        "name": "deepseek-v4-pro",
        "rpm": 250,
        "concurrency": 50,
    },
]
DEFAULT_STRATEGIES = ["zero-shot", "cot", "self-cons", "tool-aug"]


def build_plan(models: list[dict], strategies: list[str], run_id: str) -> dict:
    rows = []
    for model in models:
        model_key = model["key"]
        model_name = model.get("name") or get_model_config(model_key)["model"]
        rpm = model.get("rpm", 100)
        concurrency = model.get("concurrency", 30)
        for strategy in strategies:
            safe_model = model_name.replace("/", "_").replace(".", "_").replace("-", "_")
            safe_strategy = strategy.replace("-", "_")
            self_cons_arg = " --self-cons-samples 3" if strategy == "self-cons" else ""
            rows.append({
                "family": model["family"],
                "model_key": model_key,
                "model_name": model_name,
                "strategy": strategy,
                "rpm": rpm,
                "concurrency": concurrency,
                "silver_two_stage_command": (
                    "uv run benchmark-two-stage-eval "
                    f"--data {SILVER_DATA} --model {model_key} --strategy {strategy} "
                    f"--concurrency {concurrency} --rpm {rpm} --retries 5{self_cons_arg} "
                    f"--output-dir experiments/main_{run_id}/silver5k_twostage_{safe_model}_{safe_strategy}"
                ),
                "gold_two_stage_command": (
                    "uv run benchmark-two-stage-eval "
                    f"--data {GOLD_DATA} --model {model_key} --strategy {strategy} "
                    f"--concurrency {concurrency} --rpm {rpm} --retries 5{self_cons_arg} "
                    f"--output-dir experiments/main_{run_id}/gold683_twostage_{safe_model}_{safe_strategy}"
                ),
            })
    return {
        "run_id": run_id,
        "datasets": {
            "silver_two_stage": SILVER_DATA,
            "gold_two_stage": GOLD_DATA,
        },
        "models": models,
        "strategies": strategies,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Print the main experiment command matrix")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of shell commands")
    args = parser.parse_args()
    plan = build_plan(DEFAULT_MODELS, DEFAULT_STRATEGIES, args.run_id)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    for row in plan["rows"]:
        print(f"# {row['family']} | {row['model_name']} | {row['strategy']} | Silver 5k Two-stage")
        print(row["silver_two_stage_command"])
        print()
        print(f"# {row['family']} | {row['model_name']} | {row['strategy']} | Gold 683 Two-stage")
        print(row["gold_two_stage_command"])
        print()


if __name__ == "__main__":
    main()
