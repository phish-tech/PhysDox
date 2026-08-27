#!/usr/bin/env python3
"""
增量生成脚本 - 在现有benchmark基础上补充生成新样本
支持去重合并，避免与旧数据重复
"""

import json
import os
import time
from datetime import datetime

from hallucination_benchmark.config import OUTPUT_DIR, SEED_DATA_DIR, get_model_config
from hallucination_benchmark.generator import (
    load_seed_data,
    _generate_one_protocol,
    _inject_one_sabotage,
    _verify_one_item,
    _generate_one_negative,
    assign_difficulty,
)
from hallucination_benchmark.schema import make_item, make_clean_for_output
from hallucination_benchmark.rule_checkers import run_checks

from concurrent.futures import ThreadPoolExecutor, as_completed


def load_existing_benchmark(path: str) -> list[dict]:
    """加载现有的benchmark数据"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_existing_protocol_keys(items: list[dict]) -> set[str]:
    """提取现有数据的protocol前200字符用于去重"""
    return {item["protocol"][:200] for item in items if item.get("protocol")}


def merge_and_dedup(existing: list[dict], new_items: list[dict]) -> list[dict]:
    """合并新旧数据并去重"""
    existing_keys = get_existing_protocol_keys(existing)
    
    merged = list(existing)  # 保留旧数据
    added = 0
    
    for item in new_items:
        key = item["protocol"][:200]
        if key not in existing_keys:
            merged.append(item)
            existing_keys.add(key)
            added += 1
    
    print(f"  Existing: {len(existing)}")
    print(f"  New items: {len(new_items)}")
    print(f"  Added (after dedup): {added}")
    print(f"  Total merged: {len(merged)}")
    
    return merged


def run_incremental(
    seed_dir: str = SEED_DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    additional_rounds: int = 10,
    protocols_per_domain: int = 8,
    sabotages_per_protocol: int = 2,
    concurrency: int = 8,
    model_key: str = None,
):
    """运行增量生成"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载现有数据
    benchmark_path = os.path.join(output_dir, "benchmark.json")
    existing_items = load_existing_benchmark(benchmark_path)
    print(f"Loaded {len(existing_items)} existing items")
    
    # 加载seed数据
    constraints, domains = load_seed_data(seed_dir)
    model_cfg = get_model_config(model_key)
    
    temperatures = [0.5, 0.7, 0.9]
    new_items = []
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Incremental Generation")
    print(f"  Additional rounds: {additional_rounds}")
    print(f"  Protocols/domain/round: {protocols_per_domain}")
    print(f"  Concurrency: {concurrency}")
    print(f"{'='*70}")
    
    for round_idx in range(additional_rounds):
        temp = temperatures[round_idx % len(temperatures)]
        print(f"\nRound {round_idx + 1}/{additional_rounds} (temp={temp})")
        
        # Phase 2: Generate correct protocols
        print("  Phase 2: Generating protocols...")
        protocol_futures = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for domain in domains:
                for i in range(protocols_per_domain):
                    f = pool.submit(_generate_one_protocol, domain, model_key, temp)
                    protocol_futures[f] = (domain, i)
        
        domain_protocols = {}
        for f in protocol_futures:
            domain, idx = protocol_futures[f]
            result = f.result()
            if result:
                domain_protocols.setdefault(domain["key"], []).append(result)
        
        total_protos = sum(len(v) for v in domain_protocols.values())
        print(f"    Generated {total_protos} protocols")
        
        # Phase 3+4+5: Sabotage injection
        sabotage_tasks = []
        for domain in domains:
            dk = domain["key"]
            applicable = [c for c in constraints if dk in c.get("domains", [])]
            selected = applicable[:sabotages_per_protocol]
            for correct_proto in domain_protocols.get(dk, []):
                for constraint in selected:
                    sabotage_tasks.append((domain, correct_proto, constraint))
        
        print(f"  Phase 3+4+5: {len(sabotage_tasks)} sabotage tasks...")
        
        def _process_sabotage(task):
            domain, correct_proto, constraint = task
            dk = domain["key"]
            ck = constraint["key"]
            
            sabotage = _inject_one_sabotage(correct_proto, constraint, model_key)
            if not sabotage:
                return None
            
            rule_results = run_checks(sabotage["sabotaged_protocol"], dk, ck)
            rule_passed = all(r.get("passed", True) for r in rule_results) if rule_results else True
            
            verifier = _verify_one_item(
                sabotage["sabotaged_protocol"],
                sabotage.get("violated_principle", constraint["principle"]),
                sabotage.get("why_invalid", ""),
                model_key,
            )
            quality = verifier.get("quality_score", 0)
            is_wrong = verifier.get("is_physically_wrong", False)
            is_realistic = verifier.get("is_protocol_realistic", False)
            verifier_passed = is_wrong and is_realistic and quality >= 3
            
            return {
                "domain": domain,
                "constraint": constraint,
                "correct_proto": correct_proto,
                "sabotage": sabotage,
                "rule_passed": rule_passed,
                "verifier": verifier,
                "verifier_passed": verifier_passed,
            }
        
        item_counter = len(existing_items) + len(new_items)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_process_sabotage, t): t for t in sabotage_tasks}
            for f in as_completed(futures):
                result = f.result()
                if not result:
                    continue
                
                item_counter += 1
                domain = result["domain"]
                constraint = result["constraint"]
                sabotage = result["sabotage"]
                verifier = result["verifier"]
                
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
                    surface_plausibility="high" if verifier.get("llm_would_fool") else "medium",
                    severity=constraint.get("severity", "fatal"),
                    source_correct_protocol=result["correct_proto"],
                    what_changed=sabotage.get("what_changed", ""),
                    idx=item_counter,
                )
                new_items.append(item)
        
        # Phase 6: Negative samples
        n_valid = max(1, int(protocols_per_domain * 0.2))
        n_ambig = max(1, int(protocols_per_domain * 0.1))
        neg_tasks = []
        for domain in domains:
            for _ in range(n_valid):
                neg_tasks.append((domain, "valid"))
            for _ in range(n_ambig):
                neg_tasks.append((domain, "minor_issue"))
        
        print(f"  Phase 6: {len(neg_tasks)} negative samples...")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_generate_one_negative, d, l, model_key): (d, l) for d, l in neg_tasks}
            for f in as_completed(futures):
                ns = f.result()
                if not ns:
                    continue
                domain, label = futures[f]
                item_counter += 1
                item = make_item(
                    domain=domain["key"],
                    scenario=domain["description"],
                    label=ns["label"],
                    protocol=ns["protocol"],
                    error_category="N/A" if ns["label"] == "valid" else "minor_issue",
                    violated_constraint="none" if ns["label"] == "valid" else "suboptimal",
                    violated_principle="",
                    why_invalid="" if ns["label"] == "valid" else "Minor suboptimal issue",
                    minimal_fix="",
                    difficulty="easy" if ns["label"] == "valid" else "medium",
                    surface_plausibility="high",
                    severity="none" if ns["label"] == "valid" else "minor",
                    source_correct_protocol="",
                    what_changed="",
                    idx=item_counter + 10000,
                )
                new_items.append(item)
    
    # 合并去重
    print(f"\n{'='*70}")
    print("Merging with existing data...")
    print(f"{'='*70}")
    
    merged_items = merge_and_dedup(existing_items, new_items)
    
    # 输出
    clean_items = [make_clean_for_output(item) for item in merged_items]
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(clean_items, f, ensure_ascii=False, indent=2)
    
    # 更新manifest
    elapsed = time.time() - start_time
    label_counts = {}
    for item in merged_items:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
    
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "model": model_cfg["model"],
        "rounds": "incremental",
        "additional_rounds": additional_rounds,
        "elapsed_seconds": round(elapsed),
        "total_items": len(merged_items),
        "label_distribution": label_counts,
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(merged_items)} items to {benchmark_path}")
    return merged_items


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental benchmark generation")
    parser.add_argument("--rounds", type=int, default=10, help="Additional rounds to generate")
    parser.add_argument("--protocols-per-domain", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    
    run_incremental(
        additional_rounds=args.rounds,
        protocols_per_domain=args.protocols_per_domain,
        concurrency=args.concurrency,
    )
