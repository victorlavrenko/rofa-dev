"""Analyze summary.jsonl logs to compute aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from rofa.metrics import r_w_other_class, top2_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ROFA summary logs.")
    parser.add_argument("--summary", required=True, help="Path to summary.jsonl")
    parser.add_argument("--output", required=False, help="Path to output report JSON")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def analyze_greedy(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r.get("prediction") == r.get("gold"))
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "greedy_accuracy": accuracy,
    }


def analyze_branch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    leader_correct = [bool(r.get("leader_correct")) for r in records]
    leader_accuracy = sum(1 for v in leader_correct if v) / total if total else 0.0

    unanimous = [r for r in records if r.get("max_frac") == 1.0]
    near_unanimous = [r for r in records if (r.get("max_frac") or 0.0) >= 0.9]

    unanimous_acc = (
        sum(1 for r in unanimous if r.get("leader_correct")) / len(unanimous)
        if unanimous
        else 0.0
    )
    near_unanimous_acc = (
        sum(1 for r in near_unanimous if r.get("leader_correct")) / len(near_unanimous)
        if near_unanimous
        else 0.0
    )

    top2_hits = sum(
        1
        for r in records
        if top2_coverage(r.get("branch_preds", []), r.get("gold"))
    )
    top2_rate = top2_hits / total if total else 0.0

    r_w_other_counts = {"R": 0, "W": 0, "Other": 0}
    for r in records:
        label = r_w_other_class(r.get("max_frac") or 0.0, r.get("leader_correct"))
        r_w_other_counts[label] += 1

    return {
        "total": total,
        "leader_accuracy": leader_accuracy,
        "unanimous": {
            "count": len(unanimous),
            "accuracy": unanimous_acc,
        },
        "near_unanimous": {
            "count": len(near_unanimous),
            "accuracy": near_unanimous_acc,
        },
        "top2_coverage": {
            "count": top2_hits,
            "rate": top2_rate,
        },
        "r_w_other": r_w_other_counts,
    }


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.summary)

    if not records:
        report = {"total": 0}
    elif "prediction" in records[0]:
        report = analyze_greedy(records)
    else:
        report = analyze_branch(records)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    output_path = args.output
    if not output_path:
        output_path = os.path.join(os.path.dirname(args.summary), "report.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
