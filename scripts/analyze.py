"""Analyze summary.jsonl logs to compute aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

from rofa.core.io import unpack_zip
from rofa.core.metrics import r_w_other_class, top2_coverage


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for analysis."""
    parser = argparse.ArgumentParser(description="Analyze ROFA summary logs.")
    parser.add_argument(
        "--run",
        dest="run_paths",
        nargs="+",
        required=True,
        help="Run directory or zip path(s) to analyze.",
    )
    parser.add_argument("--output", required=False, help="Path to output report JSON")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL records into a list."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def analyze_greedy(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute greedy accuracy metrics from summary records."""
    total = len(records)
    correct = sum(1 for r in records if r.get("prediction") == r.get("gold"))
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "greedy_accuracy": accuracy,
    }


def analyze_branch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute ensemble consensus metrics from summary records."""
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

    top2_hits = 0
    for record in records:
        gold = record.get("gold")
        if isinstance(gold, str) and top2_coverage(record.get("branch_preds", []), gold):
            top2_hits += 1
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


def _resolve_run_dir(run_path: str) -> Tuple[str, bool]:
    if os.path.isdir(run_path):
        return run_path, False
    if run_path.endswith(".zip") and os.path.isfile(run_path):
        tmp_dir = tempfile.mkdtemp(prefix="rofa_run_")
        unpack_zip(run_path, tmp_dir)
        return tmp_dir, True
    raise FileNotFoundError(f"Run directory or zip not found: {run_path}")


def _default_report_path(run_id: str) -> str:
    reports_root = os.path.join("notebooks", "reports", run_id)
    os.makedirs(reports_root, exist_ok=True)
    return os.path.join(reports_root, "report.json")


def main() -> None:
    """Generate analysis reports for one or more runs.

    Outputs:
        Writes ``report.json`` under ``notebooks/reports/<run_id>/`` by default.

    Failure modes:
        Raises FileNotFoundError if the run directory or summary log is missing.
    """
    args = parse_args()
    for run_path in args.run_paths:
        run_dir, is_temp = _resolve_run_dir(run_path)
        if is_temp:
            pass

        summary_path = os.path.join(run_dir, "summary.jsonl")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"summary.jsonl not found in {run_dir}")

        records = load_jsonl(summary_path)
        if not records:
            report = {"total": 0}
        elif "prediction" in records[0]:
            report = analyze_greedy(records)
        else:
            report = analyze_branch(records)

        print(json.dumps(report, indent=2, ensure_ascii=False))

        output_path = args.output
        if not output_path:
            run_id = os.path.basename(os.path.abspath(run_dir))
            output_path = _default_report_path(run_id)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
