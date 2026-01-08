"""Validate ROFA run artifacts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ROFA run artifacts.")
    parser.add_argument("--run-dir", required=True, help="Path to a run directory.")
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}: {exc}") from exc
    return records


def _require_keys(record: Dict[str, Any], keys: List[str], context: str) -> List[str]:
    missing = [key for key in keys if key not in record]
    if missing:
        return [f"{context}: missing {', '.join(missing)}"]
    return []


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise ValueError(f"Run directory not found: {run_dir}")

    manifest_path = os.path.join(run_dir, "manifest.json")
    summary_path = os.path.join(run_dir, "summary.jsonl")
    progress_path = os.path.join(run_dir, "progress.json")

    errors: List[str] = []

    if not os.path.exists(summary_path):
        errors.append("summary.jsonl is missing.")
        summary_records = []
    else:
        summary_records = _load_jsonl(summary_path)

    if os.path.exists(manifest_path):
        manifest = _load_json(manifest_path)
        errors.extend(_require_keys(manifest, ["run_id", "created_at", "method", "config"], "manifest"))
        method = manifest.get("method")
        config = manifest.get("config", {})
    else:
        manifest = {}
        method = None
        config = {}
        errors.append("manifest.json is missing.")

    if os.path.exists(progress_path):
        progress = _load_json(progress_path)
        errors.extend(_require_keys(progress, ["run_id", "i", "picked", "summary_written"], "progress"))
        summary_written = progress.get("summary_written")
        if isinstance(summary_written, int) and summary_written != len(summary_records):
            errors.append(
                f"progress summary_written={summary_written} but summary has {len(summary_records)} records."
            )
    else:
        progress = {}
        summary_written = None

    if method is None and summary_records:
        sample = summary_records[0]
        method = "branches" if "branch_preds" in sample else "greedy"

    for idx, record in enumerate(summary_records):
        errors.extend(_require_keys(record, ["index", "id", "gold", "subject_name", "timestamp"], f"summary[{idx}]"))
        if method == "greedy":
            errors.extend(
                _require_keys(record, ["prediction", "is_correct", "model_output"], f"summary[{idx}]")
            )
        elif method == "branches":
            errors.extend(
                _require_keys(record, ["branch_preds", "leader", "max_frac", "valid_n"], f"summary[{idx}]")
            )

    if config:
        expected_n = config.get("n")
        if isinstance(expected_n, int) and summary_records and len(summary_records) > expected_n:
            errors.append(
                f"summary has {len(summary_records)} records but manifest config.n={expected_n}."
            )

    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))

    print(f"Run directory OK: {run_dir}")
    print(f"summary.jsonl records: {len(summary_records)}")
    if summary_written is not None:
        print(f"progress summary_written: {summary_written}")


if __name__ == "__main__":
    main()
