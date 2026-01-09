"""Validate ROFA run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

from rofa.core.io import unpack_zip


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for run validation."""
    parser = argparse.ArgumentParser(description="Validate ROFA run artifacts.")
    parser.add_argument("--run", required=True, help="Path to a run directory or zip.")
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file with line-level error reporting."""
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
    """Return list of missing key errors for a record."""
    missing = [key for key in keys if key not in record]
    if missing:
        return [f"{context}: missing {', '.join(missing)}"]
    return []


def _resolve_run_dir(run_path: str) -> Tuple[str, bool]:
    if os.path.isdir(run_path):
        return os.path.abspath(run_path), False
    if run_path.endswith(".zip") and os.path.isfile(run_path):
        tmp_dir = tempfile.mkdtemp(prefix="rofa_run_")
        unpack_zip(run_path, tmp_dir)
        return tmp_dir, True
    raise ValueError(f"Run directory or zip not found: {run_path}")


def main() -> None:
    """Validate required run artifacts and summary schema.

    Outputs:
        Prints a human-readable summary on success.

    Failure modes:
        Raises SystemExit with a list of missing files/columns on validation errors.
    """
    args = parse_args()
    run_dir, _ = _resolve_run_dir(args.run)

    manifest_path = os.path.join(run_dir, "manifest.json")
    summary_path = os.path.join(run_dir, "summary.jsonl")
    progress_path = os.path.join(run_dir, "progress.json")
    question_set_path = os.path.join(run_dir, "question_set.json")

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
        errors.extend(
            _require_keys(progress, ["run_id", "position", "summary_written", "full_written"], "progress")
        )
        summary_written = progress.get("summary_written")
        if isinstance(summary_written, int) and summary_written != len(summary_records):
            errors.append(
                f"progress summary_written={summary_written} but summary has {len(summary_records)} records."
            )
    else:
        progress = {}
        summary_written = None

    if not os.path.exists(question_set_path):
        errors.append("question_set.json is missing.")

    if method is None and summary_records:
        sample = summary_records[0]
        method = "k_sample_ensemble" if "branch_preds" in sample else "greedy"

    for idx, record in enumerate(summary_records):
        errors.extend(_require_keys(record, ["index", "id", "gold", "subject_name", "timestamp"], f"summary[{idx}]"))
        if method == "greedy":
            errors.extend(
                _require_keys(record, ["prediction", "is_correct", "model_output"], f"summary[{idx}]")
            )
        elif method in {"k_sample_ensemble", "branches"}:
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
