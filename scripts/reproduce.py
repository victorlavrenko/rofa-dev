"""Reproduce a ROFA report from a run directory."""

from __future__ import annotations

import argparse
import json
import os

from rofa.core.model_id import to_slug
from rofa.core.run_paths import find_latest_run_dir
from rofa.papers.from_answers_to_hypotheses import analysis as paper_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce ROFA report artifacts.")
    parser.add_argument("--run-dir", help="Explicit path to a run directory.")
    parser.add_argument("--model", help="Optional model id to locate the latest run.")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs (default: runs).",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for report.json (default: <run-dir>/report.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir and args.model:
        model_slug = to_slug(args.model)
        run_dir = find_latest_run_dir(args.runs_root, model_slug)

    if not run_dir:
        raise ValueError("Provide --run-dir or --model to locate a run.")

    print(f"Using run directory: {run_dir}")

    summary_df = paper_analysis.load_summary(run_dir)
    run_metadata = paper_analysis.load_run_metadata(run_dir)
    report = paper_analysis.run_report(summary_df, run_metadata=run_metadata)

    output_path = args.output or os.path.join(run_dir, "report.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
