"""Analyze summary.jsonl logs to compute aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os

from rofa.papers.from_answers_to_hypotheses import analysis as paper_analysis


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


def _default_report_path(run_id: str) -> str:
    reports_root = os.path.join("notebooks", "from_answers_to_hypotheses", "reports", run_id)
    os.makedirs(reports_root, exist_ok=True)
    return os.path.join(reports_root, "report.json")


def main() -> None:
    """Generate analysis reports for one or more runs.

    Outputs:
        Writes ``report.json`` under 
        ``notebooks/from_answers_to_hypotheses/reports/<run_id>/`` by default.

    Failure modes:
        Raises FileNotFoundError if the run directory or summary log is missing.
    """
    args = parse_args()
    for run_path in args.run_paths:
        run_dir, is_temp = paper_analysis.resolve_run_dir(run_path)
        if is_temp:
            pass

        summary_path = os.path.join(run_dir, "summary.jsonl")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"summary.jsonl not found in {run_dir}")

        df_summary = paper_analysis.load_summary(run_dir)
        report = paper_analysis.run_report(df_summary)

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
