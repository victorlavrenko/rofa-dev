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
    parser.add_argument(
        "--top2-flip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute top-2 flip subset discovery outputs.",
    )
    parser.add_argument(
        "--top2-flip-min-support",
        type=int,
        default=5,
        help="Minimum support for top-2 flip rectangle aggregation.",
    )
    parser.add_argument(
        "--top2-flip-thresholds",
        default=",".join([f"{1.0 + 0.1 * i:.1f}" for i in range(41)]),
        help="Comma-separated ratio thresholds for top-2 flip reporting.",
    )
    parser.add_argument(
        "--top2-flip-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require unique top-1 and top-2 (strict mode) for top-2 flip analysis.",
    )
    return parser.parse_args()


def _default_report_path(run_id: str) -> str:
    reports_root = os.path.join("notebooks", "from_answers_to_hypotheses", "reports", run_id)
    os.makedirs(reports_root, exist_ok=True)
    return os.path.join(reports_root, "report.json")


def _parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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

        if args.top2_flip:
            if "branch_preds" not in df_summary.columns or "gold" not in df_summary.columns:
                raise ValueError("top2-flip requires branch_preds and gold columns.")
            thresholds = _parse_thresholds(args.top2_flip_thresholds)
            matrix, rectangles, threshold_rectangles, tie_stats = (
                paper_analysis.top2_flip_analysis(
                    df_summary,
                    strict=args.top2_flip_strict,
                    min_support=args.top2_flip_min_support,
                    ratio_thresholds=thresholds,
                )
            )
            output_dir = os.path.dirname(output_path) or os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            matrix.to_csv(os.path.join(output_dir, "top2_flip_matrix.csv"), index=False)
            rectangles.to_csv(os.path.join(output_dir, "top2_flip_rectangles.csv"), index=False)
            threshold_rectangles.to_csv(
                os.path.join(output_dir, "top2_flip_thresholds.csv"), index=False
            )
            with open(
                os.path.join(output_dir, "top2_flip_tie_stats.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(tie_stats, handle, ensure_ascii=False, indent=2)
            print(f"Top-2 flip outputs written to {output_dir}")


if __name__ == "__main__":
    main()
