"""Helper utilities to keep notebook cells concise."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast
from urllib.parse import urlparse

import pandas as pd

from rofa.core.io import download, unpack_zip
from rofa.papers.from_answers_to_hypotheses import analysis


def _download_and_unpack(asset_url: str) -> str:
    runs_root = Path("runs")
    runs_root.mkdir(exist_ok=True)
    filename = Path(urlparse(asset_url).path).name or "run.zip"
    zip_path = runs_root / filename
    download(asset_url, str(zip_path))
    run_dir = runs_root / zip_path.stem
    unpack_zip(str(zip_path), str(run_dir))
    return str(run_dir)


def resolve_run_input(run_dir: str, asset_url: str) -> str:
    if run_dir:
        return run_dir
    if asset_url:
        return _download_and_unpack(asset_url)
    return ""


def resolve_run_inputs(
    run_dir_greedy: str,
    greedy_asset_url: str,
    run_dir_k_sample: str,
    k_sample_asset_url: str,
) -> list[str]:
    run_inputs = [
        resolve_run_input(run_dir_greedy, greedy_asset_url),
        resolve_run_input(run_dir_k_sample, k_sample_asset_url),
    ]
    run_inputs = [run_input for run_input in run_inputs if run_input]
    if len(run_inputs) < 2:
        raise ValueError("Provide both greedy and k-sample ensemble runs.")
    return run_inputs


def validate_required_columns(df_greedy: pd.DataFrame, df_branches: pd.DataFrame) -> None:
    required_greedy_cols = {"gold", "prediction", "is_correct"}
    required_branch_cols = {"gold", "leader", "leader_correct", "max_frac", "branch_preds"}

    missing_greedy = required_greedy_cols - set(df_greedy.columns)
    missing_branches = required_branch_cols - set(df_branches.columns)
    if missing_greedy:
        raise ValueError(f"Greedy run missing required columns: {missing_greedy}")
    if missing_branches:
        raise ValueError(f"k-sample run missing required columns: {missing_branches}")


def print_run_summary(
    df_greedy: pd.DataFrame, df_branches: pd.DataFrame, metadata: Dict[str, Any]
) -> None:
    print("df_greedy:", df_greedy.shape)
    print("df_branches:", df_branches.shape)
    print("Resolved runs:", metadata.get("resolved_runs"))


def plot_accuracy_vs_consensus(
    df_branches: pd.DataFrame, output_path: str = "figure1_max_frac_exact.png"
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    df_max_frac_exact = analysis.accuracy_by_max_frac_exact(df_branches)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_max_frac_exact["max_frac_exact"], df_max_frac_exact["accuracy"], marker="o")
    ax.set_xlabel("max_frac_exact (leader fraction, N=10)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Internal Consensus (max_frac_exact)")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.show()
    return df_max_frac_exact


def majority_vote_table(df_greedy: pd.DataFrame, df_branches: pd.DataFrame) -> pd.DataFrame:
    merge_keys = [
        key
        for key in ["id", "index", "question"]
        if key in df_greedy.columns and key in df_branches.columns
    ]
    if not merge_keys:
        raise ValueError("No shared keys available to merge greedy and k-sample runs.")

    df_merged = df_greedy.merge(df_branches, on=merge_keys, suffixes=("_greedy", "_branches"))
    greedy_correct = df_merged["is_correct"].fillna(False).astype(bool)
    leader_correct = df_merged["leader_correct"].fillna(False).astype(bool)
    return pd.DataFrame(
        {
            "metric": ["greedy_accuracy", "leader_accuracy"],
            "value": [greedy_correct.mean(), leader_correct.mean()],
        }
    )


def _resolve_report_dir(metadata: Dict[str, Any]) -> Path:
    runs = cast(Dict[str, str], metadata.get("resolved_runs", {}))
    run_path = runs.get("k_sample_ensemble") or runs.get("greedy")
    if run_path is None:
        raise ValueError("No run path found in metadata.")
    run_id = Path(run_path).name
    report_dir = Path("notebooks") / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def export_paper_reports(
    metadata: Dict[str, Any],
    df_greedy_accuracy: pd.DataFrame,
    df_leader_accuracy: pd.DataFrame,
    unanimous_stats: Dict[str, float],
    near_unanimous_stats: Dict[str, float],
    df_top2: pd.DataFrame,
    df_max_frac: pd.DataFrame,
    df_rw_other: pd.DataFrame,
    df_subject_breakdown: pd.DataFrame,
) -> Path:
    report_dir = _resolve_report_dir(metadata)

    def _cell_to_float(df: pd.DataFrame, column: str) -> float:
        return float(df.at[df.index[0], column])

    paper_report = {
        "greedy_accuracy": _cell_to_float(df_greedy_accuracy, "value"),
        "leader_accuracy": _cell_to_float(df_leader_accuracy, "value"),
        "unanimous": unanimous_stats,
        "near_unanimous": near_unanimous_stats,
        "top2_coverage": _cell_to_float(df_top2, "value"),
    }
    report_path = report_dir / "paper_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(paper_report, handle, indent=2)

    df_max_frac.to_csv(report_dir / "max_frac_distribution.csv", index=False)
    df_rw_other.to_csv(report_dir / "rw_other_breakdown.csv")
    df_subject_breakdown.to_csv(report_dir / "subject_accuracy.csv")
    return report_dir


def subject_breakdown(df_greedy: pd.DataFrame, df_branches: pd.DataFrame) -> pd.DataFrame:
    df_subject_greedy = analysis.subject_accuracy(df_greedy, accuracy_field="is_correct")
    df_subject_branches = analysis.subject_accuracy(
        df_branches, accuracy_field="leader_correct"
    )
    return pd.DataFrame(
        {
            "greedy_accuracy": df_subject_greedy,
            "leader_accuracy": df_subject_branches,
        }
    )
