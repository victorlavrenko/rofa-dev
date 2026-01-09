"""Notebook-friendly analysis helpers for ROFA runs."""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from rofa.core.io import unpack_zip
from rofa.core.metrics import r_w_other_class, top2_coverage as metrics_top2_coverage

METHOD_GREEDY = "greedy"
METHOD_K_SAMPLE = "k_sample_ensemble"


def load_summary(run_dir: str) -> pd.DataFrame:
    """Load summary.jsonl from a run directory into a DataFrame."""
    summary_path = os.path.join(run_dir, "summary.jsonl")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.jsonl not found in {run_dir}")
    return pd.read_json(summary_path, lines=True)


def _resolve_run_dir(run_dir_or_zip: str) -> Tuple[str, bool]:
    if os.path.isdir(run_dir_or_zip):
        return run_dir_or_zip, False
    if run_dir_or_zip.endswith(".zip") and os.path.isfile(run_dir_or_zip):
        tmp_dir = tempfile.mkdtemp(prefix="rofa_run_")
        unpack_zip(run_dir_or_zip, tmp_dir)
        return tmp_dir, True
    raise FileNotFoundError(f"Run directory or zip not found: {run_dir_or_zip}")


def load_paper_runs(
    run_dirs_or_zips: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Load greedy and k-sample ensemble runs for paper reproduction.

    Args:
        run_dirs_or_zips: Sequence of run directories or packed zip paths.

    Returns:
        Tuple of (df_greedy, df_branches, metadata). Metadata includes resolved run
        directories and any temporary extraction paths.

    Raises:
        ValueError: If required artifacts are missing or schema drift is detected.
    """
    df_greedy: Optional[pd.DataFrame] = None
    df_branches: Optional[pd.DataFrame] = None
    resolved_runs: Dict[str, str] = {}
    temp_paths: List[str] = []

    for run_dir_or_zip in run_dirs_or_zips:
        run_dir, is_temp = _resolve_run_dir(run_dir_or_zip)
        if is_temp:
            temp_paths.append(run_dir)

        manifest_path = os.path.join(run_dir, "manifest.json")
        method = None
        if os.path.exists(manifest_path):
            manifest = pd.read_json(manifest_path, typ="series")
            method = manifest.get("method")
        summary_df = load_summary(run_dir)
        if method is None:
            method = METHOD_K_SAMPLE if "branch_preds" in summary_df.columns else METHOD_GREEDY

        if method == METHOD_GREEDY:
            df_greedy = summary_df
            resolved_runs["greedy"] = run_dir
        elif method in {METHOD_K_SAMPLE, "branches"}:
            df_branches = summary_df
            resolved_runs["k_sample_ensemble"] = run_dir
        else:
            raise ValueError(f"Unsupported method in {run_dir}: {method}")

    if df_greedy is None:
        raise ValueError("Missing greedy run for paper reproduction.")
    if df_branches is None:
        raise ValueError("Missing k_sample_ensemble run for paper reproduction.")

    return df_greedy, df_branches, {"resolved_runs": resolved_runs, "temp_dirs": temp_paths}


def accuracy_greedy(df_greedy: pd.DataFrame) -> float:
    """Compute greedy accuracy from a summary DataFrame."""
    if "prediction" in df_greedy.columns and "gold" in df_greedy.columns:
        return (df_greedy["prediction"] == df_greedy["gold"]).mean()
    if "is_correct" in df_greedy.columns:
        return df_greedy["is_correct"].fillna(False).astype(bool).mean()
    raise ValueError("DataFrame does not contain greedy prediction fields.")


def accuracy_leader(df_branches: pd.DataFrame) -> float:
    """Compute leader accuracy from a branch summary DataFrame."""
    if "leader_correct" not in df_branches.columns:
        raise ValueError("DataFrame does not contain leader_correct.")
    return df_branches["leader_correct"].fillna(False).astype(bool).mean()


def unanimous_stats(df_branches: pd.DataFrame) -> Dict[str, float]:
    """Compute unanimous count and accuracy for max_frac == 1.0."""
    if "max_frac" not in df_branches.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    unanimous = df_branches[df_branches["max_frac"] == 1.0]
    count = len(unanimous)
    accuracy = (
        unanimous["leader_correct"].fillna(False).astype(bool).mean() if count else 0.0
    )
    return {"count": count, "accuracy": accuracy}


def near_unanimous_stats(df_branches: pd.DataFrame, threshold: float = 0.9) -> Dict[str, float]:
    """Compute near-unanimous count and accuracy for max_frac >= threshold."""
    if "max_frac" not in df_branches.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    near = df_branches[df_branches["max_frac"] >= threshold]
    count = len(near)
    accuracy = near["leader_correct"].fillna(False).astype(bool).mean() if count else 0.0
    return {"count": count, "accuracy": accuracy}


def top2_coverage(df_branches: pd.DataFrame) -> float:
    """Compute top-2 coverage rate for branch predictions."""
    if "branch_preds" not in df_branches.columns or "gold" not in df_branches.columns:
        raise ValueError("DataFrame does not contain branch_preds/gold.")
    hits = sum(
        1
        for preds, gold in zip(df_branches["branch_preds"], df_branches["gold"])
        if metrics_top2_coverage(preds, gold)
    )
    total = len(df_branches)
    return hits / total if total else 0.0


def max_frac_distribution(
    df_branches: pd.DataFrame,
    bins: Sequence[float] = (0.0, 0.5, 0.8, 0.9, 1.0),
) -> pd.Series:
    """Return a histogram of max_frac across bins."""
    if "max_frac" not in df_branches.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    categories = pd.cut(df_branches["max_frac"], bins=bins, include_lowest=True)
    return categories.value_counts().sort_index()


def rw_other_breakdown(
    df_branches: pd.DataFrame,
    bins: Sequence[float] = (0.0, 0.5, 0.8, 0.9, 1.0),
) -> pd.DataFrame:
    """Return an R/W/Other breakdown grouped by max_frac bins."""
    if "max_frac" not in df_branches.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    if "leader_correct" not in df_branches.columns:
        raise ValueError("DataFrame does not contain leader_correct.")
    labels = []
    for mf, lc in zip(df_branches["max_frac"], df_branches["leader_correct"]):
        lc_value = None if pd.isna(lc) else bool(lc)
        labels.append(r_w_other_class(mf, lc_value))
    bins_series = pd.cut(df_branches["max_frac"], bins=bins, include_lowest=True)
    breakdown = (
        pd.DataFrame({"bin": bins_series, "label": labels})
        .groupby(["bin", "label"])
        .size()
        .unstack(fill_value=0)
    )
    return breakdown


def paper_metrics(df_summary: pd.DataFrame) -> Dict[str, object]:
    """Compute a bundle of paper metrics for a summary DataFrame."""
    metrics: Dict[str, object] = {"total": len(df_summary)}
    if "prediction" in df_summary.columns:
        metrics["greedy_accuracy"] = accuracy_greedy(df_summary)
    if "leader_correct" in df_summary.columns:
        metrics["leader_accuracy"] = accuracy_leader(df_summary)
        metrics["unanimous"] = unanimous_stats(df_summary)
        metrics["near_unanimous"] = near_unanimous_stats(df_summary)
        metrics["top2_coverage"] = top2_coverage(df_summary)
    return metrics


def unanimous_wrong(df_branches: pd.DataFrame) -> pd.DataFrame:
    """Return unanimous-but-wrong cases for branch runs."""
    if "max_frac" not in df_branches.columns or "leader_correct" not in df_branches.columns:
        raise ValueError("DataFrame does not contain unanimous fields.")
    return df_branches[(df_branches["max_frac"] == 1.0) & (df_branches["leader_correct"] == False)]


def subject_accuracy(df_summary: pd.DataFrame, accuracy_field: str = "leader_correct") -> pd.Series:
    """Compute accuracy by subject if subject_name is present."""
    if "subject_name" not in df_summary.columns:
        raise ValueError("DataFrame does not contain subject_name.")
    if accuracy_field not in df_summary.columns:
        raise ValueError(f"DataFrame does not contain {accuracy_field}.")
    return df_summary.groupby("subject_name")[accuracy_field].mean().sort_values(ascending=False)


def compute_table_accuracy(df_greedy: pd.DataFrame, df_branches: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with greedy and leader accuracy."""
    return pd.DataFrame(
        {
            "metric": ["greedy_accuracy", "leader_accuracy"],
            "value": [accuracy_greedy(df_greedy), accuracy_leader(df_branches)],
        }
    )


def compute_table_consensus(df_branches: pd.DataFrame) -> pd.DataFrame:
    """Return consensus statistics for branch predictions."""
    unanimous = unanimous_stats(df_branches)
    near_unanimous = near_unanimous_stats(df_branches)
    return pd.DataFrame(
        {
            "metric": ["unanimous_count", "unanimous_accuracy", "near_unanimous_count", "near_unanimous_accuracy"],
            "value": [
                unanimous["count"],
                unanimous["accuracy"],
                near_unanimous["count"],
                near_unanimous["accuracy"],
            ],
        }
    )


def compute_table_top2(df_branches: pd.DataFrame) -> pd.DataFrame:
    """Return top-2 coverage rate as a single-row table."""
    return pd.DataFrame({"metric": ["top2_coverage"], "value": [top2_coverage(df_branches)]})
