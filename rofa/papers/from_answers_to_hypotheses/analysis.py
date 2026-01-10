"""Notebook-friendly analysis helpers for ROFA runs."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd

from rofa.core.io import unpack_zip
from rofa.core.metrics import r_w_other_class
from rofa.core.metrics import top2_coverage as metrics_top2_coverage

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
        predictions = cast(pd.Series, df_greedy["prediction"])
        gold = cast(pd.Series, df_greedy["gold"])
        return float((predictions == gold).mean())
    if "is_correct" in df_greedy.columns:
        is_correct = cast(pd.Series, df_greedy["is_correct"])
        return float(is_correct.fillna(False).astype(bool).mean())
    raise ValueError("DataFrame does not contain greedy prediction fields.")


def accuracy_leader(df_branches: pd.DataFrame) -> float:
    """Compute leader accuracy from a branch summary DataFrame."""
    if "leader_correct" not in df_branches.columns:
        raise ValueError("DataFrame does not contain leader_correct.")
    leader_correct = cast(pd.Series, df_branches["leader_correct"])
    return float(leader_correct.fillna(False).astype(bool).mean())


def unanimous_stats(df_branches: pd.DataFrame) -> Dict[str, float]:
    """Compute unanimous count and accuracy for max_frac == 1.0."""
    if "max_frac" not in df_branches.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    max_frac = cast(pd.Series, df_branches["max_frac"])
    unanimous = df_branches[max_frac == 1.0]
    count = len(unanimous)
    accuracy = (
        float(
            cast(pd.Series, unanimous["leader_correct"]).fillna(False).astype(bool).mean()
        )
        if count
        else 0.0
    )
    return {"count": count, "accuracy": accuracy}


def near_unanimous_stats(df_branches: pd.DataFrame, threshold: float = 0.9) -> Dict[str, float]:
    """Compute near-unanimous count and accuracy for max_frac >= threshold."""
    if "max_frac" not in df_branches.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    max_frac = cast(pd.Series, df_branches["max_frac"])
    near = df_branches[max_frac >= threshold]
    count = len(near)
    accuracy = (
        float(cast(pd.Series, near["leader_correct"]).fillna(False).astype(bool).mean())
        if count
        else 0.0
    )
    return {"count": count, "accuracy": accuracy}


def top2_coverage(df_branches: pd.DataFrame) -> float:
    """Compute top-2 coverage rate for branch predictions."""
    if "branch_preds" not in df_branches.columns or "gold" not in df_branches.columns:
        raise ValueError("DataFrame does not contain branch_preds/gold.")
    branch_preds = cast(pd.Series, df_branches["branch_preds"])
    gold_series = cast(pd.Series, df_branches["gold"])
    hits = sum(
        1
        for preds, gold in zip(branch_preds, gold_series, strict=False)
        if isinstance(gold, str) and metrics_top2_coverage(preds, gold)
    )
    total = len(df_branches)
    return hits / total if total else 0.0


def compute_max_frac_exact(branch_preds: Sequence[Optional[str]]) -> float:
    """Compute max_frac_exact using full branch count as denominator."""
    total = len(branch_preds)
    if total == 0:
        return 0.0
    valid = [
        pred
        for pred in branch_preds
        if isinstance(pred, str) and pred.strip() and not pd.isna(pred)
    ]
    if not valid:
        return 0.0
    counts = Counter(valid)
    max_count = max(counts.values()) if counts else 0
    return max_count / total


def accuracy_by_max_frac_exact(df_branches: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy grouped by exact max_frac_exact values."""
    if "branch_preds" not in df_branches.columns:
        raise ValueError("DataFrame does not contain branch_preds.")
    if "leader_correct" in df_branches.columns:
        leader_correct = cast(pd.Series, df_branches["leader_correct"]).fillna(False).astype(bool)
    elif {"leader", "gold"}.issubset(df_branches.columns):
        leader = cast(pd.Series, df_branches["leader"])
        gold = cast(pd.Series, df_branches["gold"])
        leader_correct = (leader == gold).fillna(False).astype(bool)
    else:
        raise ValueError("DataFrame does not contain leader_correct or leader/gold.")

    max_frac_exact = cast(pd.Series, df_branches["branch_preds"]).apply(compute_max_frac_exact)
    grouped = (
        pd.DataFrame({"max_frac_exact": max_frac_exact, "leader_correct": leader_correct})
        .groupby("max_frac_exact")
        .agg(count=("leader_correct", "size"), accuracy=("leader_correct", "mean"))
        .reset_index()
        .sort_values("max_frac_exact")
    )
    grouped["error_rate"] = 1.0 - grouped["accuracy"]
    return cast(pd.DataFrame, grouped)


def max_frac_distribution(
    df_branches: pd.DataFrame,
    bins: Sequence[float] = (0.0, 0.5, 0.8, 0.9, 1.0),
) -> pd.Series:
    """Return a histogram of max_frac across bins."""
    if "max_frac" not in df_branches.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    max_frac = cast(pd.Series, df_branches["max_frac"])
    categories = cast(pd.Series, pd.cut(max_frac, bins=bins, include_lowest=True))
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
    max_frac = cast(pd.Series, df_branches["max_frac"])
    leader_correct = cast(pd.Series, df_branches["leader_correct"])
    labels = []
    for mf, lc in zip(max_frac, leader_correct, strict=False):
        lc_value = None if pd.isna(lc) else bool(lc)
        mf_value = float(mf) if pd.notna(mf) else float("nan")
        labels.append(r_w_other_class(mf_value, lc_value))
    bins_series = pd.cut(max_frac, bins=bins, include_lowest=True)
    breakdown = (
        pd.DataFrame({"bin": bins_series, "label": labels})
        .groupby(["bin", "label"])
        .size()
        .unstack(fill_value=0)
    )
    return cast(pd.DataFrame, breakdown)


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
    max_frac = cast(pd.Series, df_branches["max_frac"])
    leader_correct = cast(pd.Series, df_branches["leader_correct"]).fillna(False).astype(bool)
    return cast(pd.DataFrame, df_branches.loc[(max_frac == 1.0) & (~leader_correct)])


def subject_accuracy(df_summary: pd.DataFrame, accuracy_field: str = "leader_correct") -> pd.Series:
    """Compute accuracy by subject if subject_name is present."""
    if "subject_name" not in df_summary.columns:
        raise ValueError("DataFrame does not contain subject_name.")
    if accuracy_field not in df_summary.columns:
        raise ValueError(f"DataFrame does not contain {accuracy_field}.")
    grouped = df_summary.groupby("subject_name")[accuracy_field].mean()
    return cast(pd.Series, grouped).sort_values(ascending=False)


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
            "metric": [
                "unanimous_count",
                "unanimous_accuracy",
                "near_unanimous_count",
                "near_unanimous_accuracy",
            ],
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
