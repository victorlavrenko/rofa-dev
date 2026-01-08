"""Notebook-friendly analysis helpers for ROFA runs."""

from __future__ import annotations

import os
from typing import Dict, Sequence

import pandas as pd

from .metrics import r_w_other_class, top2_coverage as metrics_top2_coverage


def load_summary(run_dir: str) -> pd.DataFrame:
    """Load summary.jsonl from a run directory into a DataFrame."""
    summary_path = os.path.join(run_dir, "summary.jsonl")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.jsonl not found in {run_dir}")
    return pd.read_json(summary_path, lines=True)


def accuracy_greedy(df: pd.DataFrame) -> float:
    """Compute greedy accuracy from a summary DataFrame."""
    if "prediction" in df.columns and "gold" in df.columns:
        return (df["prediction"] == df["gold"]).mean()
    if "is_correct" in df.columns:
        return df["is_correct"].fillna(False).astype(bool).mean()
    raise ValueError("DataFrame does not contain greedy prediction fields.")


def accuracy_leader(df: pd.DataFrame) -> float:
    """Compute leader accuracy from a branch summary DataFrame."""
    if "leader_correct" not in df.columns:
        raise ValueError("DataFrame does not contain leader_correct.")
    return df["leader_correct"].fillna(False).astype(bool).mean()


def unanimous_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute unanimous count and accuracy for max_frac == 1.0."""
    if "max_frac" not in df.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    unanimous = df[df["max_frac"] == 1.0]
    count = len(unanimous)
    accuracy = (
        unanimous["leader_correct"].fillna(False).astype(bool).mean() if count else 0.0
    )
    return {"count": count, "accuracy": accuracy}


def near_unanimous_stats(df: pd.DataFrame, threshold: float = 0.9) -> Dict[str, float]:
    """Compute near-unanimous count and accuracy for max_frac >= threshold."""
    if "max_frac" not in df.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    near = df[df["max_frac"] >= threshold]
    count = len(near)
    accuracy = near["leader_correct"].fillna(False).astype(bool).mean() if count else 0.0
    return {"count": count, "accuracy": accuracy}


def top2_coverage(df: pd.DataFrame) -> float:
    """Compute top-2 coverage rate for branch predictions."""
    if "branch_preds" not in df.columns or "gold" not in df.columns:
        raise ValueError("DataFrame does not contain branch_preds/gold.")
    hits = sum(
        1
        for preds, gold in zip(df["branch_preds"], df["gold"])
        if metrics_top2_coverage(preds, gold)
    )
    total = len(df)
    return hits / total if total else 0.0


def max_frac_distribution(
    df: pd.DataFrame,
    bins: Sequence[float] = (0.0, 0.5, 0.8, 0.9, 1.0),
) -> pd.Series:
    """Return a histogram of max_frac across bins."""
    if "max_frac" not in df.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    categories = pd.cut(df["max_frac"], bins=bins, include_lowest=True)
    return categories.value_counts().sort_index()


def rw_other_breakdown(
    df: pd.DataFrame,
    bins: Sequence[float] = (0.0, 0.5, 0.8, 0.9, 1.0),
) -> pd.DataFrame:
    """Return an R/W/Other breakdown grouped by max_frac bins."""
    if "max_frac" not in df.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    if "leader_correct" not in df.columns:
        raise ValueError("DataFrame does not contain leader_correct.")
    labels = []
    for mf, lc in zip(df["max_frac"], df["leader_correct"]):
        lc_value = None if pd.isna(lc) else bool(lc)
        labels.append(r_w_other_class(mf, lc_value))
    bins_series = pd.cut(df["max_frac"], bins=bins, include_lowest=True)
    breakdown = (
        pd.DataFrame({"bin": bins_series, "label": labels})
        .groupby(["bin", "label"])
        .size()
        .unstack(fill_value=0)
    )
    return breakdown


def paper_metrics(df: pd.DataFrame) -> Dict[str, object]:
    """Compute a bundle of paper metrics for a summary DataFrame."""
    metrics: Dict[str, object] = {"total": len(df)}
    if "prediction" in df.columns:
        metrics["greedy_accuracy"] = accuracy_greedy(df)
    if "leader_correct" in df.columns:
        metrics["leader_accuracy"] = accuracy_leader(df)
        metrics["unanimous"] = unanimous_stats(df)
        metrics["near_unanimous"] = near_unanimous_stats(df)
        metrics["top2_coverage"] = top2_coverage(df)
    return metrics


def unanimous_wrong(df: pd.DataFrame) -> pd.DataFrame:
    """Return unanimous-but-wrong cases for branch runs."""
    if "max_frac" not in df.columns or "leader_correct" not in df.columns:
        raise ValueError("DataFrame does not contain unanimous fields.")
    return df[(df["max_frac"] == 1.0) & (df["leader_correct"] == False)]


def subject_accuracy(df: pd.DataFrame, accuracy_field: str = "leader_correct") -> pd.Series:
    """Compute accuracy by subject if subject_name is present."""
    if "subject_name" not in df.columns:
        raise ValueError("DataFrame does not contain subject_name.")
    if accuracy_field not in df.columns:
        raise ValueError(f"DataFrame does not contain {accuracy_field}.")
    return df.groupby("subject_name")[accuracy_field].mean().sort_values(ascending=False)
