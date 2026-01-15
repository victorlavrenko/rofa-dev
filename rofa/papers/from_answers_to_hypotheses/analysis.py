"""Notebook-friendly analysis helpers for ROFA runs."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import pandas as pd

from rofa.core.io import load_manifest, unpack_zip
from rofa.core.model_id import to_slug
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


def _metadata_from_summary(df_summary: pd.DataFrame) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    if "model_id" in df_summary.columns:
        model_id_series = df_summary["model_id"].dropna().astype(str)
        if not model_id_series.empty:
            model_id = model_id_series.iloc[0]
            metadata["model_id"] = model_id
            metadata["model_slug"] = to_slug(model_id)
    if "model_slug" in df_summary.columns:
        model_slug_series = df_summary["model_slug"].dropna().astype(str)
        if not model_slug_series.empty:
            metadata["model_slug"] = model_slug_series.iloc[0]
    decoding_fields = [
        "seed",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "n_branches",
    ]
    decoding = {
        field: df_summary[field].dropna().iloc[0]
        for field in decoding_fields
        if field in df_summary.columns and not df_summary[field].dropna().empty
    }
    if decoding:
        metadata["decoding_params"] = decoding
    return metadata


def load_run_metadata(run_dir: str) -> Dict[str, object]:
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return {}
    config = manifest.config
    model_slug = config.model_slug or to_slug(config.model_id)
    decoding = {
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "n_branches": config.n_branches,
    }
    return {
        "model_id": config.model_id,
        "model_slug": model_slug,
        "decoding_params": decoding,
    }


def resolve_run_dir(run_dir_or_zip: str) -> Tuple[str, bool]:
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
    resolved_metadata: Dict[str, Dict[str, object]] = {}

    for run_dir_or_zip in run_dirs_or_zips:
        run_dir, is_temp = resolve_run_dir(run_dir_or_zip)
        if is_temp:
            temp_paths.append(run_dir)

        manifest_path = os.path.join(run_dir, "manifest.json")
        method = None
        if os.path.exists(manifest_path):
            manifest = load_manifest(manifest_path)
            if manifest is not None:
                method = manifest.method
        summary_df = load_summary(run_dir)
        if method is None:
            method = METHOD_K_SAMPLE if "branch_preds" in summary_df.columns else METHOD_GREEDY

        if method == METHOD_GREEDY:
            df_greedy = summary_df
            resolved_runs["greedy"] = run_dir
            resolved_metadata["greedy"] = (
                load_run_metadata(run_dir) if os.path.exists(manifest_path) else {}
            )
        elif method in {METHOD_K_SAMPLE, "branches"}:
            df_branches = summary_df
            resolved_runs["k_sample_ensemble"] = run_dir
            resolved_metadata["k_sample_ensemble"] = (
                load_run_metadata(run_dir) if os.path.exists(manifest_path) else {}
            )
        else:
            raise ValueError(f"Unsupported method in {run_dir}: {method}")

    if df_greedy is None:
        raise ValueError("Missing greedy run for paper reproduction.")
    if df_branches is None:
        raise ValueError("Missing k_sample_ensemble run for paper reproduction.")

    return (
        df_greedy,
        df_branches,
        {
            "resolved_runs": resolved_runs,
            "temp_dirs": temp_paths,
            "resolved_metadata": resolved_metadata,
        },
    )


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
        .groupby(["bin", "label"], observed=False)
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


def run_report(
    df_summary: pd.DataFrame, *, run_metadata: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Compute a JSON-ready report from a summary DataFrame."""
    total = len(df_summary)
    report: Dict[str, object] = {"total": total}
    metadata = run_metadata or _metadata_from_summary(df_summary)
    if metadata:
        report.update(metadata)
    if total == 0:
        return report

    if "prediction" in df_summary.columns or "is_correct" in df_summary.columns:
        report["greedy_accuracy"] = accuracy_greedy(df_summary)
        return report

    if "leader_correct" not in df_summary.columns:
        raise ValueError("DataFrame does not contain leader_correct.")
    if "max_frac" not in df_summary.columns:
        raise ValueError("DataFrame does not contain max_frac.")
    if "branch_preds" not in df_summary.columns or "gold" not in df_summary.columns:
        raise ValueError("DataFrame does not contain branch_preds/gold.")

    report["leader_accuracy"] = accuracy_leader(df_summary)
    report["unanimous"] = unanimous_stats(df_summary)
    report["near_unanimous"] = near_unanimous_stats(df_summary)

    branch_preds = cast(pd.Series, df_summary["branch_preds"])
    gold_series = cast(pd.Series, df_summary["gold"])
    top2_hits = sum(
        1
        for preds, gold in zip(branch_preds, gold_series, strict=False)
        if isinstance(gold, str) and metrics_top2_coverage(preds, gold)
    )
    report["top2_coverage"] = {
        "count": top2_hits,
        "rate": top2_hits / total if total else 0.0,
    }

    max_frac = cast(pd.Series, df_summary["max_frac"])
    leader_correct = cast(pd.Series, df_summary["leader_correct"])
    r_w_other_counts = {"R": 0, "W": 0, "Other": 0}
    for mf, lc in zip(max_frac, leader_correct, strict=False):
        lc_value = None if pd.isna(lc) else bool(lc)
        mf_value = float(mf) if pd.notna(mf) else float("nan")
        label = r_w_other_class(mf_value, lc_value)
        r_w_other_counts[label] += 1
    report["r_w_other"] = r_w_other_counts
    return report


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


def _ranked_top_k(preds: Sequence[Optional[str]], k: int = 2) -> List[str]:
    valid = [
        pred
        for pred in preds
        if isinstance(pred, str) and pred.strip() and not pd.isna(pred)
    ]
    if not valid:
        return []
    first_idx: Dict[str, int] = {}
    for i, pred in enumerate(preds):
        if not isinstance(pred, str) or not pred.strip() or pd.isna(pred):
            continue
        if pred not in first_idx:
            first_idx[pred] = i
    counts = Counter(valid)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], first_idx[kv[0]]))
    return [label for label, _ in ranked[:k]]


def failure_mode_breakdown(
    df_branches: pd.DataFrame,
    top_k: int = 2,
    near_threshold: float = 0.9,
) -> Dict[str, float | int]:
    """Compute failure-mode stats for selection vs unsurfaced errors and unanimity."""
    required_cols = {"gold", "leader_correct", "max_frac", "branch_preds"}
    missing = required_cols - set(df_branches.columns)
    if missing:
        raise ValueError(f"DataFrame does not contain required columns: {missing}")

    gold_series = cast(pd.Series, df_branches["gold"])
    branch_series = cast(pd.Series, df_branches["branch_preds"])
    leader_correct = cast(pd.Series, df_branches["leader_correct"]).fillna(False).astype(bool)
    leader_wrong = ~leader_correct

    gold_in_topk = [
        gold in _ranked_top_k(preds, k=top_k) if isinstance(gold, str) else False
        for preds, gold in zip(branch_series, gold_series, strict=False)
    ]

    n_total = len(df_branches)
    n_errors = int(leader_wrong.sum())
    selection_mask = leader_wrong & pd.Series(gold_in_topk, index=df_branches.index)
    unsurfaced_mask = leader_wrong & ~pd.Series(gold_in_topk, index=df_branches.index)

    sel_n = int(selection_mask.sum())
    uns_n = int(unsurfaced_mask.sum())

    def _pct(num: int, denom: int) -> float:
        return 0.0 if denom == 0 else 100.0 * num / denom

    sel_total_pct = _pct(sel_n, n_total)
    sel_error_pct = _pct(sel_n, n_errors)
    uns_total_pct = _pct(uns_n, n_total)
    uns_error_pct = _pct(uns_n, n_errors)

    max_frac = cast(pd.Series, df_branches["max_frac"]).fillna(0.0)
    unanimous_mask = max_frac == 1.0
    unanim_n = int(unanimous_mask.sum())
    unanim_wrong_n = int((unanimous_mask & leader_wrong).sum())
    unanim_wrong_pct = _pct(unanim_wrong_n, unanim_n)
    unanim_acc_pct = 100.0 - unanim_wrong_pct if unanim_n else 0.0
    unanim_share_errors_pct = _pct(unanim_wrong_n, n_errors)

    near_mask = max_frac >= near_threshold
    near_n = int(near_mask.sum())
    near_wrong_n = int((near_mask & leader_wrong).sum())
    near_wrong_pct = _pct(near_wrong_n, near_n)

    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "selection_errors": sel_n,
        "selection_total_pct": sel_total_pct,
        "selection_error_pct": sel_error_pct,
        "unsurfaced_errors": uns_n,
        "unsurfaced_total_pct": uns_total_pct,
        "unsurfaced_error_pct": uns_error_pct,
        "unanimous_n": unanim_n,
        "unanimous_wrong_n": unanim_wrong_n,
        "unanimous_wrong_pct": unanim_wrong_pct,
        "unanimous_acc_pct": unanim_acc_pct,
        "unanimous_error_pct": unanim_share_errors_pct,
        "near_unanimous_n": near_n,
        "near_unanimous_wrong_n": near_wrong_n,
        "near_unanimous_wrong_pct": near_wrong_pct,
    }


@dataclass(frozen=True)
class VotePatternSpec:
    """Configuration for synthetic vote-pattern examples."""

    top1_votes: int
    top2_votes: int
    examples: int
    top1_correct: int
    top2_correct: int
    tie_top2: int = 0


def _valid_predictions(preds: Iterable[Optional[str]]) -> List[str]:
    valid = [
        pred
        for pred in preds
        if isinstance(pred, str) and pred.strip() and not pd.isna(pred)
    ]
    return valid


def _top_two_counts(counter: Counter) -> Tuple[List[str], int, List[str], int, int]:
    if not counter:
        return [], 0, [], 0, 0
    top1_count = max(counter.values())
    top1_labels = [label for label, count in counter.items() if count == top1_count]
    remaining = {label: count for label, count in counter.items() if label not in top1_labels}
    if not remaining:
        return top1_labels, top1_count, [], 0, 0
    top2_count = max(remaining.values())
    top2_labels = [label for label, count in remaining.items() if count == top2_count]
    remaining_after_top2 = {
        label: count for label, count in remaining.items() if label not in top2_labels
    }
    top3_count = max(remaining_after_top2.values()) if remaining_after_top2 else 0
    return top1_labels, top1_count, top2_labels, top2_count, top3_count


def compute_top2_flip_matrix(
    df_branches: pd.DataFrame,
    strict: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Compute contingency table for (top1_votes, top2_votes) patterns."""
    if "branch_preds" not in df_branches.columns or "gold" not in df_branches.columns:
        raise ValueError("DataFrame does not contain branch_preds/gold.")

    branch_preds = cast(pd.Series, df_branches["branch_preds"])
    gold_series = cast(pd.Series, df_branches["gold"])
    max_votes = max((len(preds) for preds in branch_preds), default=0)

    counts: Dict[Tuple[int, int], Dict[str, int]] = {}
    tie_stats = {
        "total_examples": len(df_branches),
        "included_examples": 0,
        "excluded_tie_top1": 0,
        "excluded_tie_top2": 0,
        "excluded_no_top2": 0,
        "excluded_no_valid": 0,
    }

    for preds, gold in zip(branch_preds, gold_series, strict=False):
        valid = _valid_predictions(preds)
        if not valid:
            tie_stats["excluded_no_valid"] += 1
            continue
        counter = Counter(valid)
        top1_labels, top1_count, top2_labels, top2_count, top3_count = _top_two_counts(counter)
        if len(top1_labels) != 1:
            tie_stats["excluded_tie_top1"] += 1
            if strict:
                continue
        if not top2_labels:
            tie_stats["excluded_no_top2"] += 1
            if strict:
                continue
        if len(top2_labels) != 1 or (top3_count and top3_count == top2_count):
            tie_stats["excluded_tie_top2"] += 1
            if strict:
                continue

        top1_label = top1_labels[0]
        top2_label = top2_labels[0] if top2_labels else ""
        key = (top1_count, top2_count)
        entry = counts.setdefault(
            key,
            {
                "examples_count": 0,
                "top1_correct_count": 0,
                "top2_correct_count": 0,
            },
        )
        entry["examples_count"] += 1
        if isinstance(gold, str) and gold == top1_label:
            entry["top1_correct_count"] += 1
        if isinstance(gold, str) and gold == top2_label:
            entry["top2_correct_count"] += 1
        tie_stats["included_examples"] += 1

    rows = []
    for top1_votes in range(1, max_votes + 1):
        for top2_votes in range(1, max_votes):
            key = (top1_votes, top2_votes)
            entry = counts.get(
                key,
                {
                    "examples_count": 0,
                    "top1_correct_count": 0,
                    "top2_correct_count": 0,
                },
            )
            examples_count = entry["examples_count"]
            top1_correct_count = entry["top1_correct_count"]
            top2_correct_count = entry["top2_correct_count"]
            keep_top1_accuracy = top1_correct_count / examples_count if examples_count else 0.0
            flip_to_top2_accuracy = top2_correct_count / examples_count if examples_count else 0.0
            ratio = (
                top1_correct_count / top2_correct_count
                if top2_correct_count
                else float("inf")
            )
            rows.append(
                {
                    "top1_votes": top1_votes,
                    "top2_votes": top2_votes,
                    "examples_count": examples_count,
                    "top1_correct_count": top1_correct_count,
                    "top2_correct_count": top2_correct_count,
                    "keep_top1_accuracy": keep_top1_accuracy,
                    "flip_to_top2_accuracy": flip_to_top2_accuracy,
                    "ratio_top1_over_top2": ratio,
                }
            )

    matrix = pd.DataFrame(rows)
    return matrix, tie_stats


def aggregate_top2_flip_rectangles(
    matrix: pd.DataFrame,
    min_support: int = 1,
) -> pd.DataFrame:
    """Aggregate vote-pattern rectangles over the matrix grid."""
    if matrix.empty:
        return pd.DataFrame(
            columns=[
                "top1_votes_min",
                "top1_votes_max",
                "top2_votes_min",
                "top2_votes_max",
                "total_examples_count",
                "total_top1_correct_count",
                "total_top2_correct_count",
                "harm_to_benefit_ratio",
                "always_flip_delta_accuracy",
                "random_flip_expected_delta_accuracy",
            ]
        )

    max_top1 = int(matrix["top1_votes"].max())
    max_top2 = int(matrix["top2_votes"].max())
    rows = []
    for top1_min in range(1, max_top1 + 1):
        for top1_max in range(top1_min, max_top1 + 1):
            for top2_min in range(1, max_top2 + 1):
                for top2_max in range(top2_min, max_top2 + 1):
                    subset = matrix[
                        (matrix["top1_votes"].between(top1_min, top1_max))
                        & (matrix["top2_votes"].between(top2_min, top2_max))
                    ]
                    total_examples = int(subset["examples_count"].sum())
                    if total_examples < min_support:
                        continue
                    total_top1 = int(subset["top1_correct_count"].sum())
                    total_top2 = int(subset["top2_correct_count"].sum())
                    ratio = total_top1 / total_top2 if total_top2 else float("inf")
                    always_delta = (
                        (total_top2 - total_top1) / total_examples if total_examples else 0.0
                    )
                    rows.append(
                        {
                            "top1_votes_min": top1_min,
                            "top1_votes_max": top1_max,
                            "top2_votes_min": top2_min,
                            "top2_votes_max": top2_max,
                            "total_examples_count": total_examples,
                            "total_top1_correct_count": total_top1,
                            "total_top2_correct_count": total_top2,
                            "harm_to_benefit_ratio": ratio,
                            "always_flip_delta_accuracy": always_delta,
                            "random_flip_expected_delta_accuracy": 0.5 * always_delta,
                        }
                    )

    rectangles = pd.DataFrame(rows)
    if rectangles.empty:
        return rectangles
    rectangles = rectangles.sort_values(
        ["harm_to_benefit_ratio", "always_flip_delta_accuracy", "total_examples_count"],
        ascending=[True, True, False],
    )
    return rectangles.reset_index(drop=True)


def top2_flip_rectangle_stats(
    matrix: pd.DataFrame,
    *,
    top1_votes_min: int,
    top1_votes_max: int,
    top2_votes_min: int,
    top2_votes_max: int,
) -> pd.DataFrame:
    """Return aggregate stats for a single vote-pattern rectangle."""
    if top2_votes_min < 1:
        raise ValueError("top2_votes_min must be >= 1.")
    subset = matrix[
        (matrix["top1_votes"].between(top1_votes_min, top1_votes_max))
        & (matrix["top2_votes"].between(top2_votes_min, top2_votes_max))
    ]
    total_examples = int(subset["examples_count"].sum())
    total_top1 = int(subset["top1_correct_count"].sum())
    total_top2 = int(subset["top2_correct_count"].sum())
    ratio = total_top1 / total_top2 if total_top2 else float("inf")
    always_delta = (total_top2 - total_top1) / total_examples if total_examples else 0.0
    return pd.DataFrame(
        [
            {
                "top1_votes_min": top1_votes_min,
                "top1_votes_max": top1_votes_max,
                "top2_votes_min": top2_votes_min,
                "top2_votes_max": top2_votes_max,
                "total_examples_count": total_examples,
                "total_top1_correct_count": total_top1,
                "total_top2_correct_count": total_top2,
                "harm_to_benefit_ratio": ratio,
                "always_flip_delta_accuracy": always_delta,
                "random_flip_expected_delta_accuracy": 0.5 * always_delta,
            }
        ]
    )


def best_rectangles_by_ratio_threshold(
    rectangles: pd.DataFrame,
    thresholds: Sequence[float],
    min_support: int = 1,
) -> pd.DataFrame:
    """Select best rectangle per ratio threshold.

    Preference order:
    1) Maximize total_examples_count (support).
    2) Minimize harm_to_benefit_ratio.
    3) Minimize absolute always_flip_delta_accuracy.
    """
    rows = []
    for threshold in thresholds:
        subset = rectangles[
            (rectangles["harm_to_benefit_ratio"] <= threshold)
            & (rectangles["total_examples_count"] >= min_support)
        ]
        if subset.empty:
            rows.append({"ratio_threshold": threshold})
            continue
        best = subset.sort_values(
            [
                "total_examples_count",
                "harm_to_benefit_ratio",
                "always_flip_delta_accuracy",
            ],
            ascending=[False, True, True],
        ).iloc[0]
        row = {"ratio_threshold": threshold}
        row.update(best.to_dict())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    filtered = []
    prev_signature = None
    for row in rows:
        signature = {key: value for key, value in row.items() if key != "ratio_threshold"}
        if signature != prev_signature:
            filtered.append(row)
            prev_signature = signature
    return pd.DataFrame(filtered)


def top2_flip_playground(
    df_branches: pd.DataFrame,
    *,
    top1_votes_min: int,
    top1_votes_max: int,
    top2_votes_min: int,
    top2_votes_max: int,
    strict: bool = True,
) -> pd.DataFrame:
    """Compute top-2 flip stats for a specified vote range."""
    matrix, _ = compute_top2_flip_matrix(df_branches, strict=strict)
    return top2_flip_rectangle_stats(
        matrix,
        top1_votes_min=top1_votes_min,
        top1_votes_max=top1_votes_max,
        top2_votes_min=top2_votes_min,
        top2_votes_max=top2_votes_max,
    )


def top2_flip_playground_relative(
    df_branches: pd.DataFrame,
    *,
    top1_votes_min: int,
    top1_votes_max: int,
    gap_min: int,
    gap_max: int,
    strict: bool = True,
) -> pd.DataFrame:
    """Compute top-2 flip stats where top2 votes are relative to top1."""
    if gap_min < 0 or gap_max < gap_min:
        raise ValueError("gap_min must be >= 0 and gap_max must be >= gap_min.")
    matrix, _ = compute_top2_flip_matrix(df_branches, strict=strict)
    subset = matrix[
        (matrix["top1_votes"].between(top1_votes_min, top1_votes_max))
        & (matrix["top2_votes"] <= matrix["top1_votes"])
        & ((matrix["top1_votes"] - matrix["top2_votes"]).between(gap_min, gap_max))
    ]
    total_examples = int(subset["examples_count"].sum())
    total_top1 = int(subset["top1_correct_count"].sum())
    total_top2 = int(subset["top2_correct_count"].sum())
    ratio = total_top1 / total_top2 if total_top2 else float("inf")
    always_delta = (total_top2 - total_top1) / total_examples if total_examples else 0.0
    return pd.DataFrame(
        [
            {
                "top1_votes_min": top1_votes_min,
                "top1_votes_max": top1_votes_max,
                "gap_min": gap_min,
                "gap_max": gap_max,
                "total_examples_count": total_examples,
                "total_top1_correct_count": total_top1,
                "total_top2_correct_count": total_top2,
                "harm_to_benefit_ratio": ratio,
                "always_flip_delta_accuracy": always_delta,
                "random_flip_expected_delta_accuracy": 0.5 * always_delta,
            }
        ]
    )


def top2_flip_analysis(
    df_branches: pd.DataFrame,
    *,
    strict: bool = True,
    min_support: int = 1,
    ratio_thresholds: Sequence[float] = tuple(round(1.0 + 0.1 * i, 1) for i in range(41)),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Run full top-2 flip subset discovery analysis."""
    matrix, tie_stats = compute_top2_flip_matrix(df_branches, strict=strict)
    rectangles = aggregate_top2_flip_rectangles(matrix, min_support=min_support)
    threshold_rectangles = best_rectangles_by_ratio_threshold(
        rectangles, ratio_thresholds, min_support=min_support
    )
    return matrix, rectangles, threshold_rectangles, tie_stats


def aggregate_top2_flip_relative_rectangles(
    matrix: pd.DataFrame,
    min_support: int = 1,
) -> pd.DataFrame:
    """Aggregate vote-pattern rectangles using relative (gap) ranges."""
    if matrix.empty:
        return pd.DataFrame(
            columns=[
                "top1_votes_min",
                "top1_votes_max",
                "gap_min",
                "gap_max",
                "total_examples_count",
                "total_top1_correct_count",
                "total_top2_correct_count",
                "harm_to_benefit_ratio",
                "always_flip_delta_accuracy",
                "random_flip_expected_delta_accuracy",
            ]
        )

    max_top1 = int(matrix["top1_votes"].max())
    rows = []
    for top1_min in range(1, max_top1 + 1):
        for top1_max in range(top1_min, max_top1 + 1):
            for gap_min in range(0, max_top1 + 1):
                for gap_max in range(gap_min, max_top1 + 1):
                    subset = matrix[
                        (matrix["top1_votes"].between(top1_min, top1_max))
                        & (matrix["top2_votes"] <= matrix["top1_votes"])
                        & ((matrix["top1_votes"] - matrix["top2_votes"]).between(gap_min, gap_max))
                    ]
                    total_examples = int(subset["examples_count"].sum())
                    if total_examples < min_support:
                        continue
                    total_top1 = int(subset["top1_correct_count"].sum())
                    total_top2 = int(subset["top2_correct_count"].sum())
                    ratio = total_top1 / total_top2 if total_top2 else float("inf")
                    always_delta = (
                        (total_top2 - total_top1) / total_examples if total_examples else 0.0
                    )
                    rows.append(
                        {
                            "top1_votes_min": top1_min,
                            "top1_votes_max": top1_max,
                            "gap_min": gap_min,
                            "gap_max": gap_max,
                            "total_examples_count": total_examples,
                            "total_top1_correct_count": total_top1,
                            "total_top2_correct_count": total_top2,
                            "harm_to_benefit_ratio": ratio,
                            "always_flip_delta_accuracy": always_delta,
                            "random_flip_expected_delta_accuracy": 0.5 * always_delta,
                        }
                    )

    rectangles = pd.DataFrame(rows)
    if rectangles.empty:
        return rectangles
    rectangles = rectangles.sort_values(
        ["harm_to_benefit_ratio", "always_flip_delta_accuracy", "total_examples_count"],
        ascending=[True, True, False],
    )
    return rectangles.reset_index(drop=True)


def top2_flip_analysis_relative(
    df_branches: pd.DataFrame,
    *,
    strict: bool = True,
    min_support: int = 1,
    ratio_thresholds: Sequence[float] = tuple(round(1.0 + 0.1 * i, 1) for i in range(41)),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Run top-2 flip subset discovery using gap-based rectangles."""
    matrix, tie_stats = compute_top2_flip_matrix(df_branches, strict=strict)
    rectangles = aggregate_top2_flip_relative_rectangles(matrix, min_support=min_support)
    threshold_rectangles = best_rectangles_by_ratio_threshold(
        rectangles, ratio_thresholds, min_support=min_support
    )
    return matrix, rectangles, threshold_rectangles, tie_stats


def top2_flip_analysis_relative_with_plot(
    df_branches: pd.DataFrame,
    *,
    baseline_acc: float,
    total_n: int = 400,
    strict: bool = True,
    min_support: int = 1,
    ratio_thresholds: Sequence[float] = tuple(round(1.0 + 0.1 * i, 1) for i in range(41)),
    use_frontier_df: Optional[pd.DataFrame] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], object, object, pd.DataFrame
]:
    """Run gap-based top-2 flip analysis and plot feasibility in one step."""
    matrix, rectangles, threshold_rectangles, tie_stats = top2_flip_analysis_relative(
        df_branches,
        strict=strict,
        min_support=min_support,
        ratio_thresholds=ratio_thresholds,
    )
    from rofa.analysis.plots import plot_top2_flip_feasibility

    plot_source = use_frontier_df if use_frontier_df is not None else threshold_rectangles
    fig, ax, plot_df = plot_top2_flip_feasibility(
        plot_source, baseline_acc, total_n, use_frontier_df=None
    )
    return matrix, rectangles, threshold_rectangles, tie_stats, fig, ax, plot_df


def make_gap_neighbor_rows(
    df_branches: pd.DataFrame,
    optimal_df: pd.DataFrame,
    *,
    include_gap_min_neighbors: bool = False,
    gap_step: int = 1,
) -> pd.DataFrame:
    """Build a comparison table with nearby gap-range variants."""
    required = {"top1_votes_min", "top1_votes_max", "gap_min", "gap_max"}
    missing = required - set(optimal_df.columns)
    if missing:
        raise ValueError(f"optimal_df is missing columns: {missing}")

    optimal_param_set = set(
        tuple(map(int, row))
        for row in optimal_df[
            ["top1_votes_min", "top1_votes_max", "gap_min", "gap_max"]
        ].itertuples(index=False, name=None)
    )

    def eval_params(t1min: int, t1max: int, gmin: int, gmax: int) -> pd.DataFrame:
        out = top2_flip_playground_relative(
            df_branches,
            top1_votes_min=t1min,
            top1_votes_max=t1max,
            gap_min=gmin,
            gap_max=gmax,
        )
        if not isinstance(out, pd.DataFrame) or out.empty:
            return pd.DataFrame(
                [
                    {
                        "top1_votes_min": t1min,
                        "top1_votes_max": t1max,
                        "gap_min": gmin,
                        "gap_max": gmax,
                        "total_examples_count": 0,
                    }
                ]
            )
        for key, value in {
            "top1_votes_min": t1min,
            "top1_votes_max": t1max,
            "gap_min": gmin,
            "gap_max": gmax,
        }.items():
            if key not in out.columns:
                out[key] = value
        return out

    rows = []
    opt_params = optimal_df[
        ["top1_votes_min", "top1_votes_max", "gap_min", "gap_max"]
    ].copy()
    for idx, row in opt_params.iterrows():
        t1min = int(row["top1_votes_min"])
        t1max = int(row["top1_votes_max"])
        gmin = int(row["gap_min"])
        gmax = int(row["gap_max"])

        base = eval_params(t1min, t1max, gmin, gmax)
        base["variant"] = "optimal"
        base["source_row"] = idx
        rows.append(base)

        for delta in (-gap_step, +gap_step):
            ngmax = gmax + delta
            if ngmax < gmin:
                continue
            cand = (t1min, t1max, gmin, ngmax)
            if cand in optimal_param_set:
                continue
            df_variant = eval_params(t1min, t1max, gmin, ngmax)
            df_variant["variant"] = f"gap_max{delta:+d}"
            df_variant["source_row"] = idx
            rows.append(df_variant)

        if include_gap_min_neighbors:
            for delta in (-gap_step, +gap_step):
                ngmin = gmin + delta
                if ngmin < 0 or ngmin > gmax:
                    continue
                cand = (t1min, t1max, ngmin, gmax)
                if cand in optimal_param_set:
                    continue
                df_variant = eval_params(t1min, t1max, ngmin, gmax)
                df_variant["variant"] = f"gap_min{delta:+d}"
                df_variant["source_row"] = idx
                rows.append(df_variant)

    out = pd.concat(rows, ignore_index=True)
    key_cols = [
        "harm_to_benefit_ratio",
        "always_flip_delta_accuracy",
        "random_flip_expected_delta_accuracy",
        "total_examples_count",
    ]
    for col in key_cols:
        if col in out.columns:
            base_map = (
                out[out["variant"] == "optimal"][["source_row", col]]
                .set_index("source_row")[col]
            )
            out[f"delta_{col}"] = out.apply(
                lambda x, col=col: x[col] - base_map.get(x["source_row"], float("nan")),
                axis=1,
            )

    variant_order = {
        "optimal": 0,
        f"gap_max{-gap_step:+d}": 1,
        f"gap_max{+gap_step:+d}": 2,
        f"gap_min{-gap_step:+d}": 3,
        f"gap_min{+gap_step:+d}": 4,
    }
    out["variant_rank"] = out["variant"].map(variant_order).fillna(99).astype(int)
    out = out.sort_values(
        ["source_row", "variant_rank", "gap_max", "gap_min"]
    ).drop(columns=["variant_rank"])
    return out


def generate_synthetic_vote_pattern_dataset(
    patterns: Sequence[VotePatternSpec],
    *,
    n_branches: int = 10,
    seed: int = 0,
    labels: Sequence[str] = ("A", "B", "C", "D"),
) -> pd.DataFrame:
    """Generate synthetic data for vote-pattern testing."""
    if len(labels) < 4:
        raise ValueError("Provide at least four distinct labels.")
    rows: List[Dict[str, object]] = []
    for pattern in patterns:
        if pattern.examples <= 0:
            continue
        if pattern.top1_votes + pattern.top2_votes > n_branches:
            raise ValueError("top1_votes + top2_votes exceeds n_branches.")
        if pattern.top1_votes + 2 * pattern.top2_votes > n_branches and pattern.tie_top2:
            raise ValueError("top1_votes + 2 * top2_votes exceeds n_branches for tie cases.")
        if pattern.tie_top2 > pattern.examples:
            raise ValueError("tie_top2 exceeds examples.")
        non_tie_examples = pattern.examples - pattern.tie_top2
        if pattern.top1_correct + pattern.top2_correct > non_tie_examples:
            raise ValueError("top1_correct + top2_correct exceeds non-tie examples.")

        top1_label, top2_label, other_label, spare_label = labels[:4]
        remainder = n_branches - pattern.top1_votes - pattern.top2_votes
        if remainder < 0:
            raise ValueError("Invalid remainder for votes.")
        if non_tie_examples > 0 and remainder >= pattern.top2_votes and pattern.top2_votes > 0:
            raise ValueError("Non-tie examples would create a top-2 tie.")

        for idx in range(pattern.examples):
            is_tie = idx < pattern.tie_top2
            if is_tie:
                preds = (
                    [top1_label] * pattern.top1_votes
                    + [top2_label] * pattern.top2_votes
                    + [other_label] * pattern.top2_votes
                )
                leftover = n_branches - len(preds)
                if leftover < 0:
                    raise ValueError("Tie example exceeds n_branches.")
                preds += [spare_label] * leftover
                gold = top2_label
            else:
                if idx < pattern.top1_correct + pattern.tie_top2:
                    gold = top1_label
                elif idx < pattern.top1_correct + pattern.top2_correct + pattern.tie_top2:
                    gold = top2_label
                else:
                    gold = spare_label
                preds = (
                    [top1_label] * pattern.top1_votes
                    + [top2_label] * pattern.top2_votes
                    + [other_label] * remainder
                )

            if len(preds) != n_branches:
                raise ValueError("Generated predictions do not match n_branches.")
            rng = Random(seed + idx)
            rng.shuffle(preds)
            shuffled = preds
            rows.append({"gold": gold, "branch_preds": shuffled})

    return pd.DataFrame(rows)
