"""Plotting utilities for ROFA analysis."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

_REQUIRED_COLUMNS = {
    "top1_votes_min",
    "top1_votes_max",
    "gap_min",
    "gap_max",
    "total_examples_count",
    "total_top1_correct_count",
    "total_top2_correct_count",
}


def _scale_bubble_sizes(baseline_wrong: Iterable[float]) -> pd.Series:
    sizes = pd.Series(baseline_wrong, dtype=float).clip(lower=1)
    if sizes.empty:
        return sizes
    percentile_95 = float(sizes.quantile(0.95))
    if percentile_95 <= 0:
        percentile_95 = float(sizes.max())
    sizes = sizes.clip(upper=percentile_95 if percentile_95 > 0 else sizes.max())
    median = float(sizes.median())
    if median <= 0:
        median = 1.0
    sizes = (sizes / median) * 450.0
    sizes = sizes * 3
    sizes = sizes.clip(lower=160.0, upper=2400.0)
    return sizes


def plot_top2_flip_feasibility(
    regimes_df: pd.DataFrame,
    baseline_acc: float,
    total_n: int = 400,
    *,
    use_frontier_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = "top2_flip_feasibility.png",
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot selective top-2 flip feasibility for regime rectangles.

    Returns the matplotlib figure/axes and the DataFrame used for plotting.
    """
    df_source = use_frontier_df if use_frontier_df is not None else regimes_df
    missing = _REQUIRED_COLUMNS - set(df_source.columns)
    if missing:
        raise ValueError(f"regimes_df missing required columns: {sorted(missing)}")

    plot_df = df_source.copy()
    plot_df["baseline_wrong"] = (
        plot_df["total_examples_count"] - plot_df["total_top1_correct_count"]
    )

    plot_df = plot_df[plot_df["total_top2_correct_count"] > 0].copy()
    if plot_df.empty:
        raise ValueError("No regimes with total_top2_correct_count > 0 to plot.")

    plot_df["required_fp_suppression"] = (
        plot_df["total_top1_correct_count"] / plot_df["total_top2_correct_count"]
    )
    plot_df["oracle_overall_acc"] = baseline_acc + (
        plot_df["total_top2_correct_count"] / total_n
    )

    total_wrong_overall = (1.0 - baseline_acc) * total_n
    if total_wrong_overall > 0:
        plot_df["wrong_share_pct"] = (
            100.0 * plot_df["baseline_wrong"] / total_wrong_overall
        )
    else:
        plot_df["wrong_share_pct"] = 0.0

    plot_df["label"] = plot_df.apply(
        lambda row: (
            "top1["
            f"{int(row['top1_votes_min'])}-{int(row['top1_votes_max'])}], "
            "gap["
            f"{int(row['gap_min'])}-{int(row['gap_max'])}]\n"
            f"{row['wrong_share_pct']:.1f}% of all baseline mistakes"
        ),
        axis=1,
    )

    bubble_sizes = _scale_bubble_sizes(plot_df["baseline_wrong"])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(
        plot_df["oracle_overall_acc"],
        plot_df["required_fp_suppression"],
        s=bubble_sizes,
    )
    ax.scatter(
        [baseline_acc],
        [1.0],
        marker="D",
        s=160,
        label="do nothing (baseline)",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")

    for row, bubble_size in zip(plot_df.itertuples(index=False), bubble_sizes):
        bubble_radius = (float(bubble_size) ** 0.5) / 2.0
        offset_x = -(bubble_radius + 3.0)
        ax.annotate(
            row.label,
            (row.oracle_overall_acc, row.required_fp_suppression),
            textcoords="offset points",
            xytext=(offset_x, -7),
            ha="right",
            fontsize=8,
        )

    ax.set_title("Achievable accuracy vs required false-override suppression")
    ax.set_xlabel("Oracle overall accuracy (ideal top-2 alternative selection)")
    ax.set_ylabel("Required FP suppression (X / Y)")
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    table_cols = [
        "top1_votes_min",
        "top1_votes_max",
        "gap_min",
        "gap_max",
        "total_examples_count",
        "total_top1_correct_count",
        "total_top2_correct_count",
        "baseline_wrong",
        "wrong_share_pct",
        "oracle_overall_acc",
        "required_fp_suppression",
        "always_flip_delta_accuracy",
    ]
    available_cols = [col for col in table_cols if col in plot_df.columns]
    table = plot_df.sort_values("oracle_overall_acc", ascending=False).head(20)
    print(table[available_cols].to_string(index=False))

    return fig, ax, plot_df
