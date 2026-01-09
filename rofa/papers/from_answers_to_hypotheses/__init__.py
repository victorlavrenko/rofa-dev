"""Paper package for 'From Answers to Hypotheses'."""

from rofa.papers.from_answers_to_hypotheses.analysis import load_paper_runs, paper_metrics
from rofa.papers.from_answers_to_hypotheses.config import (
    PAPER_ID,
    PAPER_SPEC,
    PAPER_TITLE,
)


def default_configs() -> dict:
    """Return default paper configuration values."""
    return {
        "paper_id": PAPER_ID,
        "title": PAPER_TITLE,
        "default_dataset": PAPER_SPEC.default_dataset,
        "default_split": PAPER_SPEC.default_split,
        "methods": list(PAPER_SPEC.methods.keys()),
    }


def run_paper_analysis(run_dirs_or_zips):
    """Run paper analysis from run directories or zip paths."""
    df_greedy, df_branches, metadata = load_paper_runs(run_dirs_or_zips)
    return {
        "metadata": metadata,
        "greedy": paper_metrics(df_greedy),
        "k_sample_ensemble": paper_metrics(df_branches),
    }


__all__ = [
    "PAPER_ID",
    "PAPER_SPEC",
    "PAPER_TITLE",
    "default_configs",
    "run_paper_analysis",
]
