import pandas as pd

from rofa.analysis.plots import plot_top2_flip_feasibility


def test_plot_top2_flip_feasibility_metrics() -> None:
    regimes = pd.DataFrame(
        [
            {
                "top1_votes_min": 7,
                "top1_votes_max": 8,
                "gap_min": 1,
                "gap_max": 2,
                "total_examples_count": 10,
                "total_top1_correct_count": 6,
                "total_top2_correct_count": 2,
            },
            {
                "top1_votes_min": 5,
                "top1_votes_max": 6,
                "gap_min": 2,
                "gap_max": 3,
                "total_examples_count": 20,
                "total_top1_correct_count": 10,
                "total_top2_correct_count": 5,
            },
            {
                "top1_votes_min": 9,
                "top1_votes_max": 9,
                "gap_min": 0,
                "gap_max": 0,
                "total_examples_count": 3,
                "total_top1_correct_count": 3,
                "total_top2_correct_count": 0,
            },
        ]
    )

    baseline_acc = 0.5
    total_n = 100
    _, _, plot_df = plot_top2_flip_feasibility(regimes, baseline_acc, total_n)

    assert plot_df.shape[0] == 2

    first = plot_df.iloc[0]
    assert first["required_fp_suppression"] == 6 / 2
    assert first["oracle_overall_acc"] == baseline_acc + (2 / total_n)
    assert first["wrong_share_pct"] == 100 * (10 - 6) / ((1 - baseline_acc) * total_n)

    second = plot_df.iloc[1]
    assert second["required_fp_suppression"] == 10 / 5
    assert second["oracle_overall_acc"] == baseline_acc + (5 / total_n)
    assert second["wrong_share_pct"] == 100 * (20 - 10) / ((1 - baseline_acc) * total_n)
